# Step-3.5-Flash → XTuner integration design

Source: `stepfun-ai/Step-3.5-Flash`, `model_type = "step3p5"`, remote-code
(`modeling_step3p5.py` / `configuration_step3p5.py`, `architectures = ["Step3p5ForCausalLM"]`).
bf16 checkpoint, training-suitable.

Bucket: **MoE LLM, trust_remote_code** (no built-in `transformers` config class →
`hf_config` returns `None`; `save_hf` copies `config.json` / tokenizer / `*.py`).

## 1. Architecture summary (from the HF modeling code)

- 45 transformer layers. Layers 0–2 are **dense** MLP; layers 3–44 are **MoE**.
- Vocab 128896, hidden 4096, untied `lm_head` (separate tensor in the index).
- MoE: 288 routed experts, top-k 8, `moe_intermediate_size` 1280, plus **one shared
  expert** (`share_expert_dim` 1280). Experts stored as **fused 3-D** tensors
  `(num_experts, out, in)` — `moe.{gate_proj,up_proj}.weight (288,1280,4096)`,
  `moe.down_proj.weight (288,4096,1280)`.
- Router: **sigmoid** activation, per-expert **`router_bias`** added before top-k,
  weights gathered from the **pre-bias** probabilities, renormalized, then scaled by
  `moe_router_scaling_factor = 3.0`; router logits computed in **fp32**
  (`need_fp32_gate`).
- RMSNorm is **zero-centered** (scale = `weight + 1`), eps 1e-5, throughout.
- Attention is a **hybrid of two softmax-attention profiles keyed by `layer_types`**:
  - `full_attention` (every 4th layer, idx % 4 == 0): 64 heads, 8 KV heads, head_dim 128.
  - `sliding_attention` (the other layers): **96 heads**, 8 KV heads, head_dim 128,
    sliding window 512.
  - Both profiles: `qk_norm` on head_dim (zero-centered), and a **head-wise output
    gate** — a separate `g_proj: Linear(hidden, num_heads)` whose per-head sigmoid
    multiplies the attention output before `o_proj`.
- RoPE differs **per profile**:
  - full_attention: `rope_theta = 5e6`, `partial_rotary_factor = 0.5`, **llama3**
    scaling (`yarn_only_types = ["full_attention"]`).
  - sliding_attention: `rope_theta = 1e4`, `partial_rotary_factor = 1.0`, default rope
    (no scaling).
- swiglu clamp limits on a few late layers only (MoE layers 43,44 → 7; shared expert
  layer 44 → 16; all others 0/None).
- MTP: `num_nextn_predict_layers = 3`, stored as `model.layers.45/46/47.*` and ignored
  on load by HF. We **drop MTP** for the port (load layers 0–44, `strict=False`).

## 2. Mapping to XTuner — what is reused vs. new

Reused as-is (config wiring only):

| Feature | XTuner mechanism |
|---|---|
| first 3 dense + rest MoE | `MoEConfig.first_k_dense_replace = 3` |
| 288 experts / top-8 / shared expert | `n_routed_experts`, `num_experts_per_tok`, `n_shared_experts = 1` |
| expert tensors | **split / per-expert checkpoint** (`.dev_scripts/convert_step3p5_to_split.py`) → `to_hf_key_list` emits interleaved `[gate_i, up_i, …]` (Qwen3-MoE style); default load/save shards on FSDP+EP. See "Expert layout" below. |
| sigmoid + router_bias + renorm + scale 3.0 + fp32 gate | `NoAuxRouterConfig(scoring_func="sigmoid", n_group=1, topk_group=1, norm_topk_prob=True, router_scaling_factor=3.0)` + `router_compute_dtype="float32"` (its math matches HF `router_bias_func`) |
| zero-centered RMSNorm, qk_norm | `rms_norm_type="zero_centered"`, `MHAConfig.qk_norm=True` |
| sliding window in training | `MHAConfig.sliding_window` + `layer_type="sliding_attention"` |
| partial rotary | `RopeParametersConfig.partial_rotary_factor` (already supported) |
| MTP | `mtp_config=None`, load `strict=False` |

Three things the current design **cannot express** — these are the design forks:

### Fork A — head-wise attention gate (new MHA option)

XTuner's existing `MHAConfig.with_gate` is a **per-(head,dim) element** gate fused into a
doubled `q_proj` (Qwen3.5 / gpt-oss style). Step-3.5 uses a **separate `g_proj` of shape
`(num_heads, hidden)`** producing **one scalar per head**. Different weight layout and
different broadcast. Proposal: add a new, general option to `MHAConfig`
(`head_gate: bool = False`) that builds `self.g_proj = Linear(hidden, num_heads,
bias=False)` and applies `out.view(...,H,Dh) * g.sigmoid().unsqueeze(-1)` before `o_proj`.
HF `self_attn.g_proj.weight` maps 1→1 to xtuner `self_attn.g_proj.weight`. This is a small,
self-contained addition at the module layer; its own commit.

### Fork B — two attention profiles with different head counts (CONFIRMED)

Today a model has a single `attention: MHAConfig` shared by all full/sliding layers
(`linear_attention` is GatedDeltaNet-only). Step-3.5 needs **full=64 heads,
sliding=96 heads**. Decision (user): the existing `layers_type` mechanism already supports
mixed attention; add a `sliding_attention: MHAConfig` field on the **Step-3.5 config
only** and **override `build_layers`** in the Step-3.5 model to select the per-layer
attention config by `layers_type`. No change to `base.py` / `MoEConfig`.
(`num_attention_heads` etc. computed fields keep reading the `full` profile — fine for
training; kv-cache / generate with mixed head counts is out of scope for the baseline.)

### Fork C — per-profile RoPE (CONFIRMED — contain in Step-3.5 decoder layer)

Today the model builds one `self.rotary_emb` and passes one `(cos, sin)` to every layer.
Step-3.5 needs different `(theta, partial_rotary, scaling)` for full vs sliding layers.
Decision (user): keep it **inside the Step-3.5 decoder layer** for now; generalize later
once precision is aligned. Each Step-3.5 decoder layer holds its own
`Step3p5RotaryEmbedding` (full: theta 5e6 / partial 0.5 / **llama3** scaling; sliding:
theta 1e4 / partial 1.0 / default) built faithfully via HF's
`ROPE_INIT_FUNCTIONS[rope_type]` on a per-profile shim so inv_freq matches HF bitwise. Its
`forward` recomputes `position_embeddings` from `seq_ctx.position_ids` and passes them to
`self.self_attn`, **ignoring** the model-level `(cos,sin)` (which stays valid but unused).
No change to shared `MoE.forward` or `MultiHeadAttention.forward`. The per-layer partial-rotary
apply is set **directly on the attention** in the decoder-layer `__init__`
(`self.self_attn.apply_rotary_emb = get_apply_rotary_emb(None, enable_partial_rotary=…)`) rather than
threading a rope config through `build` — this keeps the per-layer RoPE fully contained and avoids the
deprecated `RopeScalingConfig` (whose only consumer in `MultiHeadAttention` is that apply selection).

### Swiglu clamp (CONFIRMED — include now)

Step clamps `silu(gate).clamp(max=limit) * up.clamp(±limit)` on a few late layers
(routed experts L43/L44 → 7; shared expert L44 → 16; all others none). XTuner's existing
`clipped_swiglu` is gpt-oss-shaped (sigmoid-GLU + `(up+1)`) and does **not** match. Add a
new act variant `swiglu_clip` (silu + post-activation clamp) to `act_fn.py` /
`MoEActFnConfig`, and build a **per-layer** `MoEActFnConfig` (clip only where the config
lists a nonzero limit) in `build_layers`. For the shared expert clamp (L44), thread an
optional clamp limit into the shared-expert MLP. MTP (3 nextn layers) remains deferred
(load layers 0–44, `strict=False`).

### Expert layout — split / per-expert checkpoint (CONFIRMED)

The released checkpoint stores each MoE layer's experts as **three fused 3-D tensors**
(`moe.{gate,up,down}_proj.weight`, `(num_experts, *, *)`). XTuner fuses experts
**expert-major-interleaved** (`[g0,u0,g1,u1,…]`, each expert's `[gate;up]` contiguous — required by the
grouped GEMM). XTuner's loader can only shard a fused parameter when each HF key is a *contiguous*
slice of it, so a single fused `w1w3` mapped to two HF tensors (`gate`, `up`) **cannot be sharded**:
a 2-GPU FSDP load was empirically confirmed to crash (`gate, up = safetensors` receives 1 tensor),
and `save_hf` corrupted gate/up (`(288,640,4096)` vs `(288,1280,4096)` — the FUSED save split runs
before `param_to_safetensor`). This is a real XTuner limitation, not specific to this model.

Decision (user): **convert the checkpoint to a split / per-expert layout** offline rather than change
the shared MoE block. `.dev_scripts/convert_step3p5_to_split.py` explodes each fused expert tensor
into `moe.experts.{i}.{gate,up,down}_proj.weight` (Qwen3-MoE style) and drops the unused MTP layers
(45–47). With that layout `to_hf_key_list` emits the interleaved key order `[gate_0, up_0, …]`, which
lines up with the expert-major fused weight, so the **default** `safetensors_to_params` (concat dim 0)
and the default save split both shard correctly on any number of GPUs (FSDP and EP) — **no per-model
checkpoint override and no MoE-block change**. Converted checkpoint:
`/mnt/shared-storage-user/llmrazor-share/yehaochen/model/Step-3.5-Flash-split`.

## 3. File layout

- `xtuner/v1/model/moe/step3p5.py` — `Step3p5MoEConfig` (+ base) and `Step3p5MoE`
  (`to_hf_key_list`, `safetensors_to_params`/`param_to_safetensor`, `build_layers`,
  `build_rotary_embedding`, `hf_config -> None`), plus `Step3p5Attention` +
  `Step3p5RotaryEmbedding` (or co-located in the module layer if cleaner).
- `xtuner/v1/module/attention/mha.py` — Fork A (`head_gate`) + Fork C (unpack hook).
- `xtuner/v1/model/__init__.py` — import / `model_mapping` alias / `get_model_config_from_hf`
  dispatch on `model_type == "step3p5"` / `__all__`.
- `tests/model/test_step3p5_moe.py` — baseline tests (§6/§7 of the skill).
- `ci/config/step3p5.py` — drop-in training config.

## 4. Commit plan (stacked, dependency-ordered)

1. `[Feature]` MHA head-wise gate (`head_gate`) + position-embeddings unpack hook (Forks A & C seam).
2. `[Feature]` Step-3.5 model + config + registration (Forks B & C, router/expert mapping).
3. `[Feature]` `XTUNER_HF_IMPL` parity wiring if any new op branch is needed.
4. `[Test]` baseline tests (decoder-layer bitwise parity for one full + one sliding layer;
   whole-model forward+backward parity at the deployable scale; save_hf round-trip).
5. `[CI]` drop-in training config + convergence trace.
6. Then §8 optimizations (EP/SP, compile, fp8, offload) — multi-agent, post-baseline.
