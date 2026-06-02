# Qwen3.5-4B (dense VLM) integration design

## 1. What the model is

`Qwen/Qwen3.5-4B` — `model_type: "qwen3_5"`, architecture
`Qwen3_5ForConditionalGeneration`. A **vision-language model** whose text tower is
a **dense hybrid** of linear (GatedDeltaNet) and full (gated MHA) attention.

| Tower | Shape |
|-------|-------|
| Vision (`model.visual.*`) | Qwen3VL-style: `patch_embed.proj`, `pos_embed`, `blocks.{N}` (attn qkv/proj, mlp linear_fc1/fc2, norm1/norm2), `merger`. depth 24, hidden 1024, `out_hidden_size` 2560, `num_position_embeddings` 2304, `deepstack_visual_indexes: []` |
| Projector (`model.visual.merger`) | vision_hidden 1024 → text_hidden 2560 |
| Text (`model.language_model.*`) | dense, 32 layers, hidden 2560, head_dim 256, 16 q-heads, 4 kv-heads, `intermediate_size` 9216 |

Text per-layer attention follows `full_attention_interval = 4`: layers where
`(i+1) % 4 == 0` (3, 7, 11, …) are `full_attention`; the rest are
`linear_attention`.

- **full_attention layer**: gated MHA — `q_proj` emits `head_dim*2` then chunks
  into `(query, gate)`, and the attention output is multiplied by
  `sigmoid(gate)` before `o_proj`. Has `q_norm`/`k_norm`. **No sliding window**
  (global attention). → maps to `MHAConfig(with_gate=True, qk_norm=True)`.
- **linear_attention layer**: `Qwen3_5GatedDeltaNet` — `in_proj_qkv`,
  `in_proj_z`, `in_proj_b`, `in_proj_a`, depthwise `conv1d`, `A_log`, `dt_bias`,
  gated RMSNorm `norm`, `out_proj`. → maps to `GatedDeltaNetConfig`.
- **RoPE**: `mrope_interleaved: true`, `mrope_section: [11, 11, 10]`,
  `rope_type: "default"`, `rope_theta: 1e7`, `partial_rotary_factor: 0.25`.
- **MTP**: 1 dense full-attention MTP layer in the checkpoint (15 `mtp.*`
  weights). **Deferred** (see §4).
- `tie_word_embeddings: true`.

## 2. Reuse map — what already exists

The MoE sibling (`Qwen/Qwen3.5-…-A3B`) is **already ported**, so almost every
building block exists:

- **Hybrid dispatch** — `Dense.build_layers` (`xtuner/v1/model/dense/dense.py`)
  already selects `config.linear_attention` (`GatedDeltaNetConfig`) for
  `linear_attention` layers and `config.attention` for `full_attention`. No body
  change needed for the hybrid.
- **Gated MHA / GatedDeltaNet / gated-deltanet ops** — exist
  (`xtuner/v1/module/attention/{mha,gated_deltanet}.py`, `xtuner/v1/ops/gated_deltanet/`).
- **Vision + projector** — `Qwen3_5_VisionConfig` / `Qwen3_5_ProjectorConfig`
  (`xtuner/v1/model/compose/qwen3_5/qwen3_5_config.py`) already exist with
  `deepstack_visual_indexes: []`; only need the 4B dims.
- **Compose base** — `Qwen3_5_BaseConfig` exists; its `from_hf`/`save_hf` come
  from `Qwen3VLBaseConfig` / `BaseComposeModel`.

The **only genuinely new code** is the *dense* hybrid text tower + its config,
plus a 4B compose config and registration.

## 3. New code (kept inside the three seams)

### 3.1 Text tower — `xtuner/v1/model/dense/qwen3_5_text.py`

- `Qwen3_5_VLTextDense(Qwen3VLTextDense)` — reuse the deepstack-aware dense
  forward (`xtuner/v1/model/dense/qwen3vl_text.py`); deepstack is inert here
  (`deepstack_visual_indexes: []`). Override only `to_hf_key_list`:
  - prefix `model.language_model.` for `layers.*` / `embed_tokens`;
  - for `linear_attention` layers, rename `self_attn` → `linear_attn`
    (same rule as the MoE tower, driven by `config.layers_type[idx]`);
  - top-level `norm.` → `model.language_model.norm.`;
  - `tie_word_embeddings`: redirect `lm_head` → `embed_tokens`.
  - No MoE fusion / `safetensors_to_params` needed (plain dense MLP).
- `Qwen3_5_VLTextDenseConfig(TransformerConfig)` — `layers_type` computed
  (`full` every 4th), `attention = MHAConfig(with_gate=True, qk_norm=True,
  head_dim=256, num_attention_heads=16, num_key_value_heads=4)` **without**
  `sliding_window`, `linear_attention = GatedDeltaNetConfig(...)`,
  `rope_parameters_cfg` for interleaved partial-rotary mrope, `rms_norm_type`
  per parity check (§5).
- `Qwen3_5_VLTextDense4BConfig` — hard-coded 4B dims.

### 3.2 Compose config — `compose/qwen3_5/qwen3_5_config.py`

- Add `Qwen3_5_VLDense4BConfig(Qwen3_5_BaseConfig)`: vision `Qwen3_5_VisionConfig(depth=24, hidden_size=1024, out_hidden_size=2560)`,
  projector `Qwen3_5_ProjectorConfig(vision_hidden_size=1024, text_hidden_size=2560)`,
  `text_config = Qwen3_5_VLTextDense4BConfig(...)`.
- Widen `Qwen3_5_BaseConfig.text_config` type from `MoEConfig` to the shared base
  (`TransformerConfig` / `XTunerBaseModelConfig`) so a dense text config is
  accepted. (Small, localized interface widening — no behavior change.)

### 3.3 Registration — `xtuner/v1/model/__init__.py`

Import `Qwen3_5_VLDense4BConfig`, add a `model_mapping` alias
(`"qwen3_5-vl-dense-4b"`), extend `__all__`. Consistent with the existing
hard-coded `qwen3_5` family (no `get_model_config_from_hf` dispatch is added for
this family today).

## 4. MTP — deferred (decision)

Baseline ships **without** MTP. The 15 `mtp.*` checkpoint keys are not built, so
`from_hf` reports them as unexpected (matching HF, which lists
`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`), and `save_hf` does not
re-emit them. The round-trip test is scoped to non-`mtp.*` keys. Adding MTP to
the Dense path (mirroring `MoE.build_mtp_block` + forward/loss integration) is a
follow-up commit.

## 5. Parity result (decoder-layer bitwise)

Verified empirically (single GPU, bf16) against HF `Qwen3_5ForConditionalGeneration`'s
`language_model`. Findings:

- **GatedDeltaNet (linear) layers — bitwise (0.0)** out of the box: XTuner and HF
  share the same `fla` / `causal_conv1d` kernels.
- **RoPE — bitwise (0.0)**: `rope_type="qwen3_vl"` reproduces HF's interleaved
  partial-rotary mrope exactly. RMSNorm `zero_centered` matches HF's
  `output * (1 + weight)`. Gated-MHA q/gate chunk + `sigmoid(gate)` matches.
- **Full-attention layers** only diverged because XTuner defaults to **flash
  attention** while HF eager upcasts softmax to fp32. XTuner's own
  `eager_attention` op already matches HF's `eager_attention_forward` **bitwise**.

So the only thing needed for bitwise parity is forcing the eager attention path.
That is what the **`XTUNER_HF_IMPL`** switch does (§5.1).

Discovered + fixed along the way: `Dense.build_layers` did not forward
`rms_norm_type` to `DenseDecoderLayer` (MoE did), so a `zero_centered` dense model
would silently use the default RMSNorm. Fixed generally (no-op for existing
`"default"` dense models).

Parity outcome: with `XTUNER_HF_IMPL=true`, all 32 decoder layers (linear + full)
and the final-norm hidden state match HF **bitwise (max diff 0.0)**; logits match
bitwise. On the default flash path, the full-model forward matches HF within
`1e-2`.

### 5.1 `XTUNER_HF_IMPL` switch (ops-level)

New env var `XTUNER_HF_IMPL` (`xtuner/v1/utils/misc.py`) selects HF-exact op
implementations, patched **only at the ops layer** (never the decoder-layer
forward):

- `xtuner/v1/ops/attn_imp.py::get_attn_impl_fn` returns `eager_attention`
  regardless of the configured backend.
- `xtuner/v1/ops/rms_norm/__init__.py::get_rms_norm_fn` forces the native torch
  path (over triton).

Both read the env var live so a test can toggle it per model instance. `mha.py`
selects its attention op through `get_attn_impl_fn`.

## 6. Tests — `tests/model/test_qwen3_5_dense.py`

- Decoder-layer bitwise parity: one linear + one full layer vs HF.
- `save_hf` round-trip (byte-equal, non-`mtp.*` keys).
- Checkpoint path from env var (e.g. `QWEN3_5_DENSE_4B_PATH`).

## 7. Commit plan (stacked)

1. Dense text tower + config + compose 4B config + registration (baseline,
   no MTP).
2. Baseline tests (decoder-layer parity + save_hf round-trip).
3. (later) MTP support in the Dense path.
4. (later) §8 optimizations: EP n/a (dense), SP, torch.compile, fp8,
   activation offload — each with a comparison test.
