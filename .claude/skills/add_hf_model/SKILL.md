---
name: add_hf_model
description: >
  Add support for a HuggingFace model in XTuner so it can be trained with the
  full set of parallelism strategies and optimization switches. The user can
  provide a HuggingFace Hub repo id, a local model directory, or a model family
  already supported by `transformers`. This skill walks through the full path:
  locating the reference implementation, classifying the model, splitting new
  code across the model / module / ops layers, implementing the XTuner model
  class and config, registering the entry point, validating bitwise numerical
  parity with HuggingFace, writing regression tests, and enabling the training
  optimizations (EP/SP, micro-batch, torch.compile, fp8, activation offload).
---

# Add a HuggingFace model to XTuner

A complete handbook for porting any HuggingFace causal LM into XTuner with
correct training behavior. Covers four buckets: **Dense LLM**, **MoE LLM**,
**VLM / compose model**, and **`trust_remote_code` model**. Every step below is
grounded in concrete `file:line` references in this repository — read the cited
code before writing yours.

> **Anti-spaghetti reminder** — XTuner separates *what the model is* (`*Config`),
> *what it computes* (`Dense` / `MoE` / `BaseComposeModel` subclass), and *how its
> weights map to HF safetensors* (`to_hf_key_list`, optional `safetensors_to_params`
> / `param_to_safetensor`). Keep new code inside those three seams. **Never** add
> `if model_name == "..."` branches to the base classes.

---

## 0. Locate the source code

The only goal of this step is to find the **HuggingFace reference implementation
you will port from**, and to note the `model_type` you will register under. Do
not pick attention/router/layer here — that happens naturally in §1 as you read
the code.

1. If the user gives a **remote-code** model, the modeling code ships in the repo
   as `modeling_*.py` — read it directly.
2. If the user gives **weights only**, confirm the target Python environment with
   the user, then find the matching implementation inside that env's
   `transformers` install. If no code exists for that `model_type`, stop and ask
   the user.
3. Confirm the weights are **training-suitable** — at least bf16 — so training
   precision can be aligned later. Flag fp8/int4-only checkpoints to the user.
4. Note two strings from the HF config — you will need both downstream:
   ```python
   from transformers import AutoConfig
   cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
   cfg.model_type        # registration key (§4)
   cfg.architectures[0]  # the modeling class you are mirroring
   ```

---

## 1. How to do the migration

XTuner splits a model into three layers. You port by reusing what already exists
at each layer and only adding what is genuinely new — so the work is "read the HF
modeling code, then match each of its pieces to the XTuner component below". The
locations are listed plainly; you do not need extra guidance to map your model's
attention / router / layer onto the matching one.

- **model layer** (`xtuner/v1/model`) — the model body.
  - Dense body: `xtuner/v1/model/dense/dense.py`
  - MoE body: `xtuner/v1/model/moe/moe.py`
  - Multi-modal (compose) bodies: `xtuner/v1/model/compose/` — reuse the language
    model and add only the ViT / projector pieces.
- **module layer** (`xtuner/v1/module`):
  - Decoder layers: `DenseDecoderLayer`, `MoEDecoderLayer` —
    `xtuner/v1/module/decoder_layer`. Small changes add a switch on the existing
    layer; a genuinely novel architecture may add `{model_name}_decoder_layer`.
  - Attention configs: `MHAConfig` (`xtuner/v1/module/attention/mha.py`),
    `MLAConfig` (`.../mla.py`, DeepSeek-style latent),
    `GatedDeltaNetConfig` (`.../gated_deltanet.py`, linear).
  - Router configs (MoE): `GreedyRouterConfig`
    (`xtuner/v1/module/router/greedy.py`, softmax top-k — Qwen3 MoE, GptOss),
    `NoAuxRouterConfig` (`xtuner/v1/module/router/noaux_router.py`, grouped sigmoid
    aux-loss-free — DeepSeek V3).
  - Also: rope, rms_norm, etc.
- **ops layer** (`xtuner/v1/ops`) — kernels such as attention and rms_norm.

> **Caveat — don't reach for deprecated config classes.** `RopeScalingConfig` is
> **deprecated**; `RopeParametersConfig` (`xtuner/v1/module/rope/rope.py`, re-exported from
> `xtuner/v1/model/base.py`) is the source of truth — use it everywhere (your IDE/pyright will flag
> the deprecated one). Note the decoder-layer / `MHAConfig.build` signatures still *type* their rope
> argument as `RopeScalingConfig` for backward compatibility, so don't satisfy them by constructing
> the deprecated class. When a per-layer value is only needed to select **one module behavior** (e.g.
> `partial_rotary_factor` only chooses which `apply_rotary_emb` the attention uses), set that behavior
> **directly on the module** instead of threading a config through `build` — e.g. in your model's
> decoder layer, `self.self_attn.apply_rotary_emb = get_apply_rotary_emb(None,
> enable_partial_rotary=...)` (`xtuner/v1/ops`). This keeps per-layer behavior contained (the §C
> per-profile-RoPE pattern) and avoids the deprecated API entirely.

### Existing models to copy from

Pick the one whose attention + (router) match yours; the closer it is, the
smaller your diff.

| Your model is…                       | Copy from                            |
|--------------------------------------|--------------------------------------|
| Dense LLM                            | `xtuner/v1/model/dense/qwen3.py`     |
| MoE LLM                              | `xtuner/v1/model/moe/qwen3.py`       |
| MoE with MLA + aux-loss-free router  | `xtuner/v1/model/moe/deepseek_v3.py` |
| MoE needing weight-layout transform  | `xtuner/v1/model/moe/gpt_oss.py`     |
| Compose / VLM                        | `xtuner/v1/model/compose/qwen3_vl/`  |


After you have a plan, **write an integration design doc into `docs/design/model`**
and confirm with the user that the decomposition is reasonable before
implementing.

### File layout for the new model

| Bucket               | Model file                                                  | Public re-export                              | Test file                                  |
|----------------------|-------------------------------------------------------------|-----------------------------------------------|--------------------------------------------|
| Dense LLM            | `xtuner/v1/model/dense/<name>.py`                           | `xtuner/v1/model/__init__.py`                 | `tests/model/test_<name>_dense.py`         |
| MoE LLM              | `xtuner/v1/model/moe/<name>.py`                             | `xtuner/v1/model/__init__.py`                 | `tests/model/test_<name>_moe.py`           |
| Compose / VLM        | `xtuner/v1/model/compose/<name>/...`                        | `xtuner/v1/model/__init__.py`                 | `tests/model/test_<name>.py`               |

---

## 2. Implement the model class

Subclass `Dense`, `MoE`, or `BaseComposeModel`. You typically only need to
override one method.

### 2.1 `to_hf_key_list(self, key: str) -> list[str]` — **mandatory, abstract on `BaseModel`**

Declared at `xtuner/v1/model/base.py`. Translates an **XTuner-side** parameter
name into one or more **HF-side** safetensors keys. Return-list cardinality:

- **1 → 1** for plain weights (most params).
- **1 → N** for fused MoE experts (one fused param explodes into N per-expert
  HF keys; see `xtuner/v1/model/moe/qwen3.py`).

Patterns you almost always need (study `xtuner/v1/model/dense/qwen3.py`
and `xtuner/v1/model/moe/qwen3.py`):

```python
# 1. Tied embeddings: redirect lm_head → embed_tokens before adding the model. prefix.
if self.config.tie_word_embeddings and "lm_head" in key:
    key = key.replace("lm_head", "embed_tokens")

# 2. Add the "model." prefix that HF wraps everything in.
if "layers" in key or "embed_tokens" in key:
    key = "model." + key

# 3. HF MoE nests experts/gate under .mlp.; XTuner does not.
if "layers" in key:
    key = re.sub(r"layers\.(\d+)\.(experts|gate)", r"layers.\1.mlp.\2", key)

# 4. Top-level norm.
if key.startswith("norm."):
    return [key.replace("norm.", "model.norm.")]

# 5. MoE expert fusion: one fused param → N HF keys.
if "fused_w1w3.weight" in key:
    out = []
    for i in range(self.config.n_routed_experts):
        out.append(key.replace("fused_w1w3.weight", f"{i}.gate_proj.weight"))
        out.append(key.replace("fused_w1w3.weight", f"{i}.up_proj.weight"))
    return out

# 6. Model-specific buffers (e.g. FoPE: rotary_emb.sin_coef / cos_coef are persistent buffers).
if key.startswith("rotary_emb."):
    return [key.replace("rotary_emb.", "model.rotary_emb.")]
```

A prefix-region remap that differs by *deployment* (e.g. a text tower nested under a
VLM as `model.language_model.`) does **not** belong here — put it in the config's
`hf_key_mapping`. See §5.1.

### 2.2 `safetensors_to_params` / `param_to_safetensor` — **optional, only for layout mismatches**

If the HF storage layout cannot be expressed as “same tensor, different name”,
you must transform the bytes. Canonical example: GptOss stores expert weights as
`(num_experts, hidden_size, expert_dim * 2)` but XTuner fuses them as
`(num_experts * 2 * expert_dim, hidden_size)` — see overrides at
`xtuner/v1/model/moe/gpt_oss.py`.

Use this hook only when transposing/reshaping is unavoidable. Renames go in
`to_hf_key_list`; numerics go here.

### 2.3 Do **not** override

- `build_layers`, `build_embeddings`, `_init_weights` — the base handles them
  via the config. Override only if the new model adds a layer type the base
  cannot express (rare).
- `forward` — same reasoning. If you find yourself touching it, re-evaluate the
  bucket choice; the problem is probably in the config (`attention`, `router`,
  `first_k_dense_replace`).

---

## 3. Implement the config

Two layers: a **base config** that reads HF and a **size-specific subclass** per
released checkpoint.

### 3.1 Base config — three required members

Mirror `Qwen3MoEConfig` (`xtuner/v1/model/moe/qwen3.py`) or
`Qwen3DenseConfig` (`xtuner/v1/model/dense/qwen3.py`).

1. **`build(self) -> <ModelClass>`** — return `<ModelClass>(self)`.

2. **`@classmethod from_hf(cls, hf_path) -> Self`** — read the HF config and map
   its fields one-to-one onto your config. The field set is whatever the live
   config class declares (`TransformerConfig` in `xtuner/v1/model/base.py`,
   `MoEConfig` in `xtuner/v1/model/moe/moe.py`); read it, don't work from a
   hard-coded list. A few non-obvious traps:
   - Use `RopeParametersConfig.from_hf_config(hf_config)` from
     `xtuner/v1/module/rope/rope.py`. It already handles both the
     `rope_parameters` dict (HF ≥ 5.2.0) and the legacy `rope_scaling` dict
     (HF 4.57.x), plus YARN / FoPE / mrope special fields. **Do not** parse
     rope yourself.
   - For optional HF fields, use `getattr(hf_config, "<field>", <default>)`
     rather than `hf_config.<field>` — older checkpoints drop fields.

3. **`@property hf_config(self) -> <HFConfigClass> | None`** — the inverse of
   `from_hf`: every field `from_hf` reads must be re-emitted so `save_hf`
   round-trips.

**Built-in vs. `trust_remote_code` — the only axis that changes `from_hf` /
`hf_config`:** a `trust_remote_code` model is structurally still Dense / MoE /
VLM; the *only* difference is whether `transformers` ships a built-in
`<Name>Config` class. Decide which case you are in and follow the matching column
— this is the single place that contract is defined.

| | Built-in config (e.g. Qwen3) | `trust_remote_code` (no built-in config) |
|---|---|---|
| `from_hf` reads via | `<HFConfigClass>.from_pretrained(hf_path)` | `AutoConfig.from_pretrained(hf_path, trust_remote_code=True)`, then `assert hf_config.model_type == "<expected>"` |
| `hf_config` returns | a populated `<HFConfigClass>` re-emitting every field | `None` |
| `save_hf` weight-map | written from `hf_config` | same |
| `save_hf` side files | derived from `hf_config` | falls back to copying `config.json` / tokenizer / `*.py` from `self._hf_path` so the dir stays loadable with `trust_remote_code=True`; module-cache test in the matrix below |

Compare `Qwen3MoEConfig.hf_config` (built-in) and `Qwen3MoEFoPEConfig.hf_config`
(returns `None`), both in `xtuner/v1/model/moe/qwen3.py`. **Do not** invent a
built-in HF config class for a remote-code model — return `None`.

**Invariants:**

- `from_hf` ↔ `hf_config` must round-trip. The `test_save_hf` test (§7) is the
  enforcement mechanism — every key in the original index must appear in the
  saved index with byte-equal tensors.
- The config must be **pickleable and side-effect free**. No env vars, no file
  I/O, no `torch.cuda.*` at construction time.

### 3.2 Size-specific subclasses

Add one subclass per released checkpoint that hard-codes the published
dimensions. Mirror `Qwen3MoE30BA3Config` / `Qwen3MoE235BA22Config` in
`xtuner/v1/model/moe/qwen3.py`. Even if only one size ships, define at least one
— it is the entry point for `get_model_config` and for tests.

---

## 4. Register the entry point

Open `xtuner/v1/model/__init__.py` — all four edits are short:

1. **Import** — bring in the new config classes.
2. **Alias** — add `"<alias>": <NewSizeConfig>()` to `model_mapping`. Lookup
   normalizes case and `-`/`_`, so `"my-model-7b"` and `"my_model_7b"` are the
   same key.
3. **Dispatch** — add `elif cfg.model_type == "<model_type>":` to
   `get_model_config_from_hf` returning `<NewConfigClass>.from_hf(model_path)`.
   The catch-all `raise ValueError(...)` must stay last.
4. **`__all__`** — re-export the new public config names.

**Never** key dispatch on parameter names, file paths, or architecture strings.
Only `model_type`.

---

## 5. Compose / VLM specifics

Compose models live under `xtuner/v1/model/compose/<name>/` and follow a
three-tower pattern. See `xtuner/v1/model/compose/base.py` and
`xtuner/v1/model/compose/qwen3_vl/qwen3_vl_config.py`.

### 5.1 Config shape

`BaseComposeConfig` (`compose/base.py`) carries three sub-configs:

```python
vision_config:    XTunerBaseModelConfig     # e.g. Qwen3VLVisionConfig
projector_config: XTunerBaseModelConfig     # e.g. Qwen3VLProjectorConfig
text_config:      XTunerBaseModelConfig     # any Dense/MoEConfig subclass
freeze_vision:    bool = False
freeze_projector: bool = False
freeze_language:  bool = False
```

Reuse existing text-tower configs where possible. `Qwen3VLDense*` delegates to
the Dense path via `xtuner/v1/model/dense/qwen3vl_text.py`. **Do not** copy-paste
the entire dense/MoE config — compose over them.

**Reused text-tower key prefixes — use `hf_key_mapping`, not `to_hf_key_list`.** A
text tower reused inside a VLM keeps its *standalone* `to_hf_key_list` (it emits
`model.<...>` as if it were a plain LLM). The compose checkpoint nests it one level
deeper (`model.language_model.<...>`). Do **not** teach the text tower's
`to_hf_key_list` about that prefix — that leaks the compose context into a tower that
must also work standalone. Set the remap on the **text config** instead:
`hf_key_mapping = {r"^model\.": "model.language_model."}` (applied in
`_init_load_spec`). Keep *structural* renames in `to_hf_key_list`; the
deployment-dependent prefix region belongs in `hf_key_mapping`.

### 5.2 Model shape

`BaseComposeModel` (`compose/base.py`) builds three sub-models:
`vision_tower`, `multi_modal_projector`, `language_model`.

- **`from_hf(hf_path)`** — delegates to each sub-model with
  `strict=False` (vision keys are missing from the language-model state dict
  and vice versa) and unions the missing-keys sets.
- **`save_hf(hf_dir)`** — saves each tower with a distinct prefix
  (`"model-language"`, `"model-vision"`, `"model-projector"`) and merges the
  three `weight_map`s into one `model.safetensors.index.json`.

### 5.3 `hf_config` for VLMs

Many VLMs ship without a stable HF top-level config class; `hf_config` returns
`None` (`compose/qwen3_vl/qwen3_vl_config.py`). In that case, `save_hf`
falls back to copying source files from `self._hf_path` (see §3.1).

### 5.4 Vision-side quirks to watch for

- `attn_impl` selectable per checkpoint: `"flash_attention" | "flex_attention" |
  "eager_attention"`.
- `deepstack_visual_indexes` (e.g. `[8, 16, 24]`) — auxiliary supervision depths
  on the vision tower; must round-trip.
- Specialized rope: `rope_type="qwen3_vl"`, `mrope_section=[24, 20, 20]` —
  delegated to `RopeParametersConfig.from_hf_config`.
- The projector’s `torch.compile` path is enabled on torch ≥ 2.9.1 (see commit
  `f6d74efb`). Tests must validate both compiled and eager outputs match.

---

## 6. Baseline parity — no complex parallelism yet

First make the model correct on the **simplest execution path**: single rank, no
EP/SP, no `torch.compile`, no fp8, no activation offload. Align precision from
the inside out — ops → decoder layer → full model — reusing XTuner's own modules
at every level. Only once this baseline holds do you add the parallel and
optimization features (§8).

1. **Reuse, don't reimplement.** Use the existing XTuner modules — MHA, MLA,
   GatedDeltaNet, GroupLinear, etc. Do not re-implement them with naive torch.

2. **Route op differences through `XTUNER_HF_IMPL`.** This env var already exists
   (`xtuner/v1/utils/misc.py`) and selects HF-exact ops via per-op selectors:
   `xtuner/v1/ops/attn_imp.py::get_attn_impl_fn` (eager attention),
   `xtuner/v1/ops/rms_norm/__init__.py::get_rms_norm_fn` (native torch rms_norm),
   and `xtuner/v1/ops/gated_deltanet/__init__.py::{get_chunk_gated_delta_rule_fn,
   get_causal_conv1d_fn}` (return the canonical fla / causal_conv1d wrappers HF
   calls instead of XTuner's compile-friendly `torch.library.custom_op` wraps —
   same forward output, but the backward op graph now matches HF). Selectors
   read the env var **live** so a test can toggle it per model instance. If your
   model introduces a new op whose fast path is not bitwise against HF (typical
   for fla-backed linear-attention variants), add its HF-exact branch in the
   corresponding op selector the same way. **You may only patch at the ops
   level** — never take the shortcut of patching the entire
   `XTunerDecoderLayer.forward`.

   Pair `XTUNER_HF_IMPL` with `XTUNER_DETERMINISTIC=true`. The latter gates an
   autotune-pin block in `xtuner/v1/__init__.py` that monkey-patches
   `triton.autotune` to lock every kernel to its first config and disable the
   result cache. Without the pin, fla's `@triton.autotune` picks tiling /
   reduction order per kernel and the choice drifts between runs, producing
   1 ULP per linear-attention layer that the chain rule amplifies to ~1.0 at
   the model boundary (§12). `tests/conftest.py` sets the env var and runs
   `import xtuner.v1` so the patch installs before fla is imported. The patch
   lives on the `xtuner.v1.*` import chain (not in the conftest) because
   `multiprocessing.spawn` children of `MultiProcessTestCase` don't go through
   pytest's conftest — they re-import the test class top-level (which pulls
   `xtuner.v1.model.*`) and the patch must run in those processes too.
   Production training doesn't set the env var, so the block is inert there.

3. **Decoder-layer bitwise parity — forward AND backward, both required.** Build
   one XTuner `DecoderLayer` (one per distinct layer type — e.g. a linear and a
   full layer for a hybrid model) and the matching HF layer, set `XTUNER_HF_IMPL`
   and `XTUNER_DETERMINISTIC`, feed identical inputs, and align **bitwise** — not a
   tolerance. **Backward is not optional**: a `single layer → final norm → lm_head
   → CE loss → loss.backward()` test asserts the **layer output, the loss, and the
   input gradient `dL/dx`** all bitwise-equal to HF — test one full and one linear
   layer separately (reference: `test_decoder_layer_bitwise_parity`). A bitwise
   *forward* does not imply a bitwise backward — the forward values can round
   identically while the backward graph differs (see the fp32-softmax note in §12).
   This is the **primary** path, required at all scales, and the only feasible one
   for large models (it never loads the full model — see "load only what you use"
   below). When you hand-build a single layer's
   inputs, the two sides' `causal mask` / `seq_ctx` must match exactly — see §12.
   **Load only what you use** so the test fits any model size: build the XTuner
   tower on `meta` and materialize (`to_empty` + selective load) *only* the tested
   layer + `norm` + `lm_head` via `HFCheckpointLoader.load(key)` (it reads just the
   needed safetensors shard); build a *standalone* HF `DecoderLayer` + norm +
   lm_head and load the same keys. Never `from_pretrained` the whole model for a
   single-layer test. (`build_rotary_embedding` builds the rotary on CPU with real
   buffers even under meta, so `model.rotary_emb` is usable without materializing
   the layers.) For < 40B you may also read per-layer hidden states from one
   full-model forward (HF `output_hidden_states` + XTuner `return_hidden_states`);
   mind the off-by-one in §12.

4. **Model config matches HF hyperparameters** — see the §3.1 built-in vs.
   remote-code contract for how `from_hf` / `hf_config` differ between the two.

5. **Full-model forward parity for models < 40B**, bitwise against HF, on the
   single-rank path (FSDP on or off; no EP / compile / fp8 yet).

6. **`save_hf` / `from_hf` correctness** (round-trip; see §7).

7. **Loss-convergence trace + engine test.** Once model forward and decoder-layer
   forward/backward are aligned, record a loss-convergence trajectory and add an
   engine test case.

**Commit the baseline — do not stop to ask.** Committing is part of this
workflow: invoking this skill authorizes the in-workflow commits. Once parity
holds on the simple path and lint/tests pass, commit immediately so this correct
baseline is preserved before any parallel/optimization work begins (the §8
sub-agents branch off this commit). Commit each logical step as you finish it
(§10) rather than batching everything to the end or waiting for a separate
go-ahead.

Report back to the user: whether bitwise parity was achieved, and the residual
error (decoder-layer level and model level) once XTuner's internal components are
used.

---

## 7. Baseline tests — add these first

Before any parallel feature, add the most basic regression tests so the baseline
(§6) is locked and the §8 work has a fixed reference to measure against. Add
`tests/model/test_<name>_<bucket>.py`, mirroring
`tests/model/test_qwen3_moe.py` (MoE) or `tests/model/test_qwen3_dense.py`
(Dense). All tests extend `xtuner._testing.DeterministicDDPTestCase` for
deterministic distributed setup.

Reference test cases:
- decoder-layer bitwise parity + save_hf round-trip: `tests/model/test_qwen3_5_dense.py`
  (hybrid linear/full dense VLM; toggles `XTUNER_HF_IMPL` per case)
- model forward & save & load parity: `tests/model/test_qwen3_moe.py`
- engine training test: `tests/engine/test_moe_train_engine.py`

### 7.1 Baseline test matrix

All single execution path — no EP / dispatcher / compile / fp8 here; those are
§8 tests, written against this baseline.

| Case                                            | Dense | MoE | Compose | Notes                                                                              |
|-------------------------------------------------|:-----:|:---:|:-------:|------------------------------------------------------------------------------------|
| Decoder-layer parity — output + loss + `dL/dx`  | ✅    | ✅  | ✅      | **Required at all scales, bitwise.** Test one full and one linear layer *separately*. Standalone & memory-light: build the XTuner tower on `meta` and `materialize_xtuner_submodule` only {tested layer, `norm`, `lm_head`}; build a standalone HF `DecoderLayer`; load via `HFCheckpointLoader`. Under `XTUNER_HF_IMPL` assert the **layer output, loss, and `dL/dx` all bitwise** (forward AND backward at the layer level — a bitwise forward ≠ bitwise backward, see §12). Reference: `test_decoder_layer_bitwise_parity`. |
| Whole-model forward + backward parity (by scale) | ✅(4B)| scale | ✅(4B) | Run the **top-level model** end-to-end vs HF, eager both sides (`XTUNER_HF_IMPL` + `XTUNER_DETERMINISTIC` + HF `eager`). For a **VLM** that's the compose model on an **image** prompt (vision + projector + text — subsumes "VLM forward parity"); for a plain LLM, the text/LM model. With both env switches on plus the autotune pin loaded (§6.2), **forward AND backward are bitwise**: assert logits + loss + `dL/d(pixel_values)` / `dL/d(inputs_embeds)` all `== 0.0`. **Scale**: small = fwd+bwd; medium = forward only; large (can't e2e-forward) = skip → integration. Reference: `test_vl_forward_parity`. |
| Vision-tower bitwise parity                      | —     | —   | ✅      | **Compose/VLM only.** Compare HF's `visual` output (`pooler_output`, which is **post-merger**) bitwise (0.0), both eager. XTuner splits that boundary into `vision_tower` (patches → merged hidden) + `projector` (the merger MLP → text dim), so load **both** standalone to match it — don't build the whole VLM just for the vision path. Build them on the **real device, not `meta`**: the vision rotary is computed into a buffer at forward time, so `to_empty` would leave garbage (unlike the text rotary, which `build_rotary_embedding` builds real even under meta — §6.3). Vision attention is *non-causal* → `eager_attention(causal=False)`, which `XTUNER_HF_IMPL` selects. Reference: `test_vision_tower_bitwise_parity`. |
| FSDP forward parity                              | ✅    | ✅  | ✅      | Looser tolerance (≈ 3e-2 is the established budget).                                |
| `save_hf` round-trip (byte-equal tensors)        | ✅    | ✅  | ✅      | See `tests/model/test_qwen3_moe.py`.                                       |

### 7.2 Parity bar by model scale

Two independent axes:

**Per-layer (op) parity — required at _all_ scales, bitwise.** Test one full and
one linear decoder layer separately; for each assert layer **output + loss + `dL/dx`**
bitwise vs HF under `XTUNER_HF_IMPL`. Keep it memory-light (XTuner `meta`-build +
materialize only the tested layer + `norm` + `lm_head`; standalone HF layer; load
via `HFCheckpointLoader`) so it runs even for models too large to forward
end-to-end. This is the bitwise guarantee for both forward and backward ops.

**Whole-model (integration) parity — on the _top-level_ model, graded by what fits
one GPU.** Run the real model end-to-end: for a VLM that's the **compose model on an
image prompt** (one test covers vision + projector + text — don't add a separate
text-tower model test); for a plain LLM, the text/LM model.
- **Small** (e.g. dense 4B): both forward (logits/loss) AND backward
  (`dL/d(pixel_values)` for a VLM, `dL/d(inputs_embeds)` for text) **bitwise**.
  Prerequisites: `XTUNER_HF_IMPL=true` so every op selector returns the
  HF-canonical callable (§6.2) AND `XTUNER_DETERMINISTIC=true` so Triton
  autotune is pinned (§6.2). With both on, even hybrid linear/full models match
  HF byte-for-byte through the whole compose chain. Don't check weight
  gradients — they accumulate across positions + tied `lm_head` and aren't the
  right granularity for parity.
- **Medium**: forward only (logits/loss bitwise).
- **Large** (cannot forward end-to-end on the test GPUs): skip the whole-model
  test; leave it to integration tests. The per-layer test still gives op-level
  forward+backward bitwise.

### 7.3 Required test idioms

`tests/model/test_qwen3_5_dense.py` is the reference template; extend
`DeterministicDDPTestCase` (`xtuner._testing`), which provides the shared
parity scaffolding so the test file only holds model-specific logic:

- `with self.hf_impl():` — sets/restores `XTUNER_HF_IMPL=true`; build the XTuner
  model **inside** the block so its attention picks the eager op at construction.
- `self.materialize_xtuner_submodule(model, submodule, prefix, loader)` — for a
  model built on `meta`, materialize *only* that submodule on CUDA and load its
  weights (so a single-layer test fits any model size).
- `self.load_params_from_hf(module, key_for, loader)` /
  `self.xtuner_ckpt_key(model, name)` — selective per-param load by checkpoint key
  (`HFCheckpointLoader` reads only the needed shard); `xtuner_ckpt_key` applies
  `to_hf_key_list` + the config's `hf_key_mapping`.

```python
# Construct on meta, materialize on device — single-rank baseline, eager, no EP.
with torch.device("meta"):
    cfg = get_model_config_from_hf(hf_model_path)
    cfg.compile_cfg = False       # baseline tests run eager
    model = cfg.build()._to_device_dtype(dtype=torch.bfloat16, skip_buffers_dtype=True)

# Optionally shard with FSDP (data-parallel only), then load.
model.fully_shard(fsdp_config=FSDPConfig(cpu_offload=False))
model.from_hf(hf_model_path)
```

Older `patch_hf_rms_norm` / `patch_hf_rope` helpers also live in
`xtuner._testing`. **Check they actually fit your model before using them** —
they were written for earlier models and match HF modules
by class-name substring and attribute name (e.g. `variance_epsilon`). On a newer
model they can silently mis-patch (wrong eps attribute, or replacing a
`zero_centered` RMSNorm with a default one). Often the native XTuner ops already
match HF bitwise (see §12), so no patching is needed; prefer `XTUNER_HF_IMPL`
over these helpers for parity. The §8 parallel-feature tests reuse this same
idiom with `cfg.dispatcher` / `cfg.ep_size` (and `FSDPConfig(ep_size=...)`) set.

### 7.4 Checkpoint paths

Read from env vars; **never** hard-code. Example:
`QWEN3_MOE_PATH = os.environ["QWEN3_MOE_PATH"]` at
`tests/model/test_qwen3_moe.py`. Document new env vars in the PR body.

### 7.5 Drop-in training config

Ship a runnable training config at `ci/config/<name>.py` so a developer (and CI) can verify
the port end-to-end with one command: load the HF checkpoint, run a few real training steps,
watch the loss curve descend as expected. This is the smoke test that catches anything the
unit tests miss — the full `from_hf` → `model.forward` → loss → backward → optimizer step →
FSDP shard/reduce chain — and the file doubles as the example users copy when wiring the
model into their own training pipeline. Mirror `ci/config/qwen3_moe_30BA3.py` (MoE) or
`ci/config/qwen3_dense.py` (dense), keeping its structure: one `<NewSizeConfig>()` (from §3.2)
fed into a `TrainerConfig` alongside `optim_cfg` / `lr_cfg` / `fsdp_cfg` / `dataset_cfg` /
`dataloader_cfg` / `loss_cfg`. Set `loss_cfg = CELossConfig(mode="chunk")` — the chunked
cross-entropy keeps the `logits → loss` peak memory bounded (it never materializes the full
`(seq, vocab)` logits), which matters for the large-vocab models this skill targets; do **not** leave
it on the `"eager"` default. `load_from` and `tokenizer_path` read from an env var (§7.4 — typically
the same one as the parity test). Verify by running ~50 steps and confirming the loss drops
monotonically into a plausible range for that model size; record the trajectory in the PR body
alongside the §6 convergence trace.

---

## 8. Training optimizations

Only after the §6 baseline is committed and the §7 tests are green: verify each
switch runs correctly and keeps precision within budget **against that baseline**.

1. EP and SP can be enabled normally.
2. With `ep_size > 1`, `intra_layer_micro_batch` enables correctly and stays
   precise.
3. `torch.compile` runs correctly.
4. fp8 can be enabled.
5. `XTUNER_ACTIVATION_OFFLOAD` can be enabled.

Each switch needs a test that compares its output to the §6 baseline — e.g. the
dispatcher × ep_size parity matrix (at minimum `{(None,1), ("all2all",4),
("all2all",8)}`, add `"deepep"` if supported), compiled-vs-eager, or fp8-vs-bf16
within tolerance.

These items are well-suited to the multi-agent workflow in §11: the main agent
analyzes and creates tasks (with test cases), sub-agents develop in parallel, and
the main agent merges. Report the precision error introduced by each switch.

---

## 9. Reference cheat-sheet

| Concept                 | Concrete location                                                              |
|-------------------------|--------------------------------------------------------------------------------|
| Dense base config       | `xtuner/v1/model/base.py` (`TransformerConfig`)                        |
| MoE base config         | `xtuner/v1/model/moe/moe.py` (`MoEConfig`)                             |
| Compose base config     | `xtuner/v1/model/compose/base.py` (`BaseComposeConfig`)                  |
| `from_hf` (model)       | `xtuner/v1/model/base.py` (`from_hf`)                                  |
| `save_hf` (model)       | `xtuner/v1/model/base.py` (`save_hf`)                                  |
| `to_hf_key_list` proto  | `xtuner/v1/model/base.py`                                                  |
| MoE layer split         | `xtuner/v1/model/moe/moe.py` (`first_k_dense_replace`)                 |
| RoPE auto-parse         | `xtuner/v1/module/rope/rope.py` (`RopeParametersConfig.from_hf_config`)|
| Dense reference         | `xtuner/v1/model/dense/qwen3.py`                                        |
| MoE reference           | `xtuner/v1/model/moe/qwen3.py`                                          |
| MoE w/ MLA + NoAux      | `xtuner/v1/model/moe/deepseek_v3.py`                                           |
| MoE w/ layout transform | `xtuner/v1/model/moe/gpt_oss.py`                                        |
| Remote-code reference   | `xtuner/v1/model/moe/qwen3.py` (`Qwen3MoEFoPEConfig`)                  |
| Compose reference (VLM) | `xtuner/v1/model/compose/qwen3_vl/`                                            |
| Dense test reference    | `tests/model/test_qwen3_dense.py`                                              |
| MoE test reference      | `tests/model/test_qwen3_moe.py`                                                |
| Engine test reference   | `tests/engine/test_moe_train_engine.py`                                        |

---

## 10. Commit discipline

0. **Commit as you go — this is authorized, do not pause for a separate
   go-ahead.** The user invoking this skill authorizes the commits this workflow
   produces. Finish a logical step → lint/test → commit, then move on. Do not
   leave the whole port uncommitted and end with "should I commit?"; that strands
   the work. (Hard-to-reverse or outward actions like `git push` / opening a PR
   still need explicit confirmation — committing locally does not.)
1. **Follow the stacked-PR convention.** Plan upfront how many commits the model
   port needs — one per logical step (e.g. shared-path fixes; model class +
   config + registration; the `XTUNER_HF_IMPL` switch; baseline tests; then one
   per parallel/optimization feature). Order them by dependency (a shared-path fix
   the baseline relies on goes first). Land every later change as
   `git commit --fixup=<sha>` + `git rebase --autosquash` into the commit it
   belongs to. **Never grow the history with endless patch commits just to keep
   fixing things up.**
2. **Every commit must pass lint:**
   ```bash
   pre-commit run --files $(find xtuner/v1)
   ```

---

## 11. Multi-agent workflow

This applies **only to the §8 parallel/optimization features** — and only once
the §6 baseline and §7 baseline tests are done and committed (§10). The baseline
itself is built sequentially, not multi-agent; the §8 features are independent of
each other, which is what makes them safe to develop in parallel.

1. The **main agent** splits the §8 features into tasks and, per §8, writes each
   feature's comparison test against the committed baseline and commits the tests.
2. **Sub-agents** branch off the baseline/test commit and develop one feature
   each (e.g. one switch / one tower) in parallel, making its test pass.
3. **The two sides adjust each other** — a shared-interface change is made once by
   the main agent (it updates the test/baseline; sub-agents rebase), and a failing
   test pushes the sub-agent to fix its feature, not to weaken the test.
4. The **main agent** merges, resolves conflicts, and reports the precision error
   each feature introduces relative to the baseline.

---

## 12. Parity debugging pitfalls

Hard-won notes from aligning hybrid (GatedDeltaNet + gated-MHA) models bitwise.
The meta-rule: **measure on real GPU and bisect to localize the divergence before
concluding anything** — most "bugs" here are measurement artifacts, not model
errors. A clean way to bisect is to feed one side's intermediate tensors into
both modules and compare step by step (norm → q/k/v → qk_norm → rope → attention
core → o_proj).

- **Single-layer parity: the two sides' inputs must match exactly — especially
  the causal mask.** When you hand-build one layer's inputs, passing
  `attention_mask=None` to an HF attention/layer makes HF apply **no** causal
  mask while XTuner stays causal, producing a large *false* diff. Build a causal
  mask equivalent to XTuner's `seq_ctx`, or drive the layer through the model so
  the mask is constructed internally.

- **`flash` ≠ `eager`, and HF eager upcasts softmax to fp32.** XTuner defaults to
  flash attention; HF reference eager does softmax in fp32. They are numerically
  close but not bitwise equal, so bitwise parity requires forcing eager on the
  XTuner side (that is exactly what `XTUNER_HF_IMPL` does). XTuner's
  `eager_attention` matches HF's `eager_attention_forward` bitwise.

- **A bitwise forward does not imply a bitwise backward.** The eager softmax must
  run in **fp32** (`softmax(..., dtype=torch.float32).to(x.dtype)`, like HF), not
  bf16. With a bf16 softmax the forward logits can still round identically to HF
  (bitwise), but the *backward* differs (~1e-5 on `dL/dx`) because the gradient
  accumulates at bf16 precision. So always check backward bitwise separately
  (the `loss.backward()` test), not just forward — and align op dtypes (softmax,
  norms) in the backward-sensitive direction, fp32 where HF uses fp32.

- **Whole-model backward is bitwise — but only with the full env stack.**
  Per-layer bitwise backward does not on its own imply whole-model bitwise
  backward; the missing pieces are `XTUNER_HF_IMPL=true` AND
  `XTUNER_DETERMINISTIC=true` AND the conftest must `import xtuner.v1` so the
  autotune-pin block in `xtuner/v1/__init__.py` runs before any test reaches
  fla. Without those, two distinct ULP-level sources compound through the LM
  chain rule and look indistinguishable from "fundamental bf16 noise":

  1. **Triton autotune drift in fla.** fla's `@triton.autotune` picks
     tiling / num_warps / reduction-order per kernel. Across runs the choice
     can change, producing a ~1-ULP backward diff per linear-attention layer
     that the chain rule amplifies — a 32-layer LM stack turned a per-layer
     ~2e-3 abs diff at grad magnitude ~30 into a ~1.0 diff on the
     massive-activation channels of `dL/d(pixel_values)`. The pin (configs
     held to the first one, cache disabled) is in `xtuner/v1/__init__.py`
     under an `XTUNER_DETERMINISTIC` gate. It must run **before any module
     imports fla** — fla's `@triton.autotune` decorators evaluate at import
     time, so patching after the fact has no effect on kernels already
     constructed. That's why the patch is on the `xtuner.v1.*` import chain
     rather than in the conftest: `multiprocessing.spawn` children of
     `MultiProcessTestCase` re-import the test class top-level (which pulls
     `xtuner.v1.model.*`) without running conftest, so the conftest alone
     can't install the patch in those processes.
  2. **`torch.library.custom_op` wraps for the linear-attention path**
     (`GatedDeltaNet`, `causal_conv1d`) produce a different backward autograd
     graph from the canonical fla / causal_conv1d call HF uses, even though
     the underlying kernel and forward output are identical. `XTUNER_HF_IMPL`
     routes through `get_chunk_gated_delta_rule_fn` / `get_causal_conv1d_fn`
     (§6.2) so the parity path matches HF byte-for-byte.

  Production train doesn't set either env var; the custom_op wraps + autotune
  still operate. These switches are **parity-test only**. If the whole-model
  backward is not bitwise but the per-layer backward is, suspect one of the
  two before suspecting depth-accumulated bf16 noise.

- **`flash`(XTuner varlen) vs `flash_attention_2`(HF) is bitwise only for short,
  single-tile sequences.** They are the same kernel, so at a short seqlen (e.g.
  32) the logits are bitwise (0.0); at longer sequences the tiling /
  accumulation order differs and they drift ~1e-2. So a short-seq text test can
  assert flash-vs-FA2 bitwise, but anything realistic-length (or a VLM whose
  image tokens make the sequence long) needs the **eager** path for
  seqlen-independent bitwise. Don't conclude "flash matches FA2" from one short
  input.

- **Shared kernels are already bitwise; focus on what isn't.** GatedDeltaNet /
  linear attention go through the same `fla` / `causal_conv1d` kernels as HF, and
  native rms_norm / rope match HF bitwise — these come out at 0.0 with no work.
  Spend your effort on the path that actually differs (usually full attention).

- **`output_hidden_states` off-by-one (full-forward route, < 40B only).** HF's
  tuple is `[emb, L0_out, …, L{n-2}_out, post_norm]`: index `i+1` is layer `i`'s
  pre-norm output only for `i ≤ n-2`; the **last** entry is post-norm, not the
  last layer's pre-norm output. Compare the last layer via `model.norm(...)` or
  you will see the final layer "explode".

- **A big max-diff is not automatically a bug.** Late layers have *massive
  activations* (a few channels in the tens/hundreds); a tiny relative error there
  shows up as a large absolute max-diff. Look at the mean and the trend, not just
  the max. Conversely, feeding `randn` inputs to a single attention layer is
  ill-conditioned and misleads — use real (or model-produced) hidden states.

- **A single-model failure is often a general-path bug.** E.g. `Dense.build_layers`
  not forwarding `rms_norm_type` to `DenseDecoderLayer` broke any `zero_centered`
  dense model, not just this one. Fix the shared invariant, not the symptom.

- **Loading a standalone sub-tower uses `strict=False`.** A text tower loaded
  directly from a full VLM checkpoint will see vision (and deferred `mtp.*`) keys
  as unexpected; load with `strict=False` so they are ignored.

- **Test the whole VLM, not just the text tower.** Text-tower parity leaves the
  vision tower and projector completely untested. A compose port needs a
  full-model forward-parity test with real image inputs (§7.1). The text path of
  the compose forward still runs a *dummy* vision forward to keep the tower in the
  autograd graph, so vision bugs surface even on text prompts.

- **Vision attention is non-causal — give `eager_attention` a `causal=False`
  path.** XTuner's `eager_attention` defaulted to a causal block-diagonal mask;
  that is wrong for a vision tower (bidirectional within each image), which is why
  forcing vision to the causal eager op produced large diffs. The op now takes
  `causal=True|False` (False → non-causal block-diagonal mask), the vision module
  already calls it with `causal=False`, and the vision tower routes through
  `get_attn_impl_fn` so `XTUNER_HF_IMPL` selects eager. With that, the *whole* VLM
  is bitwise in one run, everything eager: `XTUNER_HF_IMPL` on the XTuner side
  (text causal-eager + vision non-causal-eager) and HF loaded with a single
  `attn_implementation="eager"`. Prefer eager for parity — it is
  seqlen-independent, unlike flash/FA2 (see the flash seqlen bullet above).

- **Vision pos-embed dtype when reusing the Qwen3-VL tower.** HF's
  `fast_pos_embed_interpolate` returns fp32 (bilinear weights are fp32) and HF's
  vision forward casts it (`pos_embeds.to(hidden_states.dtype)`); XTuner's vision
  forward adds it without a cast. If you patch in the HF method for parity, cast
  the result back to the param dtype, or you get an fp32/bf16 `LayerNorm` mismatch
  (`expected scalar type Float but found BFloat16`).

- **Hybrid-model `torch.compile`: choose `fullgraph` per layer — verify, then fall
  back; don't gate *whether* to compile.** When a decoder layer's forward is wrapped
  into a higher-order op (HOP) — e.g. by an activation-checkpoint wrapper, which
  `torch.compile` lowers to a HOP whose body must be functional so it can be
  recomputed in backward — *and* that layer is a linear-attention layer
  (GatedDeltaNet), the layer mutates an outer-scope variable inside the HOP
  (GatedDeltaNet writes back into `seq_ctx`). Under `fullgraph=True` that raises
  `HigherOrderOperator ... Mutating a variable not in the current scope (SideEffects)`.
  This is **not** a blanket "linear attention ⇒ `fullgraph=False`" rule — whether it
  triggers depends on whether the layer is actually HOP-wrapped and whether that
  linear impl really writes outer state. So for such a HOP + linear combination,
  **first test whether `fullgraph=True` runs**; keep it if it does (it compiles more
  of the layer), and only **fall back to `fullgraph=False`** if it doesn't —
  `fullgraph=False` survives by graph-breaking at the mutation and running that
  side-effecting write in eager. Make this a **per-layer** choice in
  `Dense.fully_shard` (decide each layer's `fullgraph`), not a decision about whether
  a layer is compiled at all.

