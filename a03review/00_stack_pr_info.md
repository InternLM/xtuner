# GLM-5.3-Flash Stack PR Metadata

> **Backup (2026-10-08):** the pre-merge state of this stack was backed up to
> `feat/glm53flash-f6-text-moe-bak1008` (tip `016500f047`, identical to the original
> `feat/glm53flash-f6-text-moe`) and pushed to `InternLM/xtuner` before the rebased
> branches below were force-pushed. The metadata in this first section reflects the
> pre-merge state; see the updated H1 at the bottom for the current state.

7 stacked PRs for GLM-5.3-Flash support in XTuner (bottom → top).
Review each PR against its own diff only (base branch of that PR).

| # | PR | Title | Head → Base | State |
|---|----|-------|-------------|-------|
| 1 | [#2105](https://github.com/InternLM/xtuner/pull/2105) | [Feature] Add GLM-5.3-Flash F0: 25B cropped reference checkpoint builder | `feat/glm53flash-materialize-full-f0` → `main` | OPEN |
| 2 | [#2106](https://github.com/InternLM/xtuner/pull/2106) | [Feature] Add GLM-5.3-Flash F3: Kimi Delta Attention (KDA) | `feat/glm53flash-f3-kda` → `feat/glm53flash-materialize-full-f0` | OPEN |
| 3 | [#2107](https://github.com/InternLM/xtuner/pull/2107) | [Feature] Add GLM-5.3-Flash F4: mHC four-stream residual | `feat/glm53flash-f4-mhc` → `feat/glm53flash-f3-kda` | OPEN |
| 4 | [#2108](https://github.com/InternLM/xtuner/pull/2108) | [Feature] Add GLM-5.3-Flash F5: NoPE DSA + KPool indexer + clamped SwiGLU | `feat/glm53flash-f5-nope-dsa` → `feat/glm53flash-f4-mhc` | OPEN |
| 5 | [#2109](https://github.com/InternLM/xtuner/pull/2109) | [Feature] Add GLM-5.3-Flash F1: VL data preprocessing pipeline | `feat/glm53flash-f1-vl-data` → `feat/glm53flash-f5-nope-dsa` | OPEN |
| 6 | [#2110](https://github.com/InternLM/xtuner/pull/2110) | [Feature] Add GLM-5.3-Flash F2: vision tower + projector (eager) | `feat/glm53flash-f2-vision-tower` → `feat/glm53flash-f1-vl-data` | OPEN |
| 7 | [#2111](https://github.com/InternLM/xtuner/pull/2111) | [Feature] Add GLM-5.3-Flash F6 core: text model + MTP + compose model | `feat/glm53flash-f6-text-moe` → `feat/glm53flash-f2-vision-tower` | OPEN |


---

## Stack 1/7: PR #2105 — [Feature] Add GLM-5.3-Flash F0: 25B cropped reference checkpoint builder

- **URL**: https://github.com/InternLM/xtuner/pull/2105
- **State**: OPEN
- **Head branch**: `feat/glm53flash-materialize-full-f0`
- **Base branch**: `main`
- **Commits**: 2

### Commits

- `29a5347a296f` **[Refactor] Unify the DTensor unshard contract behind materialize_full** (jayhenry, claude, 2026-09-23)
- `91bebe283886` **[Feature] Add GLM-5.3-Flash F0: 25B cropped reference checkpoint builder** (jayhenry, claude, 2026-09-23)

### PR Message

```markdown
Stack (bottom to top):

1. **#2105 `feat/glm53flash-materialize-full-f0` → `main`** ← you are here
2. #2106 `feat/glm53flash-f3-kda` → `feat/glm53flash-materialize-full-f0`
3. #2107 `feat/glm53flash-f4-mhc` → `feat/glm53flash-f3-kda`
4. #2108 `feat/glm53flash-f5-nope-dsa` → `feat/glm53flash-f4-mhc`
5. #2109 `feat/glm53flash-f1-vl-data` → `feat/glm53flash-f5-nope-dsa`
6. #2110 `feat/glm53flash-f2-vision-tower` → `feat/glm53flash-f1-vl-data`
7. #2111 `feat/glm53flash-f6-text-moe` → `feat/glm53flash-f2-vision-tower`

> Base is `main`. Review only this PR's own diff.

---

## Summary

Stack layer 1/7 of GLM-5.3-Flash support. Two commits:

1. **[Refactor] Unify the DTensor unshard contract behind materialize_full** — GLM-5.3-Flash's later layers (mHC, KDA, NoPE-DSA) each need to unwrap a parameter before handing it to a kernel that doesn't understand `DTensor`. This existed in two unsafe flavours already (`mhc.unshard_hc_params` used `full_tensor()`, `kda._to_local`/`nope_dsa_mla._to_local` used `to_local()`) — both are wrong under `Shard` placement (silent wrong-rank slice, or an all-gather with no matching backward communication). `xtuner/v1/utils/dtensor.py::materialize_full` replaces all three call sites project-wide: accepts a plain tensor or `Replicate` DTensor, raises on `Shard`. Also folds the DTensor-aware `LayerNorm` that glm52/glm53 had each defined separately into the shared `xtuner/v1/module/rms_norm/layer_norm.py`.
2. **[Feature] Add GLM-5.3-Flash F0: 25B cropped reference checkpoint builder** — crops the published FP8 62-shard GLM-5.3-Flash checkpoint to a 5-main-layer + original-MTP-layer BF16 checkpoint (~24.9B params) for local validation, per `doc/xtuner_glm5p3flash_design.md` F0. Includes the design assessment doc (`doc/xtuner_glm5p3flash_design.{md,py}`, `doc/glm5p3flash_vs_glm5p2.md`, `demo/glm5p3_flash_demo.py`) that this and every later F* milestone in the stack implements against.

Verified against the real checkpoint: 24.95B total params (matches the design doc's 24.9B target), config loads under pinned `transformers==5.17.0`, and `Glm5NextForConditionalGeneration.from_pretrained` runs a real forward pass with finite loss. Also cross-validated the FP8 dequant math against the official native-BF16 checkpoint release (bit-for-bit on unquantized tensors, FP8-consistent tolerance elsewhere) and switched the canonical crop path to the native-BF16 source.

## Test Plan

- `tests/utils/test_dtensor.py::TestMaterializeFull`: pass-through for a plain tensor, unwrap for `Replicate`, `pytest.raises` for `Shard`.
- `tests/model/test_glm53_25b_crop.py`, including `test_glm53_25b_crop_passes_through_native_bf16_weights_unchanged` for the no-quantization_config input path.

```

### Changed Files

```
demo/glm5p3_flash_demo.py (+23 -0)
doc/glm5p3flash_vs_glm5p2.md (+464 -0)
doc/progress.md (+106 -0)
doc/xtuner_glm5p3flash_design.md (+1033 -0)
doc/xtuner_glm5p3flash_design.py (+1340 -0)
tests/model/test_glm53_25b_crop.py (+211 -0)
tests/utils/test_dtensor.py (+50 -0)
xtuner/tools/model_converters/make_glm53_25b_hf.py (+218 -0)
xtuner/v1/model/moe/glm52/dsa_mla.py (+1 -32)
xtuner/v1/module/rms_norm/__init__.py (+2 -1)
xtuner/v1/module/rms_norm/layer_norm.py (+41 -0)
xtuner/v1/utils/dtensor.py (+34 -0)
```

---

## Stack 2/7: PR #2106 — [Feature] Add GLM-5.3-Flash F3: Kimi Delta Attention (KDA)

- **URL**: https://github.com/InternLM/xtuner/pull/2106
- **State**: OPEN
- **Head branch**: `feat/glm53flash-f3-kda`
- **Base branch**: `feat/glm53flash-materialize-full-f0`
- **Commits**: 1

### Commits

- `8b519c651238` **[Feature] Add GLM-5.3-Flash F3: Kimi Delta Attention (KDA)** (jayhenry, claude, 2026-09-23)

### PR Message

```markdown
Stack (bottom to top):

1. #2105 `feat/glm53flash-materialize-full-f0` → `main`
2. **#2106 `feat/glm53flash-f3-kda` → `feat/glm53flash-materialize-full-f0`** ← you are here
3. #2107 `feat/glm53flash-f4-mhc` → `feat/glm53flash-f3-kda`
4. #2108 `feat/glm53flash-f5-nope-dsa` → `feat/glm53flash-f4-mhc`
5. #2109 `feat/glm53flash-f1-vl-data` → `feat/glm53flash-f5-nope-dsa`
6. #2110 `feat/glm53flash-f2-vision-tower` → `feat/glm53flash-f1-vl-data`
7. #2111 `feat/glm53flash-f6-text-moe` → `feat/glm53flash-f2-vision-tower`

> Base is #2105's branch (layer 1). Review only this PR's own diff.

---

## Summary

Stack layer 2/7 of GLM-5.3-Flash support (base: layer 1, F0 + materialize_full).

Adds `xtuner/v1/module/attention/kda.py` (`KDAConfig` / `KimiDeltaAttention`) per `doc/xtuner_glm5p3flash_design.md` F3, with module structure and parameter names matching the published checkpoint layout (separate `q/k/v_conv1d`, low-rank `f_a_proj->f_b_proj` forget gate with fp32 `A_log`/`dt_bias`, low-rank `g_a_proj->g_b_proj` output gate).

**Root cause of a real correctness bug found while implementing:** the installed `fla` 0.4.2's `chunk_kda`/`fused_recurrent_kda` call convention differs from the `xtuner-ncp-k3` reference — it doesn't accept `A_log`/`dt_bias`/`use_beta_sigmoid_in_kernel`, and silently swallows them into `**kwargs` without raising, producing a KDA that trains but computes the wrong gate. Fixed by computing the gate externally via `fla.ops.kda.gate.fused_kda_gate` before calling the kernel, matching HF's own `Glm5NextTextLinearAttention` reference forward. A regression test pins these three kwargs' absence from `chunk_kda`'s signature so this can't silently regress.

Also relaxes `TransformerConfig.linear_attention` to `GatedDeltaNetConfig | KDAConfig | None` (mypy-strict plumbing only, no behavior change for existing GatedDeltaNet layers).

## Test Plan

`tests/model/test_glm53_kda.py` (5 tests, single/2-GPU): gate parity against fla's naive reference, module-level parity against HF's `Glm5NextTextLinearAttention` (weights bridged per the published<->HF key mapping), packed multi-document isolation, and Ulysses SP parity.

```

### Changed Files

```
doc/progress.md (+42 -1)
tests/model/test_glm53_kda.py (+202 -0)
xtuner/v1/model/base.py (+2 -2)
xtuner/v1/model/dense/dense.py (+2 -1)
xtuner/v1/model/moe/moe.py (+3 -2)
xtuner/v1/module/__init__.py (+4 -0)
xtuner/v1/module/attention/__init__.py (+3 -0)
xtuner/v1/module/attention/kda.py (+393 -0)
xtuner/v1/module/decoder_layer/dense_decoder_layer.py (+2 -2)
xtuner/v1/module/decoder_layer/moe_decoder_layer.py (+12 -8)
```

---

## Stack 3/7: PR #2107 — [Feature] Add GLM-5.3-Flash F4: mHC four-stream residual

- **URL**: https://github.com/InternLM/xtuner/pull/2107
- **State**: OPEN
- **Head branch**: `feat/glm53flash-f4-mhc`
- **Base branch**: `feat/glm53flash-f3-kda`
- **Commits**: 1

### Commits

- `8ffc8e2cf4e6` **[Feature] Add GLM-5.3-Flash F4: mHC four-stream residual** (jayhenry, claude, 2026-09-23)

### PR Message

```markdown
Stack (bottom to top):

1. #2105 `feat/glm53flash-materialize-full-f0` → `main`
2. #2106 `feat/glm53flash-f3-kda` → `feat/glm53flash-materialize-full-f0`
3. **#2107 `feat/glm53flash-f4-mhc` → `feat/glm53flash-f3-kda`** ← you are here
4. #2108 `feat/glm53flash-f5-nope-dsa` → `feat/glm53flash-f4-mhc`
5. #2109 `feat/glm53flash-f1-vl-data` → `feat/glm53flash-f5-nope-dsa`
6. #2110 `feat/glm53flash-f2-vision-tower` → `feat/glm53flash-f1-vl-data`
7. #2111 `feat/glm53flash-f6-text-moe` → `feat/glm53flash-f2-vision-tower`

> Base is #2106's branch (layer 2). Review only this PR's own diff.

---

## Summary

Stack layer 3/7 of GLM-5.3-Flash support (base: layer 2, F3 KDA).

Adds `xtuner/v1/module/decoder_layer/mhc.py` (`MHCConfig` / `hc_split_sinkhorn` / `hc_pre` / `hc_post` / `unshard_hc_params`) and `xtuner/v1/ops/hc_post.py` (fused Triton `hc_post_fused`), per `doc/xtuner_glm5p3flash_design.md` F4. The core math is ported from xtuner's `dsv4` branch (DeepSeek-V4's Hyper-Connections) into the model-agnostic public location the design doc calls for, dropping V4's `XTUNER_V4_HF_PARITY` global toggle and the unported 721-line TileKernels backend (left as a documented `NotImplementedError` gap rather than a blind, unvalidated port).

Adds `xtuner/v1/model/moe/glm53/decoder_layer.py`: `Glm53DenseDecoderLayer`/`Glm53MoEDecoderLayer`, which only override the pre/post-forward seams so the existing EP/dispatcher/domino-micro-batch pipeline stays untouched — the mHC residual rides through as an opaque `_MHCResidual` payload in place of a plain `Tensor`. `mhc_cfg=None` degrades both layers to ordinary residual math unchanged, for reuse by the MTP layer later in the stack.

**Also fixes a real bug surfaced while testing:** KDA's `dt_bias`/`A_log` were left `torch.empty`-uninitialized; now initialized to match HF's `Glm5NextTextForgetGate._init_weights` policy.

## Test Plan

26 tests across `tests/model/test_glm53_mhc.py` (Sinkhorn math + exact HF parity in fp32, bf16 parity within ULP tolerance), `tests/ops/test_hc_post.py` (fused kernel vs eager reference, forward/backward/compile), and `tests/model/test_glm53_decoder_layer.py` (both decoder layers, `mhc_cfg=None` passthrough equivalence, gradient flow).

```

### Changed Files

```
doc/progress.md (+50 -1)
tests/model/test_glm53_decoder_layer.py (+212 -0)
tests/model/test_glm53_mhc.py (+188 -0)
tests/ops/test_hc_post.py (+109 -0)
xtuner/v1/model/moe/glm53/__init__.py (+5 -0)
xtuner/v1/model/moe/glm53/decoder_layer.py (+251 -0)
xtuner/v1/module/attention/kda.py (+11 -1)
xtuner/v1/module/decoder_layer/mhc.py (+262 -0)
xtuner/v1/ops/hc_post.py (+269 -0)
```

---

## Stack 4/7: PR #2108 — [Feature] Add GLM-5.3-Flash F5: NoPE DSA + KPool indexer + clamped SwiGLU

- **URL**: https://github.com/InternLM/xtuner/pull/2108
- **State**: OPEN
- **Head branch**: `feat/glm53flash-f5-nope-dsa`
- **Base branch**: `feat/glm53flash-f4-mhc`
- **Commits**: 1

### Commits

- `8557cdafd8a2` **[Feature] Add GLM-5.3-Flash F5: NoPE DSA + KPool indexer + clamped Sw…** (jayhenry, 2026-09-23)

### PR Message

```markdown
Stack (bottom to top):

1. #2105 `feat/glm53flash-materialize-full-f0` → `main`
2. #2106 `feat/glm53flash-f3-kda` → `feat/glm53flash-materialize-full-f0`
3. #2107 `feat/glm53flash-f4-mhc` → `feat/glm53flash-f3-kda`
4. **#2108 `feat/glm53flash-f5-nope-dsa` → `feat/glm53flash-f4-mhc`** ← you are here
5. #2109 `feat/glm53flash-f1-vl-data` → `feat/glm53flash-f5-nope-dsa`
6. #2110 `feat/glm53flash-f2-vision-tower` → `feat/glm53flash-f1-vl-data`
7. #2111 `feat/glm53flash-f6-text-moe` → `feat/glm53flash-f2-vision-tower`

> Base is #2107's branch (layer 3). Review only this PR's own diff.

---

## Summary

Stack layer 4/7 of GLM-5.3-Flash support (base: layer 3, F4 mHC).

Implements the F5 milestone: the NoPE (`qk_rope_head_dim=0`) DeepSeek Sparse Attention layers, their KPool indexer (pools of `index_kpool` consecutive tokens scored together instead of per-token top-k), a new `flash_mla_cudnn` SparseMLA backend (FlashMLA forward + existing cuDNN backward), and the clamped SwiGLU activation used by GLM-5.3-Flash's dense/shared/MoE MLPs.

Key pieces:
- `xtuner/v1/ops/act_fn.py`: `native_clamped_swiglu` + `MoEActFnConfig` support, wired through `DenseMLP`/`MoEMLP`.
- `xtuner/v1/ops/sparse_mla/kpool.py`: pool layout, causal ranges, and top-k pool selection (torch reference + TileLang-backed production path).
- `xtuner/v1/ops/sparse_mla/flash_mla_cudnn.py`: new SparseMLA backend.
- `xtuner/v1/model/moe/glm53/nope_dsa_mla.py`: `NoPEDSAMLAConfig`, `KPoolIndexer`, `NoPEDSAMultiLatentAttention`.
- `xtuner/v1/ops/sparse_mla/protocol.py`: `KPoolIndexerBackend`/`KPoolTopKIndicesProtocol` (GLM-5.3-Flash's KPool only supports `torch`/`tilelang`, unlike GLM-5.2's 6-way `DSAIndexerBackend`).

Also folds in what were originally two separate follow-up commits, now part of this PR:
- **DSA backends get explicit defaults instead of inheriting**: `indexer_backend` used to default to `None` and fall back to `sparse_mla_backend`, but the two name different, only partially-overlapping backend vocabularies (`flash_mla_cudnn` is SparseMLA-only; `deep_gemm_fp8`/`cute_dsl` are indexer-only), so the fallback could hand the indexer factory a backend it has no implementation for. Both `DSAMLAConfig`/`NoPEDSAMLAConfig` fields now default to `tilelang` directly and explicitly.
- **Bound the KPool indexer's logits tile with query chunking**, mirroring the existing tilelang DSA selector's query-chunk bound.

Known gaps recorded in `doc/progress.md`: tilelang `sparse_mla_backend` and `deep_gemm_fp8` `indexer_backend` are not implemented for NoPE (explicit `NotImplementedError`, not silent fallback); KPool/NoPE-DSA sequence-parallel path is written per design but not yet GPU-tested at `sp_size>1`.

## Test Plan

`tests/model/test_glm53_dsa.py` (12), `test_glm53_nope_dsa_mla.py` (2), `test_flash_mla_cudnn_sparse_mla.py` (3) all pass; GLM-5.2's existing `tests/module/attention/test_dsa_mla.py` (15) rerun clean against the shared `tilelang.py`/`dsa_mla.py` edits, confirming no regression. Numerical oracle: transformers 5.17.0's `glm5_next` model.

```

### Changed Files

```
doc/progress.md (+64 -1)
doc/xtuner_glm5p3flash_design.md (+1 -1)
examples/v1/config/sft_glm5p2.py (+4 -4)
tests/engine/test_glm52_moe_train_engine.py (+1 -0)
tests/model/test_glm53_dsa.py (+473 -0)
tests/model/test_glm53_nope_dsa_mla.py (+226 -0)
tests/module/attention/test_dsa_mla.py (+20 -0)
tests/module/test_dense_decoder_layer.py (+1 -0)
tests/ops/test_flash_mla_cudnn_sparse_mla.py (+108 -0)
tests/ops/test_tilelang_dsa_topk_chunk.py (+1 -0)
xtuner/v1/data_proto/sequence_context.py (+10 -0)
xtuner/v1/model/moe/glm52/dsa_mla.py (+7 -5)
xtuner/v1/model/moe/glm53/__init__.py (+8 -1)
xtuner/v1/model/moe/glm53/nope_dsa_mla.py (+349 -0)
xtuner/v1/module/decoder_layer/dense_decoder_layer.py (+6 -4)
xtuner/v1/module/decoder_layer/moe_decoder_layer.py (+18 -6)
xtuner/v1/ops/act_fn.py (+41 -0)
xtuner/v1/ops/sparse_mla/__init__.py (+34 -1)
xtuner/v1/ops/sparse_mla/cudnn_dsa.py (+2 -2)
xtuner/v1/ops/sparse_mla/flash_mla.py (+2 -2)
xtuner/v1/ops/sparse_mla/flash_mla_cudnn.py (+114 -0)
xtuner/v1/ops/sparse_mla/kpool.py (+421 -0)
xtuner/v1/ops/sparse_mla/protocol.py (+36 -1)
xtuner/v1/ops/sparse_mla/tilelang.py (+59 -9)
xtuner/v1/utils/init_weight.py (+15 -0)
```

---

## Stack 5/7: PR #2109 — [Feature] Add GLM-5.3-Flash F1: VL data preprocessing pipeline

- **URL**: https://github.com/InternLM/xtuner/pull/2109
- **State**: OPEN
- **Head branch**: `feat/glm53flash-f1-vl-data`
- **Base branch**: `feat/glm53flash-f5-nope-dsa`
- **Commits**: 5

### Commits

- `ad93479eeeb4` **[Feature] Add GLM-5.3-Flash F1: VL data preprocessing pipeline** (jayhenry, claude, 2026-09-23)
- `35d1c1d0c2af` **[Fix] Register glm53_vl_sft_collator so VL training can select it** (jayhenry, claude, 2026-09-24)
- `7677b73428e7` **[Fix] Export Glm53VLTokenizeFnConfig from xtuner.v1.datasets** (jayhenry, claude, 2026-09-24)
- `99ea7b3b949d` **[Fix] Make Glm53VLTokenizeFnConfig constructible without a chat_template** (jayhenry, claude, 2026-09-24)
- `0c2a272bf9cc` **[Fix] Stop GLM-5.3-Flash VL samples from silently training as fakes** (jayhenry, claude, 2026-09-24)

### PR Message

```markdown
Stack (bottom to top):

1. #2105 `feat/glm53flash-materialize-full-f0` → `main`
2. #2106 `feat/glm53flash-f3-kda` → `feat/glm53flash-materialize-full-f0`
3. #2107 `feat/glm53flash-f4-mhc` → `feat/glm53flash-f3-kda`
4. #2108 `feat/glm53flash-f5-nope-dsa` → `feat/glm53flash-f4-mhc`
5. **#2109 `feat/glm53flash-f1-vl-data` → `feat/glm53flash-f5-nope-dsa`** ← you are here
6. #2110 `feat/glm53flash-f2-vision-tower` → `feat/glm53flash-f1-vl-data`
7. #2111 `feat/glm53flash-f6-text-moe` → `feat/glm53flash-f2-vision-tower`

> Base is #2108's branch (layer 4). Review only this PR's own diff.

---

## Summary

Stack layer 5/7 of GLM-5.3-Flash support (base: layer 4, F5 NoPE DSA). The F1 milestone — chat template, `TokenizeFn` (cache/runtime dual path), `DataItem`, and `Collator`/`SequenceContext` plumbing for image/video SFT — plus the three fixes that make it actually reachable from a training config.

### F1 core

- `xtuner/v1/data_proto/messages/glm53_chat.py`: `render_glm53_chat`/`Glm53ChatMessages`, adapted from the landed `glm52_chat.py` at its two documented diffs. Verified byte-for-byte against the real `chat_template.jinja`.
- `xtuner/v1/datasets/mllm_tokenize_fn/glm53_vl_tokenize_fn.py`: placeholder expansion and `mm_token_type_ids` reuse the real `Glm5NextProcessor`'s own public `replace_image_token`/`replace_video_token`/`create_mm_token_type_ids` in both the cache and runtime paths, so both run identical HF code rather than two independent reimplementations.
- `xtuner/v1/datasets/collator.py`: `build_text_ctx_labels` also returns `kept_indices`; `glm53_vl_sft_collator` built on top.

Two discrepancies were root-caused by direct package inspection rather than trusting the design doc: transformers 5.17.0 has no `Glm5NextVideoProcessor.get_number_of_video_patches` (the cache-path geometry is reproduced bit-for-bit via `sample_frames` + `smart_resize` instead), and an earlier version expanded the whole `<|begin_of_image|>…<|end_of_image|>` marker instead of just the inner `<|image|>`, undercounting `num_tokens` by 2 per media item.

### Fixes that unblock VL training

These were found by running end-to-end VL SFT; each one fired before a single training step.

1. **`glm53_vl_sft_collator` was unreachable.** `DataloaderConfig.build_collator()` only knew the three older collators by name, and the dotted-path fallback could not supply the two extra arguments this collator needs (`image_token_id`/`merge_unit`, used to re-check the placeholder↔patch-count invariant after pack truncation). Any VL config died in the dataloader with a `TypeError`. Registers the name and adds a generic `collator_kwargs` bound into the collator partial; the name→function dispatch collapses into one lookup table so every branch, dotted path included, goes through the same binding.
2. **`Glm53VLTokenizeFnConfig` could not be constructed.** The class documents that GLM-5.3-Flash renders through `render_glm53_chat` instead of a `CHAT_TEMPLATE_MAP` entry, and `build()` never reads `chat_template` — but the base class declares the field required, so the documented construction raised `ValidationError: chat_template Field required`.
3. **Not exported from the package root**, unlike every other VL tokenize-fn config, so configs had to reach into the submodule by path.

### Samples silently training as fakes

Surfaced by the VL e2e matrix: `tgs` sat at roughly half of `seqlen_tgs` on every profile, where the text matrix keeps them within 0.05%. They differ only by padding (`mask.sum()` vs `mask.numel()`), so VL packs were ~50% padding — one step packed 131 real tokens into a 16384 slot — even though the corpus held enough tokens to fill them. Comparing each sample's cached `num_tokens` (what the packer budgets with) against its runtime length found **17% of samples cached at 4096 but training as 33-token fakes**. Two defects in the cache/runtime split compounded:

1. **Frame-folder videos were indexed against the original video.** Both paths built `VideoMetadata` from `origin_video_length`/`origin_fps` (182 frames @30fps), but a frame folder holds the video *after* resampling — `processed_video_length` frames at `processed_fps` (23 @4fps). Runtime indexed a 23-file folder with original frame numbers and raised `IndexError: list index out of range`; the dataset swallowed it and substituted a fake. Sampling now uses the processed sequence for frame folders — qwen3_vl's `calc_frame_info` rule 3, which also makes the frame timestamps correct. The folder/file decision is read from the path string (the trailing-`/` convention qwen3_vl's reader keys on) rather than the filesystem, so the cache path, which never touches media, reaches the same answer as the runtime path.
2. **Over-length visual samples were packed at their truncated length.** With the index fixed, those samples expand to 14,352 placeholders, far past `max_length`. Truncating text is harmless; truncating *through* a visual span leaves fewer placeholders than patches. The cache path reported the truncated length anyway, so the packer reserved space and runtime then rejected the sample — again a fake. The cache path now reports such samples as damaged (`num_tokens=0`) so they are filtered before packing, the same check qwen3_vl's cache path makes.

Both escaped the existing tests for concrete reasons: the video fixture only set `origin_*` on a folder holding exactly that many frames, so origin and processed coincided; and the truncation test only asserted that the *runtime* path raises — precisely the behavior that gets silently turned into a fake — while the cache path, where the damage is decided, was never checked.

One consequence worth stating plainly: before this fix, **no video sample ever actually trained** in the VL e2e runs, despite the corpus containing 24 of them.

## Test Plan

- `tests/datasets/test_glm53_chat.py` (byte-exact vs HF jinja), `test_glm53_vl_collator.py` (includes `build_text_ctx_labels` regression coverage for the qwen3_vl/intern_s1 collators that share the builder), `test_glm53_vl_tokenize_fn.py` (two tests assert `input_ids`/`mm_token_type_ids` match a real end-to-end `processor()` call token-for-token). CPU only.
- `test_glm53_collator_is_reachable_by_name_from_dataloader_config` builds the collator through `DataloaderConfig` and calls it with only the dataloader-supplied arguments — the exact path that was broken.
- Exercised end to end by the VL SFT run in layer 7.
- `test_glm53_frame_folder_samples_the_processed_frames`: a folder resampled from a longer original (180 frames @30fps → 6 @2fps); cache and runtime agree and runtime does not raise. `test_glm53_visual_truncation_is_dropped` now also asserts the cache path returns `num_tokens == 0`. Both **fail on the previous code with the exact production errors** (`IndexError: list index out of range`; `assert 20 == 0`) and pass with the fix.
- Over the whole ci_vl corpus: 0 cached/runtime length mismatches across the kept samples, with over-length samples filtered before packing (previously 10 of the first 60 mismatched).

```

### Changed Files

```
doc/progress.md (+72 -1)
doc/xtuner_glm5p3flash_design.md (+1 -1)
doc/xtuner_glm5p3flash_design.py (+8 -3)
tests/datasets/test_glm53_chat.py (+87 -0)
tests/datasets/test_glm53_vl_collator.py (+189 -0)
tests/datasets/test_glm53_vl_tokenize_fn.py (+269 -0)
xtuner/v1/data_proto/messages/__init__.py (+2 -1)
xtuner/v1/data_proto/messages/glm53_chat.py (+263 -0)
xtuner/v1/data_proto/sequence_context.py (+58 -0)
xtuner/v1/datasets/__init__.py (+4 -0)
xtuner/v1/datasets/collator.py (+82 -1)
xtuner/v1/datasets/config.py (+25 -14)
xtuner/v1/datasets/data_item.py (+8 -0)
xtuner/v1/datasets/mllm_tokenize_fn/__init__.py (+3 -0)
xtuner/v1/datasets/mllm_tokenize_fn/glm53_vl_tokenize_fn.py (+491 -0)
```

---

## Stack 6/7: PR #2110 — [Feature] Add GLM-5.3-Flash F2: vision tower + projector (eager)

- **URL**: https://github.com/InternLM/xtuner/pull/2110
- **State**: OPEN
- **Head branch**: `feat/glm53flash-f2-vision-tower`
- **Base branch**: `feat/glm53flash-f1-vl-data`
- **Commits**: 5

### Commits

- `897ad5ac85d2` **[Feature] Add GLM-5.3-Flash F2: vision tower + projector (eager)** (jayhenry, claude, 2026-09-24)
- `22be8585d2ea` **fix(glm53): match vision RMSNorm rounding and reject unfinished SP** (ShilohYu, cursoragent, 2026-09-24)
- `09d1d99313ae` **[Feature] Implement GLM-5.3-Flash Vision SP, compile cfg and the flas…** (jayhenry, claude, 2026-09-24)
- `53b723f5b47e` **[Fix] Move pixel_values to the vision tower's device before patch emb…** (jayhenry, claude, 2026-09-24)
- `26438cdffe07` **[Fix] Recompute the GLM-5.3-Flash ViT blocks under FSDP** (jayhenry, claude, 2026-09-24)

### PR Message

```markdown
Stack (bottom to top):

1. #2105 `feat/glm53flash-materialize-full-f0` → `main`
2. #2106 `feat/glm53flash-f3-kda` → `feat/glm53flash-materialize-full-f0`
3. #2107 `feat/glm53flash-f4-mhc` → `feat/glm53flash-f3-kda`
4. #2108 `feat/glm53flash-f5-nope-dsa` → `feat/glm53flash-f4-mhc`
5. #2109 `feat/glm53flash-f1-vl-data` → `feat/glm53flash-f5-nope-dsa`
6. **#2110 `feat/glm53flash-f2-vision-tower` → `feat/glm53flash-f1-vl-data`** ← you are here
7. #2111 `feat/glm53flash-f6-text-moe` → `feat/glm53flash-f2-vision-tower`

> Base is #2109's branch (layer 5). Review only this PR's own diff.

---

## Summary

Stack layer 6/7 of GLM-5.3-Flash support (base: layer 5, F1 VL data pipeline). The F2 vision tower and projector, taken from "eager parity only" to something that survives production training.

### F2 core

Native XTuner vision tower and projector (`xtuner/v1/model/compose/glm53/`), split per the design doc's §5.2 module boundary: HF's single `Glm5NextVisionModel` becomes `vision_tower` (patch_embed → 2D axial RoPE → 24 blocks → post_layernorm) plus `multi_modal_projector` (Conv2d downsample + clamped-SwiGLU merger). `cu_seqlens`/`position_ids` are reimplemented natively rather than imported from `transformers.vision_utils`, matching qwen3_vl's established pattern. The vision RMSNorm follows HF/Automodel's cast-then-affine rounding, which matters in bf16.

### Vision sequence parallel (design doc §9.3/§9.6)

Previously the tower rejected `sequence_parallel_mesh.size() > 1` outright — the attention had half an Ulysses path but nothing sharded the patch sequence, which silently computes the wrong attention. Now implemented properly:

- `_shard_patches_for_sequence_parallel` pads the global patch sequence up to a multiple of `sp_size * spatial_merge_size ** 2` and splits it contiguously. **Merge alignment** is a hard constraint: the projector's `downsample` treats each consecutive group of 4 patches as one 2×2 block, so a shard boundary inside a group would merge patches owned by two different ranks — alignment is also what lets each rank run the projector on its own shard (local projector). **Equal length** is what Ulysses all-to-all requires.
- The padding is declared as an extra `grid_thw` row rather than appended silently, so it forms its own `cu_seqlens` segment: padding patches can never attend to, or be attended by, a real image.
- Position ids are sharded to match the local patches (RoPE runs before the all-to-all); `cu_seqlens` stays global (after the all-to-all, attention sees the whole sequence).

### Three production bugs

1. **`attn_impl="flash_attention"` was broken, not merely unverified.** It died with `Expected a value of type 'Tensor' for argument 'max_seqlen_q' but instead found type 'int'` — the tower was the only attention call site materializing `max_seqlen` via `int(...item())`, which `flash_attn_varlen_func` v2 rejects and which also forces a device sync. Now passed as a tensor, like every other call site does with `SequenceContext.max_length_q`.
2. **`pixel_values` was never moved to the device.** `SequenceContext` deliberately leaves the media tensors on CPU (documented there as "sharded by the model side, then moved"), so real training died on step 1 with `Input type (CPUBFloat16Type) and weight type (CUDABFloat16Type) should be the same`. Every unit test passed device tensors in, so nothing caught it. The move happens after the SP split, so only this rank's shard is transferred.
3. **The ViT blocks had no activation checkpointing** — the actual cause of multimodal OOM. The tower runs over *raw* patches, and a packed VL sample carries far more of them than the LLM sequence has tokens (an 8192-token pack measured `img_tokens: 14688`; a 16384-token pack ~31k), so 24 blocks of retained attention + 4096-wide MLP activations dominated peak memory. VL SFT OOMed in backward at 16k pack and still died on step 2 at 8k, long before the language tower was near its limit. Now wired through `apply_activation_checkpointing` honouring `vision_recompute_ratio`/`checkpoint_preserve_rng_state`, the same knobs qwen3_vl's tower uses. Peak drops to 85.85 GB at 16k pack.

### Compile config

`default_compile_cfg` was empty for both modules, so the vision half stayed eager even under `MODEL_COMPILE=1`. The ViT block takes `fullgraph=False` (it dispatches into an attention kernel, and under SP it contains collectives); the projector takes a full graph.

## Test Plan

tests/model/test_glm53_vision.py 14/14:

- Real-checkpoint weight mapping: 339 + 8 = 347 parameters, `from_hf(strict=False)` with no missing/unloaded keys.
- Forward parity vs HF `Glm5NextVisionModel` at `rtol=0, atol=0` (fp32 and bf16): single image, multi-tubelet video, multi-image batch, plus backward gradient flow.
- `TestGlm53VisionSequenceParallel::test_sp_tower_and_projector_match_non_sp` (2 GPU): 20 patches over two images — deliberately not a multiple of `sp_size * merge_unit`, exercising the padding branch — tower + projector under SP, gathered and trimmed, matches the non-SP result.
- `test_unaligned_patch_count_is_rejected_under_sp`: a patch count that would split a merge block raises instead of mis-merging.
- `test_flash_attention_matches_eager` (GPU): the flash path runs and agrees with the HF-aligned eager baseline.
- `test_cpu_pixel_values_are_moved_to_device`: CPU `pixel_values` with a device `grid_thw`, the exact shape `SequenceContext` produces.

## Known gaps

Vision-side FSDP2 multi-GPU parity is still covered only by the end-to-end smoke, not a dedicated `test_vision_fsdp_parity`.

```

### Changed Files

```
doc/progress.md (+71 -1)
tests/model/test_glm53_vision.py (+366 -0)
xtuner/v1/model/compose/glm53/__init__.py (+14 -0)
xtuner/v1/model/compose/glm53/glm53_config.py (+72 -0)
xtuner/v1/model/compose/glm53/modeling_projector.py (+116 -0)
xtuner/v1/model/compose/glm53/modeling_vision.py (+468 -0)
xtuner/v1/model/compose/glm53/vision_utils.py (+21 -0)
```

---

## Stack 7/7: PR #2111 — [Feature] Add GLM-5.3-Flash F6 core: text model + MTP + compose model

- **URL**: https://github.com/InternLM/xtuner/pull/2111
- **State**: OPEN
- **Head branch**: `feat/glm53flash-f6-text-moe`
- **Base branch**: `feat/glm53flash-f2-vision-tower`
- **Commits**: 10

### Commits

- `17b39eaf4ef5` **[Feature] Add GLM-5.3-Flash F6 core: text model + MTP + compose model** (jayhenry, claude, 2026-09-24)
- `b0f76b136bde` **[Feature] Support sequence parallel in the GLM-5.3-Flash VL compose m…** (jayhenry, claude, 2026-09-24)
- `ebc25482a4a5` **[Feature] Add the GLM-5.3-Flash VL end-to-end SFT config and launcher** (jayhenry, claude, 2026-09-24)
- `f77813cf5026` **[Docs] Record the multimodal enablement round and its acceptance matrix** (jayhenry, claude, 2026-09-24)
- `552f592e3d74` **[Fix] Make chunk_kda traceable so compile stops regressing throughput** (jayhenry, claude, 2026-09-24)
- `bf33fc4fa36e` **[Docs] Record the torch.compile throughput regression and its fix** (jayhenry, claude, 2026-09-24)
- `0fb544f9d1ff` **[Fix] Take the KDA gate and short convolution off the graph-break path** (jayhenry, claude, 2026-09-24)
- `1177d37cde87` **[Docs] Record the remaining KDA ports and the Qwen3.5-VL comparison** (jayhenry, claude, 2026-09-24)
- `7bbf4398c13a` **[Fix] Train GLM-5.3-Flash VL packs that mix image and video samples** (jayhenry, claude, 2026-09-24)
- `016500f04780` **[Fix] Call the GLM-5.3-Flash vision tower once per pack** (jayhenry, 2026-09-27)

### PR Message

```markdown
Stack (bottom to top):

1. #2105 `feat/glm53flash-materialize-full-f0` → `main`
2. #2106 `feat/glm53flash-f3-kda` → `feat/glm53flash-materialize-full-f0`
3. #2107 `feat/glm53flash-f4-mhc` → `feat/glm53flash-f3-kda`
4. #2108 `feat/glm53flash-f5-nope-dsa` → `feat/glm53flash-f4-mhc`
5. #2109 `feat/glm53flash-f1-vl-data` → `feat/glm53flash-f5-nope-dsa`
6. #2110 `feat/glm53flash-f2-vision-tower` → `feat/glm53flash-f1-vl-data`
7. **#2111 `feat/glm53flash-f6-text-moe` → `feat/glm53flash-f2-vision-tower`** ← you are here

> Base is #2110's branch (layer 6). Review only this PR's own diff.

---

## Summary

Stack layer 7/7 (top) of GLM-5.3-Flash support (base: layer 6, F2 vision tower). Covers the F6 text model + MTP + compose model, and the multimodal enablement that makes VL SFT actually runnable: sequence-parallel splice, and the VL training config/launcher.

Implements the core of the F6 milestone: the 45-layer KDA/NoPE-DSA text model with mHC four-stream residual, MTP, and the compose model wiring `vision_tower` + `multi_modal_projector` + `language_model` with image/video splice.

- `xtuner/v1/model/moe/glm53/glm53.py`: `Glm53TextMoEConfig`/`Glm53TextMoE`. Layer schedule (`[KDA,KDA,KDA,DSA]x11 + KDA`) read from the real checkpoint's `text_config.layer_types`. mHC's four-stream residual is expanded/collapsed once at the decoder-stack boundary, not per layer, so `MoE._forward`'s aux-loss/router bookkeeping is reused unmodified. MTP reuses the base `MTPLayer`/`MTPBlock` unmodified — the design doc's originally-planned `mtp.py` deliverable turned out unnecessary.
- `xtuner/v1/model/compose/glm53/modeling_glm53.py` + `Glm53BaseConfig`: `Glm53ForConditionalGeneration`. Image/video splice uses the global `mm_token_type_ids`, never `input_ids==video_token_id`. A placeholder-count mismatch raises immediately (design doc §16.2) — unlike Qwen3-VL's compose path, this never silently continues training on a corrupted splice.
- Registers the `"glm5.3"` chat template / tokenize-fn branch, `get_model_config_from_hf` dispatch, and the SFT training config/launch script (`examples/v1/config/sft_glm53.py`, `sft_glm53_tiny.sh`) for the 8-GPU end-to-end smoke suite.

**Root-caused four bugs found by end-to-end gradient-flow and 8-GPU training smoke tests** (minimal repros, not assumed):
1. FLA's `chunk_kda` Triton kernel silently drops backward gradients when `head_dim < 16` (a `tl.dot` minimum-K constraint) — not a production issue since real GLM-5.3-Flash uses `head_dim=128`.
2. DSA indexer showing no gradient with `freeze_dsa_indexer=False` traced to an existing, intentional `torch.no_grad()` wrap (not a bug).
3. `torch.compile` traced KDA's Triton-kernel-heavy forward under a strict boundary and hit disabled-function errors; fixed with `GLM53_MOE_{NON_EP,EP}_COMPILE_CFG`. Surfaced a separate real upstream limitation: FLA's `prepare_chunk_indices`/`prepare_lens` do a `.tolist()`-driven Python loop over `cu_seqlens`, incompatible with dynamo's dynamic-shape tracing (`_mark_dynamic` on the packed boundaries). Initially worked around by wrapping FLA's entry points in `torch._dynamo.disable()`, which made compile *run* but cost ~2.4x throughput; now fixed properly by owning the kernel behind `torch.library.custom_op` — see **torch.compile throughput** below.
4. Under `ep_size>1`, KDA's `o_norm` (FLA's `FusedRMSNormGated`) crashed with a Triton illegal-memory-access — its inherited forward reads `self.weight` directly, which stays a DTensor under EP. Fixed with a minimal `forward` override that unshards weight/bias first.

Known gaps (recorded explicitly, not silently skipped): the compose layer has no dedicated multi-GPU FSDP parity test beyond the smoke suite and the 2-GPU SP splice test below.

## Test Plan

- `tests/model/test_glm53_text_moe.py::TestGlm53TextMoEAccuracy::test_fsdp_accuracy`: XTuner forward loss vs real `transformers.Glm5NextForConditionalGeneration` on the F0 25B cropped checkpoint, for `(dispatcher, ep_size)` in `{(None, 1), ("all2all", 4), ("all2all", 8)}` — all pass. The `ep_size=4/8` cases double as the regression test for bug 4 above.
- `sft_glm53_tiny.sh`: single-node 8-GPU end-to-end SFT, real F0 checkpoint, `PACK_MAX_LENGTH=16384`, `TOTAL_STEP=20`. Default profile plus five combos (`SP_SIZE=2 EP_SIZE=4`; `EP_SIZE=8 SP_SIZE=1`; `XTUNER_ACTIVATION_OFFLOAD=0`; `FP8=1`; `MODEL_COMPILE=1`) all complete 20 steps with loss monotonically decreasing, no NaN/OOM. See the table below.

### Text end-to-end regression (`sft_glm53_tiny.sh`, single-node 8-GPU, real F0 25B cropped checkpoint)

`PACK_MAX_LENGTH=16384`, `TOTAL_STEP=20`. All six profiles complete 20 steps, loss monotonically decreasing, no NaN/OOM. Numbers are rank0; `mem` is `max_memory` / `reserved_memory` (GB); `tgs`/`seqlen_tgs`/`exp_tgs` are step-20 throughput (tokens/gpu/s); `time` is step1's own duration (includes one-time warmup/compile) plus the step1→step20 wall-clock gap:

| profile | step1 loss | step20 loss | grad_norm@20 | mem@20 (GB) | tgs@20 | seqlen_tgs@20 | exp_tgs@20 | time |
|---|---|---|---|---|---|---|---|---|
| default (`EP4 SP1 offload=1`) | 11.30479 | 10.31409 | 48.41 | 86.22 / 106.10 | 12203.1 | 12224.0 | 4212.1 | 76s |
| `SP2 EP4` | 11.31528 | 10.32397 | 48.40 | 75.15 / 96.00 | 9631.7 | 9631.7 | 2617.5 | 61s |
| `EP8 SP1` | 11.30479 | 10.31380 | 48.18 | 100.44 / 112.75 | 11865.9 | 11886.2 | 4692.9 | 68s |
| `offload=0` (`EP4 SP1`) | 11.30479 | 10.31401 | 48.41 | 85.58 / 105.69 | 12315.3 | 12336.4 | 4944.9 | 64s |
| `FP8=1` (`EP4 SP1 offload=1`) | 11.30908 | 10.18042 | 45.88 | 84.73 / 104.35 | 6159.6 | 6170.2 | 1871.5 | 174s |
| `MODEL_COMPILE=1` (`EP4 SP1 offload=1`) | 11.30469 | 10.31426 | 48.39 | 82.11 / 103.28 | **13381.1** | 13400.0 | 2270.0 | 86s |

- `SP2`'s per-GPU effective seqlen is halved (8192 vs 16384), so `tgs` is lower and wall-clock per step is also shorter than the default despite the added SP communication.
- `offload=0` is faster with higher `tgs` than the default (`offload=1`) — expected, since activation CPU-GPU transfer is skipped.
- `EP8` tracks the default (`EP4`) on loss/tgs; higher `mem` is because `dp_size = world_size/ep_size` drops from 2 to 1, so FSDP shards non-expert params over fewer dp ranks.
- `default`/`EP8`/`offload=0` step1/step20 loss match to 1e-4, as expected — these knobs change parallelism/memory strategy, not the math.
- `FP8=1`'s step20 loss differs from the bf16 baseline (10.18 vs 10.31) because FP8 genuinely changes the numerics on the (unquantized-`kv_b_proj`) absorbed-MLA path, not a bug; its `tgs` is roughly half the baseline's at this 25B/20-step smoke scale, consistent with per-tensor quantize/dequantize overhead not yet amortized.
- `MODEL_COMPILE=1` is now the fastest profile: steady-state 1.21s/step vs eager's 1.34s (tgs 13381 vs 12203, ~+10%), at lower peak memory. step20 loss (10.31426) matches the eager baseline (10.31409) to 1.8e-4, confirming compile does not change the math.



---

## Multimodal enablement

### Sequence parallel in the compose model

The compose model asserted `sequence_parallel_mesh` was None or size 1, so VL training could not use SP at all — the last piece of the Vision SP gap, now that the tower (layer 6) shards patches merge-aligned.

The splice is the only part that needs to know about SP. `input_ids` and `mm_token_type_ids` arrive already sharded (they are split together), so the local mask is what indexes the local embeddings; what the local mask cannot say is *which* global features belong to this rank. Image and video patches are concatenated into one tower call. `_splice` gathers the mask once, `_gather_visual_features` reassembles the globally-ordered features (trimming the tower's merge-alignment padding), and each modality is sliced at the offset of same-modality placeholders on preceding ranks. The gather is the autograd-aware one, so its backward reduce-scatters and hands each rank the gradient of exactly the shard it produced.

The placeholder↔feature count check now compares *global* totals, so a corrupted sample raises identically with and without SP. The pure-text dummy visual forward stays replicated: its gradient contribution is exactly zero, so sharding it would only add collectives.

### VL training entry point

`Glm53BaseConfig.from_hf` raised `NotImplementedError`, so the VL model could not be built from a checkpoint at all. It now reads the checkpoint's single `vision_config` and fans it out to both XTuner modules — tower and projector are one HF module that §5.2 splits in two — and delegates the text half to `Glm53TextMoEConfig.from_hf`.

`examples/v1/config/sft_glm53_vl.py` + `sft_glm53_vl_tiny.sh` mirror the text pair knob for knob, so both modalities run the same acceptance matrix. Media comes from the `ci_vl` corpus (image + video samples). FP8 stays on the language tower only, matching how the published checkpoint stores the vision half in bf16.

`hf_config` intentionally remains `None`: `_write_hf_index_and_config` then copies the source checkpoint's config/tokenizer into the export, which is self-consistent for a fine-tune that does not change the architecture.

### VL end-to-end regression (`sft_glm53_vl_tiny.sh`, single-node 8-GPU, real F0 25B cropped checkpoint)

`PACK_MAX_LENGTH=16384`, `SAMPLE_MAX_LENGTH=16384`, `TOTAL_STEP=20`, image + video data. All six profiles complete 20 steps, loss decreasing, no NaN/OOM. Numbers are rank0; `mem` is `max_memory` / `reserved_memory`; `time` is step 1's duration plus the step1→step20 wall-clock gap. This replaces the earlier table: that one trained no real video (see #2109).

| profile | step1 loss | step20 loss | grad_norm@20 | mem@20 (GB) | tgs@20 | seqlen_tgs@20 | exp_tgs@20 | time | img_tokens |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| default (`EP4 SP1 offload=1`) | 12.48675251 | 10.78624153 | 25.85 | 90.59 / 111.43 | 4465.1 | 5084.9 | 2740.3 | 67s | 56424 |
| `SP2 EP4` | 12.21398830 | 11.29156303 | 40.15 | 77.09 / 97.02 | 3674.0 | 3674.0 | 1926.0 | 44s | 31884 |
| `EP8 SP1` | 12.48675251 | 10.78038311 | 47.16 | 103.60 / 121.47 | 4468.9 | 5089.2 | 2563.4 | 71s | 56424 |
| `offload=0` (`EP4 SP1`) | 12.48675251 | 10.81396198 | 24.98 | 90.25 / 111.00 | 4488.9 | 5112.0 | 2701.5 | 69s | 56424 |
| `FP8=1` | 12.49881649 | 10.27666283 | 23.44 | 89.44 / 110.15 | 3972.2 | 4523.6 | 2886.5 | 70s | 56424 |
| `MODEL_COMPILE=1` | 12.48672199 | 10.78443432 | 25.04 | 86.27 / 109.73 | 5727.1 | 6522.1 | 2366.0 | 62s | 56424 |

- `tgs` counts real tokens (`mask.sum()`), `seqlen_tgs` counts every slot including padding (`mask.numel()`). The previous table had `seqlen_tgs` ≈ 2× `tgs` because packs were half empty: 17% of samples were cached at 4096 tokens and trained as 33-token fakes, so no video sample ever ran (#2109). With videos actually in the pack, rank0 fill on the non-SP profiles is 87–99% (mean 93%) and step-20 `seqlen_tgs / tgs` is 1.14. The remainder is the tail of a 16384 pack that the next ~14k-token video does not fit into. `SP2` packs at 8192 and fills exactly, so the two rates match.
- default / `EP8` / `offload=0` step-1 loss is **bit-identical** (12.48675251).
- `SP2` is not row-comparable. `GLOBAL_BATCH_SIZE = world/sp` and the sequence is split, so a step sees about half the tokens (`img_tokens` 31884 vs 56424) and step-20 loss stays higher (11.29 vs 10.79). Vision SP correctness is the 2-GPU bitwise test, not this smoke.
- The first baseline after the data fix hung on step 4. A pack that holds both an image sample and a video sample called the vision tower once per modality, while a rank with one modality called it once. The tower's FSDP mesh is the world group (8) and the language tower's is the dp group (2), so rank 0 was blocked in the language all-gather (`NumelOut/NumelIn = 2`) while the other ranks were still in a second vision all-gather (ratio 8) until the NCCL watchdog fired at 30 min. Image and video patches are now one tower call per pack. Each `grid_thw` row stays its own `cu_seqlens` segment, and the mixed-pack logits match splicing each modality from its own call, bit for bit.
- `MODEL_COMPILE=1` is the fastest row here (tgs 5727 vs 4465) and the lowest peak memory. step-1 loss matches eager to 3e-5.
- `FP8=1`'s step-20 loss is lower (10.28 vs 10.79), the same direction as text-side FP8: the absorbed-MLA path's numerics change.
- `EP8` peaks highest because `dp_size = world_size/ep_size` drops from 2 to 1.

### torch.compile throughput

Review caught that the compile row was originally *slower* than eager (tgs 5168 vs 12203) — backwards from what compile is for. The number was real (identical config, and identical per-step `text_tokens` between the two runs), but the explanation first written here ("few steps to amortize warmup") was wrong: eager settles at 1.34s/step by step 8 and holds within 0.5%, while compile never converged, bouncing 2.6–4.3s through step 20.

Diagnosis by measurement: `TORCH_LOGS=recompiles` ruled out runaway recompilation (200 recompiles, but bounded — they stop by step 2, dynamo cache versions cap at `/3`, and the guard failure is `requires_grad mismatch`, the expected one-time doubling from activation-checkpoint recompute). `TORCH_LOGS=graph_breaks` then put 32 breaks exactly on the `torch._dynamo.disable`d FLA entry points. GLM-5.3-Flash is KDA-dominated (34 of 45 layers), so every KDA layer broke the enclosing compiled region, leaving fragments whose guard and re-entry cost exceeded the fusion benefit.

Fixed in the F3/KDA commit by taking the route XTuner's own GatedDeltaNet already takes (and the reason Qwen3.5 can run `GatedDeltaNet.forward` at `fullgraph=True`): `xtuner/v1/ops/kda` calls FLA's `chunk_kda_fwd`/`chunk_kda_bwd` behind `torch.library.custom_op` + `register_fake`, so dynamo traces through them opaquely and `prepare_chunk_indices` stays hidden without a break.

| text model | steady per-step | tgs@20 | mem@20 | step20 loss |
|---|---:|---:|---:|---:|
| eager | 1.34s | 12203.1 | 86.22 | 10.31408882 |
| compile, before | 2.6–4.3s (never converges) | 5168.4 | 82.15 | 10.31412411 |
| **compile, after** | **1.21s** | **13381.1** | 82.11 | 10.31426430 |

Verified bitwise-identical to `fla.ops.kda.chunk_kda` — forward and all five gradients at `max|diff| = 0`. That check earned its keep: the first attempt fed the *original* q/k to `l2norm_bwd` where FLA feeds the *L2-normed* ones, corrupting only the q/k gradients (~1.0–1.5 max diff) while the forward stayed bitwise correct.

The 4353 → 4090 VL comparison was on the pre-fix corpus, which never trained a real video. The VL table above replaces it: with videos in the pack, compile leads eager (tgs 5727 vs 4465) and uses less memory (86.3 vs 90.6 GB).

`fused_kda_gate` and the short convolution are ported the same way (`kda_gate_fwd`/`bwd`, `causal_conv1d_fwd`/`bwd` behind `custom_op`), bitwise with FLA on the forward and every gradient. Breaks fell from 30 to 10 and throughput did not move (tgs 13376 vs 13381, steady 1.21s/step). The regression was the `chunk_kda` break; break count is the wrong metric, location is the right one. The KPool `num_pools` site stays unported: it is a real data-dependent shape, and pushing `ctx.new_dynamic_size()` into the indexer is likely a net loss.

### VL unit coverage

`TestGlm53ComposeSequenceParallel::test_image_splice_under_sp_matches_non_sp` (2 GPU): two placeholder spans placed so one lands on each rank's shard — forcing the cross-rank feature slicing — and each rank's logits match the corresponding slice of the non-SP run. `tests/model/test_glm53_compose.py` is 9 passed on 2 GPUs, including `test_every_pack_calls_the_vision_tower_exactly_once` (text, image, image+video) and the mixed-pack bitwise splice check.



```

### Changed Files

```
doc/progress.md (+460 -4)
examples/v1/config/sft_glm53.py (+141 -0)
examples/v1/config/sft_glm53_vl.py (+156 -0)
sft_glm53_tiny.sh (+80 -0)
sft_glm53_vl_tiny.sh (+87 -0)
tests/model/test_glm53_compose.py (+280 -0)
tests/model/test_glm53_text_moe.py (+400 -0)
xtuner/v1/data_proto/templates/__init__.py (+1 -0)
xtuner/v1/datasets/sft_tokenize_fn/openai.py (+3 -1)
xtuner/v1/model/__init__.py (+9 -0)
xtuner/v1/model/compose/glm53/__init__.py (+4 -1)
xtuner/v1/model/compose/glm53/glm53_config.py (+94 -0)
xtuner/v1/model/compose/glm53/modeling_glm53.py (+210 -0)
xtuner/v1/model/moe/glm53/__init__.py (+3 -0)
xtuner/v1/model/moe/glm53/glm53.py (+453 -0)
xtuner/v1/module/attention/kda.py (+94 -26)
xtuner/v1/ops/kda/__init__.py (+48 -0)
xtuner/v1/ops/kda/causal_conv1d.py (+157 -0)
xtuner/v1/ops/kda/chunk_kda.py (+296 -0)
xtuner/v1/ops/kda/fused_kda_gate.py (+131 -0)
xtuner/v1/ops/sparse_mla/kpool.py (+11 -6)
```

---

# GLM-5.3-Flash Stack PR Metadata (updated 2026-10-08T10:12Z — precision-test merge round)

GPT's precision tests were merged into the stack, lower branches were rebased upward,
and all branches were force-pushed to `InternLM/xtuner`. The per-PR commit lists in the
sections above are superseded by the tips below; PR titles, bases, and messages are
unchanged.

| # | PR | Head → Base | New tip | Commit added this round |
|---|----|-------------|---------|-------------------------|
| 1 | [#2105](https://github.com/InternLM/xtuner/pull/2105) | `feat/glm53flash-materialize-full-f0` → `main` | `91bebe283` (unchanged) | — |
| 2 | [#2106](https://github.com/InternLM/xtuner/pull/2106) | `feat/glm53flash-f3-kda` → `feat/glm53flash-materialize-full-f0` | `dbb125697` | `[Test] Add KDA chunked backward gradient parity against HF` |
| 3 | [#2107](https://github.com/InternLM/xtuner/pull/2107) | `feat/glm53flash-f4-mhc` → `feat/glm53flash-f3-kda` | `34eafef7a` | `[Fix] Pass the text RMS epsilon into mHC hc_pre` (4 call sites; inherits #2106's test) |
| 4 | [#2108](https://github.com/InternLM/xtuner/pull/2108) | `feat/glm53flash-f5-nope-dsa` → `feat/glm53flash-f4-mhc` | `9b26573cc` | — (rebase only) |
| 5 | [#2109](https://github.com/InternLM/xtuner/pull/2109) | `feat/glm53flash-f1-vl-data` → `feat/glm53flash-f5-nope-dsa` | `ca50f0e6b` | — (rebase only) |
| 6 | [#2110](https://github.com/InternLM/xtuner/pull/2110) | `feat/glm53flash-f2-vision-tower` → `feat/glm53flash-f1-vl-data` | `dec0da87d` | — (rebase only) |
| 7 | [#2111](https://github.com/InternLM/xtuner/pull/2111) | `feat/glm53flash-f6-text-moe` → `feat/glm53flash-f2-vision-tower` | `c471e6587` | `[Test] Add crop-gradient, AdamW-step, and multimodal FSDP-backward tests` |

Notes for reviewers:

- The #2107 fix corrects all four `hc_pre` call sites in
  `xtuner/v1/model/moe/glm53/decoder_layer.py` to pass the text-config RMS epsilon
  (`input/post_attention_layernorm.variance_epsilon`) instead of the `hc_eps` default of
  1e-6; checkpoint gradients now match HF within 6% (validated by #2111's
  `test_full_crop_fsdp_gradients_match_hf`).
- Test status at merge time: #2106 KDA suite 6/6; #2107 mHC/decoder-layer/hc_post 20/21
  (the one failure, `test_dense_layer_forward_compiles_with_dynamic_cu_seqlens`, is
  pre-existing — it fails identically on the pre-fix parent commit); #2111 new tests
  3/3 plus full compose suite 13/13.
- Backup of the pre-merge stack: `feat/glm53flash-f6-text-moe-bak1008` @ `016500f047`
  on `InternLM/xtuner`.

---

# GLM-5.3-Flash Stack PR Metadata (updated 2026-10-08T12:19Z — external commit on #2110)

A collaborator pushed one commit directly onto `feat/glm53flash-f2-vision-tower` after the
precision-test merge round. All other PR tips are unchanged from the previous section.

| # | PR | Remote tip | Change |
|---|----|-----------|--------|
| 1–5, 7 | #2105–#2109, #2111 | unchanged (`91bebe283`, `dbb125697`, `34eafef7a`, `9b26573cc`, `ca50f0e6b`, `c471e6587`) | — |
| 6 | [#2110](https://github.com/InternLM/xtuner/pull/2110) | **`df2efc8854`** (was `dec0da87d`) | +1 commit: `df2efc8854` **[Fix] Preserve GLM-5.3 vision RoPE frequencies during meta initialization** by **xs1997zju** (SuperRobin, 2026-10-08 19:14 +0800), pushed directly to `InternLM/xtuner` |

✅ **Inconsistency resolved (2026-10-08T12:26Z):** `feat/glm53flash-f6-text-moe` was rebased
onto `df2efc8854` (11 commits replayed cleanly; new tip `8d7e3d2e2`) and force-pushed to
`InternLM/xtuner`. `tests/model/test_glm53_vision.py` + `test_glm53_compose.py` pass 25/25
on the rebased branch.

## Non-jayhenry authors in the stack (as of this update)

All commits outside #2110 are authored/co-authored by jayhenry only (with the `claude`
bot as co-author on most). Non-jayhenry human/bot commits, all in **#2110 (F2 vision tower)**:

| Commit | Author(s) | Stack | Title | Since |
|--------|-----------|-------|-------|-------|
| `c469842f8` | **ShilohYu** (+ cursoragent bot) | **#2110 F2 vision tower** (stack 6/7) | fix(glm53): match vision RMSNorm rounding and reject unfinished SP | pre-existing (already in #2110's original 5 commits before 2026-09-27) |
| `df2efc8854` | **xs1997zju** (SuperRobin) | **#2110 F2 vision tower** (stack 6/7) | [Fix] Preserve GLM-5.3 vision RoPE frequencies during meta initialization | new on 2026-10-08 |

---

# GLM-5.3-Flash Stack PR Metadata (updated 2026-10-09T06:36Z — rebased onto upstream/main + 1 new commit)

The whole stack was rebased onto the latest `upstream/main` (tip `c090fe5b`, "Ci/unit test
dispatch only (#2137)"), rewriting every commit SHA, and #2111 gained one new commit. All
SHAs in earlier sections are superseded by this table. Local branches are synced to the
remote tips below.

| # | PR | Head → Base | New tip (rebased) | Previous tip | Note |
|---|----|-------------|-------------------|--------------|------|
| 1 | [#2105](https://github.com/InternLM/xtuner/pull/2105) | `feat/glm53flash-materialize-full-f0` → `main` | `9ec259aec` | `91bebe283` | rebase only |
| 2 | [#2106](https://github.com/InternLM/xtuner/pull/2106) | `feat/glm53flash-f3-kda` → `feat/glm53flash-materialize-full-f0` | `9bc5db395` | `dbb125697` | rebase only |
| 3 | [#2107](https://github.com/InternLM/xtuner/pull/2107) | `feat/glm53flash-f4-mhc` → `feat/glm53flash-f3-kda` | `68414488d` | `34eafef7a` | rebase only |
| 4 | [#2108](https://github.com/InternLM/xtuner/pull/2108) | `feat/glm53flash-f5-nope-dsa` → `feat/glm53flash-f4-mhc` | `c2091a247` | `9b26573cc` | rebase only |
| 5 | [#2109](https://github.com/InternLM/xtuner/pull/2109) | `feat/glm53flash-f1-vl-data` → `feat/glm53flash-f5-nope-dsa` | `2aaecb45d` | `ca50f0e6b` | rebase only |
| 6 | [#2110](https://github.com/InternLM/xtuner/pull/2110) | `feat/glm53flash-f2-vision-tower` → `feat/glm53flash-f1-vl-data` | `72ce9ed46` | `df2efc8854` | rebase only |
| 7 | [#2111](https://github.com/InternLM/xtuner/pull/2111) | `feat/glm53flash-f6-text-moe` → `feat/glm53flash-f2-vision-tower` | `a537fcb44` | `8d7e3d2e2` | rebase + **1 new commit** (11 → 12) |

New commit in #2111:

- `a537fcb44` **[Test] Extend GLM HF parity checks with logits and image cases (#2141)** by
  **xs1997zju** (SuperRobin, 2026-10-09 14:21 +0800). Title carries the #2141 reference.

Non-jayhenry commits (SHAs updated by the rebase; stacks unchanged — both still #2110):

| Commit (new) | Commit (old) | Author(s) | Stack |
|--------------|--------------|-----------|-------|
| `a435755a6` | `c469842f8` | ShilohYu (+ cursoragent bot) | #2110 F2 vision tower (stack 6/7) |
| `72ce9ed46` | `df2efc8854` | xs1997zju (SuperRobin) | #2110 F2 vision tower (stack 6/7) |
| `a537fcb44` | — (new) | xs1997zju (SuperRobin) | #2111 F6 text+compose (stack 7/7) |

All jayhenry-authored commit titles are unchanged — only SHAs moved. Backup branch
`feat/glm53flash-f6-text-moe-bak1008` @ `016500f047` is untouched.

---

# GLM-5.3-Flash Stack PR Metadata (updated 2026-10-13 — #2153 merge and integration rerun)

The seven PR heads below are synchronized with `InternLM/xtuner`. The author's updated
[#2153](https://github.com/InternLM/xtuner/pull/2153) was merged at 2026-10-13 05:07 UTC and
replayed onto F6 as `ccc092e5`. The seven PRs remain open. Local validation completed six Text
and six VL 20-step runs. The two 8-GPU FLA sequence parallel unit failures were resolved
locally after those runs; the test changes are not yet in the PR heads below.

## Current stack (bottom to top)

| Layer | PR | Head → Base | Remote tip | State |
|---|---|---|---|---|
| F0 | [#2105](https://github.com/InternLM/xtuner/pull/2105) | `feat/glm53flash-materialize-full-f0` → `main` | `f8eaa324` | OPEN |
| F3 | [#2106](https://github.com/InternLM/xtuner/pull/2106) | `feat/glm53flash-f3-kda` → `feat/glm53flash-materialize-full-f0` | `64ab6647` | OPEN |
| F4 | [#2107](https://github.com/InternLM/xtuner/pull/2107) | `feat/glm53flash-f4-mhc` → `feat/glm53flash-f3-kda` | `eba4cba1` | OPEN |
| F5 | [#2108](https://github.com/InternLM/xtuner/pull/2108) | `feat/glm53flash-f5-nope-dsa` → `feat/glm53flash-f4-mhc` | `0cc95035` | OPEN |
| F1 | [#2109](https://github.com/InternLM/xtuner/pull/2109) | `feat/glm53flash-f1-vl-data` → `feat/glm53flash-f5-nope-dsa` | `e933fdcf` | OPEN |
| F2 | [#2110](https://github.com/InternLM/xtuner/pull/2110) | `feat/glm53flash-f2-vision-tower` → `feat/glm53flash-f1-vl-data` | `ba3dc696` | OPEN |
| F6 | [#2111](https://github.com/InternLM/xtuner/pull/2111) | `feat/glm53flash-f6-text-moe` → `feat/glm53flash-f2-vision-tower` | `ccc092e5` | OPEN |

Each preceding remote tip is an ancestor of the next. #2153's GitHub merge commit is
`31d05390`; its patch is at the current F6 tip after the stack rebase.

## #2153 unit tests

Ran the seven test files changed by #2153 under `pt29_glm2`, `zdev/env.sh`, and the shared GPU
lock on eight H200 GPUs: initially **156 passed, 2 failed**. Both failures were in
`tests/model/test_glm53_kda.py::TestKDAFLASequenceParallel8`:

- `test_packed_forward_and_backward_match_non_sp`
- `test_checkpointed_forward_and_backward`

Minimal 8-rank reproductions isolated two test issues:

- For SP versus non-SP, the gate forward is identical and per-token gate-gradient relative
  error is 0.4–1.0%. The BF16 `A_log` gradient sums signed token contributions; its net
  norm is only 6.3% of their absolute-contribution norm in the failing width-1 case.
  Thus a 0.26% contribution-scale error appears as a 4.07% net-gradient error. The
  test now gives only the BF16 width-1 `A_log` case a 5% relative budget; all other
  checks retain their existing budgets.
- For checkpointing, eager versus eager is identical, as is compiled versus compiled
  on all eight ranks. The old test compared a compiled checkpoint against an eager
  baseline, mixing compiler BF16 rounding into a checkpoint replay check. The
  baseline now uses the same compile mode.

Both original 8-GPU tests were rerun under the GPU lock and **passed (2 passed,
40.54 s)**. The other 156 cases were not rerun after these test-only changes. The
integration launchers use the default Ulysses KDA SP path, so the runs below do not
validate FLA SP end-to-end training.

## Text 20-step integration (`sft_glm53_tiny.sh`)

One node, eight H200 GPUs, the real F0 25B crop, `PACK_MAX_LENGTH=16384`,
`TOTAL_STEP=20`, and `MODEL_COMPILE=0` except in the compile row. All six profiles
completed with finite loss lower at step 20 than at step 1; no NaN or OOM. Rank 0 values
are shown. Memory is peak allocated / reserved GB; `train_s` is the log's
`Training finished in` value and includes the first-step warmup. The loss is not
strictly monotonic at every intermediate step.

| Profile | Step 1 loss | Step 20 loss | Grad norm@20 | Memory@20 GB | tgs@20 | seqlen_tgs@20 | exp_tgs@20 | train_s |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| default (`EP4 SP1 offload=1`) | 11.47560 | 10.56820 | 44.50 | 87.10 / 104.85 | 10686.2 | 10704.5 | 5045.0 | 64.88 |
| `SP2 EP4` | 11.48757 | 10.57616 | 44.59 | 75.57 / 96.24 | 8395.4 | 8395.4 | 3363.2 | 48.77 |
| `EP8 SP1` | 11.47560 | 10.56813 | 44.50 | 101.24 / 113.54 | 10618.4 | 10636.6 | 5867.7 | 55.81 |
| `offload=0` (`EP4 SP1`) | 11.47560 | 10.56826 | 44.51 | 86.53 / 106.45 | 10813.3 | 10831.8 | 6220.5 | 52.62 |
| `FP8=1` | 11.48150 | 10.45780 | 40.80 | 85.69 / 103.26 | 10035.9 | 10053.1 | 5321.3 | 61.51 |
| `MODEL_COMPILE=1` | 11.47577 | 10.56818 | 44.50 | 83.08 / 104.26 | 11759.1 | 11779.2 | 2878.3 | 113.67 |

Logs: `work_dirs/sft_glm53/stack2153_20261013/<profile>/training_log_*_0.txt`.

## VL 20-step integration (`sft_glm53_vl_tiny.sh`)

The same node, GPUs and checkpoint, with the `ci_vl` image/video corpus,
`PACK_MAX_LENGTH=SAMPLE_MAX_LENGTH=16384`, and `TOTAL_STEP=20`. All six profiles
completed with finite loss lower at step 20 than at step 1; no NaN or OOM. The step-20
visual-token counts match the historical video-inclusive runs, including the smaller
`SP2` batch. The other metric definitions match the Text table.

| Profile | Step 1 loss | Step 20 loss | Grad norm@20 | Memory@20 GB | tgs@20 | seqlen_tgs@20 | exp_tgs@20 | train_s | img_tokens@20 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| default (`EP4 SP1 offload=1`) | 12.47497 | 10.74597 | 36.94 | 90.01 / 108.64 | 3882.5 | 4421.4 | 3179.0 | 95.63 | 56424 |
| `SP2 EP4` | 12.26586 | 11.38063 | 30.97 | 77.04 / 97.14 | 3272.3 | 3272.3 | 2358.0 | 69.54 | 31884 |
| `EP8 SP1` | 12.47497 | 10.78517 | 36.69 | 103.72 / 117.04 | 3865.1 | 4401.6 | 3059.4 | 99.39 | 56424 |
| `offload=0` (`EP4 SP1`) | 12.47497 | 10.77860 | 36.59 | 89.53 / 108.13 | 3885.0 | 4424.2 | 3129.4 | 97.15 | 56424 |
| `FP8=1` | 12.46999 | 10.35798 | 27.60 | 88.54 / 107.44 | 3749.5 | 4269.9 | 3030.5 | 100.31 | 56424 |
| `MODEL_COMPILE=1` | 12.47076 | 10.77807 | 36.93 | 85.46 / 107.25 | 4788.9 | 5453.6 | 2740.3 | 110.93 | 56424 |

Logs: `work_dirs/sft_glm53_vl/stack2153_20261013/<profile>/training_log_*_0.txt`.

## Comparison with the historical Text and VL tables above

`tgs` is rank 0 step-20 tokens/GPU/s. `train_s` uses each original log's actual
`Training finished in` value. The historical tables' `time` entries, especially the VL
entries (for example default 67s versus 110.93s in its log), do not match that measure;
the old `time` cells are therefore not used for this comparison. `train_s` includes
first-step warmup and is sensitive to compilation/cache state; `tgs@20` better describes
the observed steady-state rate.

| Modality | Profile | Old → current tgs@20 | Change | Old → current step-20 loss | Old → current train_s |
|---|---|---:|---:|---:|---:|
| Text | default | 12203.1 → 10686.2 | -12.4% | 10.31409 → 10.56820 | 77.69 → 64.88 |
| Text | `SP2 EP4` | 9631.7 → 8395.4 | -12.8% | 10.32397 → 10.57616 | 62.65 → 48.77 |
| Text | `EP8 SP1` | 11865.9 → 10618.4 | -10.5% | 10.31380 → 10.56813 | 69.76 → 55.81 |
| Text | `offload=0` | 12315.3 → 10813.3 | -12.2% | 10.31401 → 10.56826 | 66.19 → 52.62 |
| Text | `FP8=1` | 6159.6 → 10035.9 | +62.9% | 10.18042 → 10.45780 | 174.88 → 61.51 |
| Text | `MODEL_COMPILE=1` | 13381.1 → 11759.1 | -12.1% | 10.31426 → 10.56818 | 94.74 → 113.67 |
| VL | default | 4465.1 → 3882.5 | -13.0% | 10.78624 → 10.74597 | 110.93 → 95.63 |
| VL | `SP2 EP4` | 3674.0 → 3272.3 | -10.9% | 11.29156 → 11.38063 | 85.12 → 69.54 |
| VL | `EP8 SP1` | 4468.9 → 3865.1 | -13.5% | 10.78038 → 10.78517 | 118.61 → 99.39 |
| VL | `offload=0` | 4488.9 → 3885.0 | -13.5% | 10.81396 → 10.77860 | 112.53 → 97.15 |
| VL | `FP8=1` | 3972.2 → 3749.5 | -5.6% | 10.27666 → 10.35798 | 105.32 → 100.31 |
| VL | `MODEL_COMPILE=1` | 5727.1 → 4788.9 | -16.4% | 10.78443 → 10.77807 | 128.48 → 110.93 |

## Same-environment control before #2153

Ran the same 20-step default Text and VL profiles on the immediate pre-#2153 F6 parent
`9cacd38c`, in an isolated worktree with the same `zdev/env.sh`, checkpoint, dataset,
launcher options, GPU lock and eight H200 GPUs. The current F6 head is `ccc092e5`.

| Modality | Pre-#2153 step-1 loss | Pre-#2153 → current step-20 loss | Pre-#2153 → current tgs@20 | Tgs change |
|---|---:|---:|---:|---:|
| Text default | 11.47560 | 10.56852 → 10.56820 | 10746.7 → 10686.2 | -0.6% |
| VL default | 12.47497 | 10.75368 → 10.74597 | 3893.4 → 3882.5 | -0.3% |

The Text and VL default first-step losses are identical before and after #2153, and
their step-20 throughput is within 0.6%. Thus the large default throughput difference
from the historical tables, and the Text loss shift, were already present before #2153
in the current environment. This control does not isolate the other five profiles.
Control logs: `work_dirs/sft_glm53/stack2153_pre_pr_20261013/default/` and
`work_dirs/sft_glm53_vl/stack2153_pre_pr_20261013/default/`.

The 16–20 step mean `tgs` also drops for every Text BF16 and VL profile, so the
step-20 throughput drop is not a single-step outlier. Text FP8 improves against the
older FP8 row. The Text loss shifts upward by about 0.25 at step 20; VL loss changes
in both directions. These short-run loss shifts do not by themselves establish a
model-quality regression. The two 8-GPU FLA sequence parallel unit failures are resolved
by the test corrections documented above; FLA SP end-to-end training was not rerun.
