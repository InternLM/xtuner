# Experimental GLM53 lazy image preprocessing

This opt-in prototype defers image decoding from workers to consumption, processes images with a bounded thread pool, optionally decodes once per sequence-parallel group and broadcasts pixels/grids, and optionally overlaps materialization with training. Videos retain the eager path.

Set `XTUNER_LAZY_MEDIA=1` to enable. `XTUNER_LAZY_MEDIA_THREADS=0` selects serial materialization; `8` enables the tested thread pool. Add `XTUNER_LAZY_MEDIA_SHARED=1` for SP decode, then `XTUNER_LAZY_MEDIA_PREFETCH=1` for asynchronous lookahead. Leave the feature flags unset for the normal eager path.

## Recorded experiment

8 GPUs, 512K, five-layer FP8 model, SP8/EP8/DP1, GBS1, same frozen 20-step window. Base: 2b79471; final tested source: b890abc. These are diagnostic results, not full-layer BF16 or 256-GPU validation.

| Variant | Peak process-tree PSS GiB | Median step seconds (4–20) | Median rank0 TGS |
|---|---:|---:|---:|
| Eager | 513.14 | 44.64 | 1468.1 |
| Serial lazy | 327.79 | 85.56 | 766.0 |
| + 8 threads | 252.41 | 58.53 | 1119.7 |
| + SP decode | 125.80 | 51.66 | 1268.7 |
| + Async | 146.65 | 42.30 | 1549.3 |

Final observed peak reduction: 71.4%; TGS increase: 5.5%. Each added row is conditional on the previous optimizations, not an independent ablation. The eager trace has fewer samples (302 versus 1306 for final), so brief peaks may be missed. No repeated 20-step baseline or long-run convergence claim is made.

Sample-level input/pixel equality was recorded by the experiment. Final versus eager maximum absolute/relative LLM loss differences from printed step logs: 0.00532436 / 4.83884e-4. Maximum MTP absolute difference: 1.274e-5; grad-norm relative difference: 6.58505e-3. First two LLM losses match at log precision; this is not proof of bitwise training or identical parameter updates. Full-layer BF16 failed during model loading before data processing.

## Current integration and validation

Rebased onto main at 40b3b18 (#2207). The historical table above predates this integration and must not be interpreted as a measurement of the current code.

- Async decoding now runs on every SP-group leader (or every rank when shared decode is disabled). Lookahead is bounded to the current batch plus one future batch. The training loop retains its original direct iterator structure. Explicit cleanup on early consumer exit is not guaranteed by an outer context manager and remains a validation item; an active external media read may delay shutdown.
- Decode and allocation errors are propagated through the main-thread SP protocol. Process death and collective failures still rely on distributed timeouts; rank-plan agreement remains an assumption.
- The GLM collator's media_dtype travels with lazy plans. Processor outputs are cast before materialization concatenation, and SP buffers use the same dtype. Legacy raw plans without the field retain FP32. Videos retain the upstream eager collator path.
- OSS configuration and mixed eager-fallback/lazy image ordering are preserved.

Current CPU validation: 17 tests passed, including real-processor BF16/FP32 transport parity, OSS, mixed eager fallback, upstream image/video collator tests, and bounded prefetch/cleanup. A four-process CPU Gloo test passed for two SP groups using BF16 transport and injected leader decode errors.

The new six-run GPU/NCCL precision and performance comparison is pending. No new speedup, long-run convergence, full-model parameter-update equality, or 256-GPU validation is claimed for this rebased version. Checkpoint resume, epoch transitions, process failure, and general microbatch workloads need further validation. Keep this PR experimental.
