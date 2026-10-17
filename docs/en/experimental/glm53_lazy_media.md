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

## Outstanding review and validation

This is a draft experimental snapshot, not ready for production merge.

- Lookahead lacks cancellation/join on early consumer exit. Queue capacity one does not imply exactly one future batch: materialization starts before `put`, and the producer may hold another batch while blocked.
- Async shared decode selects global rank zero. Other SP-group leaders fall back to synchronous materialization at consumption; the current source comment describing an all-rank fallback is inaccurate. DP>1 is unvalidated.
- Epoch transitions/checkpoint resume and worker error teardown need tests.
- SP broadcasts assume equal plan shapes/order across ranks, without cross-rank fingerprint checks; fault propagation needs validation.
- Multiple processor configurations, mixed media, general microbatch counts, and parameter-update equivalence are not fully validated.
- The original async smoke overran the step limit (5/3); b890abc bounds iteration locally. Corrected smoke completed 3/3 and final completed 20/20.

The separate per-image eager optimization has independent measurements; it has not been combined with this full ladder in training. Do not add its memory reduction to these numbers.
