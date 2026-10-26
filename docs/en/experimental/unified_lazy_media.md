# Shared lazy image processing (draft)

GLM-5.3-Flash, Qwen3-VL and Qwen3.5 share lazy image materialization, SP-group
leader decode and one-batch CPU lookahead. Features are opt-in and off by default:

```bash
export XTUNER_LAZY_MEDIA=1
export XTUNER_LAZY_MEDIA_SHARED=1
export XTUNER_LAZY_MEDIA_PREFETCH=1
export XTUNER_LAZY_MEDIA_THREADS=2
```

`XTUNER_LAZY_MEDIA` emits image descriptors instead of eager pixels when annotated
geometry is available. Invalid or missing geometry uses the existing eager path.
Cache token counting and native text/label/position construction are unchanged.
Video retains its native eager path. Mixed eager/lazy image records are resolved
in native order before collation.

`XTUNER_LAZY_MEDIA_SHARED` selects one decoder per SP group. Plan agreement
(including text-only batches), decoder status and allocation status are checked
before payload broadcasts. Without it, each rank processes its own plan.
Pixels keep native CPU placement for visual processing; grids follow the input
device. This is important for models that slice CPU pixels before transferring
their SP shard.

`XTUNER_LAZY_MEDIA_PREFETCH` fetches and materializes at most one future batch on a
single CPU thread. CUDA and distributed calls remain on the training thread.
Image processing uses up to `XTUNER_LAZY_MEDIA_THREADS` workers (default 1, range
0–64; 0 also means serial). Decoded images are closed on success and failure.
Outstanding native I/O is joined during shutdown; it is not forcibly preemptible.

## Common extension entry

`xtuner.v1.datasets.lazy_media` contains the shared implementation:

- `make_image_plan`: serialize the native processor class/config, ordered paths
  and predicted grids. It does not require a separate metadata manifest or tool.
- `pack_image_plans` / `merge_image_plans`: keep retained pack order and reject
  inconsistent processor/dtype/source settings.
- `materialize_image_plan`: reconstruct worker-local processors and process images
  independently, validating actual grids and patch shape.
- `resolve_lazy_media`: join lookahead or materialize, perform optional SP delivery
  and return a resolved context copy.

The GLM and Qwen models each explicitly call `resolve_lazy_media` in their existing
visual-input preparation method. The public compose base and training engine have
no media hooks or media scheduler. Qwen3.5 uses the Qwen compose path.

A new model supplies an importable image-processor adapter with `to_dict`,
`from_dict`, `patch_size`, `temporal_patch_size`, and the standard CPU
`images -> pixel_values/image_grid_thw` call. Its tokenizer predicts grids using
native rules and calls `make_image_plan`; its collator uses `pack_image_plans`;
its visual-input preparation calls `resolve_lazy_media`. Batch-dependent image
processing requires a suitable adapter and parity tests before opting in.
Token expansion, MRoPE and visual feature insertion stay family-specific.

The deferred consumer keeps processed pixels out of the original plan-only batch.
It does not detach vision activations required by autograd. Native eager/offload
ownership and arbitrary external tensor aliases are outside this helper.

## Checkpoint resume

Async fetching does not count as training consumption. The trainer commits sample
counts after a successful optimizer step and before checkpoint saving. Lookahead
stays inside the current epoch; only the consumer advances the epoch. At an exact
epoch boundary, the checkpoint stores the exhausted sampler offset instead of
modulo zero. Parallel, LengthGrouped and Preset samplers are covered by CPU tests.
The async-disabled path retains its existing behavior.

This does not repair older checkpoints whose cursor was already advanced, or
replay arbitrary stochastic worker transforms and their RNG state.

## Review scope

This draft has CPU processor/tokenizer, pack-order, mocked SP failure and
checkpoint-resume tests. GPU/NCCL and full-model validation are not claimed.
Qwen tokenizer parity tests exercise native text/grid processing with 3D RoPE
disabled; family-specific RoPE compatibility is separate. Real OSS service access
and optimized video are not validated here. Existing per-family native errors and
unsupported-input policies remain relevant.
