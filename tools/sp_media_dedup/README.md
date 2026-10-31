# SP media duplication: CPU diagnosis and sharing prototype (draft)

This is an isolated snapshot of ongoing work, not a trainer optimization ready for use.
Production dataset/trainer code is unchanged. No GPU training benefit is claimed.
The native data-path diagnosis has run; the D2 sharing prototype has not passed runtime validation.

## Observed duplication

Frozen XTuner source: `2b79471c3d7948e9c69d58eb62a85cfbf258e923`.
A CPU/Gloo mesh with DP=1 and SP=1 or SP=8 drives the native
`DataloaderConfig.build` path, processor and collator, without model weights or CUDA.
The real-data window is one document with two 5953 x 8419 JPEG images;
pack maximum length is 524288, and the first comparison uses num_workers=0.
This is a small representative diagnosis, not full training or an OOM reproduction.

| Run | Media reads | Processor calls | Idle tree PSS (GiB) | Peak tree PSS (GiB) | Increment (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| SP1 repeat 1 | 2 | 1 | 1.410 | 2.524 | 1.114 |
| SP1 repeat 2 | 2 | 1 | 1.407 | 2.241 | 0.834 |
| SP8 repeat 1 | 16 | 8 | 6.630 | 16.012 | 9.382 |

The eight ranks delivered identical pixel and input-ID hashes. Native fake-data
fallback count was zero. The memory increment includes temporary decoding and
processing allocations; it is NOT the predicted memory savings of sharing.
SP8 repeat and actual worker-path follow-ups are not included in this result table.
Synthetic fixtures also passed SP1/SP8 x num_workers {0,2} diagnostics.

## Files and execution

- `sp_cpu_diag.py`: native data-path instrumentation and experimental D2 hooks.
- `sp_share.py`: work-in-progress file-backed shared-memory protocol.
- `monitor_tree.py`: CPU diagnostic process-tree monitor.
- `analyze_events.py`: event aggregation.
- `make_fixture.py`: generated synthetic images and annotation fixture.

Use an isolated environment with the model tokenizer/processor dependencies.
Run the fixture generator with an output directory. The harness accepts `--config`
and is launched through `torchrun --standalone --nproc-per-node=1` (or 8).
Provide local model/annotation/media/cache paths and optional existing OSS client
configuration through the JSON configuration; no credentials or real media are included.
Required configuration fields are read explicitly in `main()`. Use a unique events_dir,
DP1/SP1 or DP1/SP8, and `d2_mode: "off"` for the validated baseline path.
Set resource limits and serialize experiments against other memory benchmarks.
Do not infer available memory from GPU utilization alone.

## D2 status and known gaps

Sharing is at collated pack level to avoid a subsequent sample-level cat recreating
private copies. However, the captured loop still calls `next(loader)` on every rank
before attaching a shared view. It does NOT yet eliminate repeated preprocessing.
Trainer integration, H2D/pinning lifetime and training performance are not validated.

The storage prototype is unfinished and must not be enabled for training:
- header magic/field layout and lease integer encoding need runtime tests/fixes;
- dtype conversion and SequenceContext serialization/reconstruction are incomplete;
- producer ownership release, exported views, failure cleanup and capacity races
  need validation; read-only mappings do not make arbitrary PyTorch writes safe;
- native Tensor references must actually be dropped before claiming deduplication;
- group/config/content identity and collective failure handling need full coverage.

Before promotion: validate publication/attach/release and bounded failure paths,
implement consumer skip before data materialization, repeat same-input CPU comparisons,
then validate complete trainer inputs and GPU training with fair memory/TGS measurements.

## Validation of this snapshot

Python syntax compilation only for the captured WIP files. Native CPU diagnostic
results above come from completed baseline runs, not from the unvalidated D2 code.
No raw media, annotations, credentials, machine-specific configuration or full logs
are published. Results are preliminary aggregate measurements.
