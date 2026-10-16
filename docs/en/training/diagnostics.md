# Opt-in training diagnostics

When a distributed job stops progressing, a collective timeout often identifies a
waiting rank rather than the operation that first stalled. Training diagnostics
record CPU progress boundaries and, optionally, module and communication stream
boundaries. They complement PyTorch NCCL Flight Recorder; they are not a kernel
profiler and do not change checkpoint or learning-rate scheduling.

```python
from xtuner.v1.train import DiagnosticsConfig, Trainer, TrainerConfig

trainer_cfg = TrainerConfig(
    # ... existing training settings ...
    diagnostics=DiagnosticsConfig(
        enabled=True,
        level="detailed",
        start_step=105,
        end_step=112,
        stack_dump_interval_s=300,
        cuda_events=True,
    ),
)
```

The SFT CLI manages the recording lifecycle around `trainer.fit()`. Direct Python
callers opt in explicitly using the same context:

```python
trainer = Trainer.from_config(trainer_cfg)
with trainer_cfg.diagnostics.record(trainer):
    trainer.fit()
```

Calling `trainer.fit()` alone does not activate diagnostics. Training-stage events
are dispatched through `ProberList.record_progress`; enabling and step/rank
filtering stay inside the diagnostics component. Existing tensor probers keep
their own setup and schedule. No new training method or wrapper around `fit` is
installed, and its loop is not reindented for diagnostics.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `enabled` | `False` | Master switch. Disabled mode installs no wrappers, tensor hooks, events, timers, or polling thread. |
| `level` | `"basic"` | `basic`: training stage markers; `detailed`: additionally selected module, gradient, DeepEP dispatcher, and offload boundaries. |
| `start_step` | `1` | First recorded global optimizer step, inclusive. |
| `end_step` | `None` | Last recorded step, inclusive; `None` records until training ends. |
| `ranks` | `None` | Selected global ranks; `None` means all ranks. This does not alter collective participation. |
| `output_dir` | `None` | Defaults to `<experiment directory>/diagnostics/`. |
| `stack_dump_interval_s` | `0` | Repeated Python stack dump interval; zero disables it. The timer resets at each selected step. |
| `cuda_events` | `False` | Detailed mode only: query CUDA events without adding stream waits or synchronization. |
| `max_pending_cuda_events` | `4096` | Bound outstanding events per rank; retain older events and report saturation. |

Basic mode records data loading, input preparation, microbatch forward/backward,
postprocessing, offload cleanup, gradient reduction/clipping, and optimizer
boundaries. CPU return markers do not prove GPU completion.

Detailed mode wraps only the explicit prototype module list:
`Glm53DenseDecoderLayer`, `Glm53MoEDecoderLayer`, `KimiDeltaAttention`,
`NoPEDSAMultiLatentAttention`, `MoEBlock`, and `KPoolIndexer`, plus DeepEP
dispatcher methods and the existing offload methods. It does not scan loaded
modules, replace CP helpers or custom-autograd functions, or patch DeepEP Buffer.
Tensor gradient-ready boundaries are recorded without modifying gradients.
Compiler-owned function attributes are not copied into wrappers:
copying `_torchdynamo_orig_callable` from a bound compiled method can cause
`torch.compiler.disable` to unwrap it into an unbound method and lose `self`.

Detailed mode intentionally creates compiler boundaries and can change timing and
compilation. Wrappers remain installed outside the recording window, but no
per-call records or tensor hooks are created there. Use basic mode first when
minimizing perturbation is important. Module families and loaded operations
outside the supported paths may not receive detailed records.

Records are line-buffered JSONL files named `rank<N>.jsonl`, containing wall and
monotonic timestamps, rank, step, microbatch, phase, scope and call IDs where
applicable. Checkpoint replay is labelled `recompute`; some compiled backward
operations may remain `unattributed`. Gradient-ready events do not prove that all
parameter gradients have completed. Python stack dumps go to stderr; retain
per-rank stderr with torchrun's `--log-dir`, `--redirects`, and `--tee` options.
The stack timer uses Python's process-wide faulthandler timer and must not be
combined with another owner of that timer.

When enabled, stream events record the caller's current stream, explicit offload
copy streams. DeepEP internal communication streams are not recorded. A
completed event proves progress only on that stream and its existing dependencies.
The polling thread emits completion records and periodic pending-event heartbeats.
`pending_at_close` means events were not observed complete before shutdown, not
necessarily that training hung. Shutdown does not synchronize the GPU. The recorder
supports one active training diagnostics instance per process; detailed wrappers
are restored on normal completion and exceptions. Forced process termination may
prevent cleanup and the final log flush.

NCCL Flight Recorder remains configured through PyTorch's own environment variables,
including `TORCH_NCCL_TRACE_BUFFER_SIZE`, `TORCH_NCCL_DUMP_ON_TIMEOUT`, and
`TORCH_NCCL_TRACE_CPP_STACK`. Consult documentation for the installed PyTorch
version for dump-path settings. It covers ProcessGroupNCCL operations, not all
DeepEP/NVSHMEM internals. Compare rank communication sequences with the stage and
stream records; a timeout alone is not proof that NCCL caused the original stall.

This feature does not change model saves, introduce an early stopping limit, or
select a resume checkpoint. Existing experiment configuration controls those.
