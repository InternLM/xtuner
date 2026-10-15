import os
import pickle
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist
from torch.autograd.profiler_util import FunctionEvent

from xtuner.v1.utils import get_logger


logger = get_logger()


MEMORY_SNAPSHOT_MAX_ENTRIES = 10000000
OOM_SNAPSHOT_MAX_ENTRIES = 100000


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def _should_profile_rank() -> bool:
    ranks = os.environ.get("PROFILE_RANKS", "all").strip().lower()
    if ranks == "all":
        return True
    return _rank() in {int(rank.strip()) for rank in ranks.split(",") if rank.strip()}


def record_memory_history_on_oom(snapshot_dir: Path) -> None:
    """Record recent allocations on every rank and save the failing rank at
    allocator OOM."""
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.memory._record_memory_history(max_entries=OOM_SNAPSHOT_MAX_ENTRIES, stacks="python")
    rank = _rank()
    snapshot_path = snapshot_dir / f"rank{rank}_memory_snapshot.pickle"

    def oom_observer(device: int, alloc: int, device_alloc: int, device_free: int) -> None:
        # Dump inside the allocator callback, before exception unwinding releases tensors.
        try:
            torch.cuda.memory._dump_snapshot(str(snapshot_path))
            logger.error(
                f"CUDA OOM on rank {rank}, device {device}: requested {alloc} bytes, "
                f"allocated {device_alloc} bytes, free {device_free} bytes; saved {snapshot_path}"
            )
        except Exception:
            logger.exception(f"Failed to save CUDA OOM snapshot to {snapshot_path}")

    torch._C._cuda_attach_out_of_memory_observer(oom_observer)  # type: ignore[attr-defined]


class TimeProfiler:
    def __init__(self, profile_dir: Path):
        self.profile_dir = profile_dir
        self.profiler = None

    def _format_time(self, time_us: float):
        """Define how to format time in FunctionEvent."""
        US_IN_SECOND = 1000.0 * 1000.0
        US_IN_MS = 1000.0
        if time_us >= US_IN_SECOND:
            return f"{time_us / US_IN_SECOND:.3f}s"
        if time_us >= US_IN_MS:
            return f"{time_us / US_IN_MS:.3f}ms"
        return f"{time_us:.3f}us"

    def get_total_time(self, events: list[FunctionEvent]):
        t = sum([x.self_device_time_total for x in events])
        return self._format_time(t)

    def __enter__(self):
        if dist.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        def trace_handler(prof):
            self.profile_dir.mkdir(parents=True, exist_ok=True)
            total_device_time = self.get_total_time(prof.key_averages())

            logger.info(f"Dumping profiler traces at step to {self.profile_dir}")
            begin = time.monotonic()
            prof.export_chrome_trace(f"{self.profile_dir}/rank{rank}_trace_time-{total_device_time}.json")
            logger.info(
                f"Finished dumping profiler traces in {time.monotonic() - begin:.2f} seconds, "
                f"total device time: {total_device_time}"
            )

        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready=trace_handler,
            record_shapes=True,
        )
        self.profiler.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self.profiler is not None, "Please make sure `__enter__` is called before `__exit__`"
        self.profiler.__exit__(exc_type, exc_val, exc_tb)


# Reference: https://github.com/pytorch/torchtitan/blob/main/torchtitan/tools/profiling.py


class MemoryProfiler:
    def __init__(self, profile_dir: Path, *, keep_oom_history: bool = False):
        torch.cuda.memory._record_memory_history(max_entries=MEMORY_SNAPSHOT_MAX_ENTRIES, stacks="python")
        self.profile_dir = profile_dir
        self.keep_oom_history = keep_oom_history

    def close(self):
        if self.keep_oom_history:
            # The sampled step needs a larger trace, then restore the bounded OOM window.
            torch.cuda.memory._record_memory_history(max_entries=OOM_SNAPSHOT_MAX_ENTRIES, stacks="python")
            return
        try:
            torch.cuda.memory._record_memory_history(enabled=None)
        except TypeError:
            torch.cuda.memory._record_memory_history(False)

    def step(self, exit_ctx: bool = False):
        if dist.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        if exit_ctx:
            output_dir = self.profile_dir.with_name(self.profile_dir.name + "_exit")
        else:
            output_dir = self.profile_dir

        output_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Dumping memory snapshot to {output_dir}")
        begin = time.monotonic()
        with open(output_dir / f"rank{rank}_memory_snapshot.pickle", "wb") as output:
            pickle.dump(torch.cuda.memory._snapshot(), output)
        logger.info(f"Finished dumping memory snapshot in {time.monotonic() - begin:.2f} seconds")


@contextmanager
def profiling_time(profile_dir: Path):
    if _should_profile_rank():
        with TimeProfiler(profile_dir=profile_dir):
            yield
    else:
        yield


@contextmanager
def profiling_memory(profile_dir: Path, *, keep_oom_history: bool = False):
    if not _should_profile_rank():
        yield
        return
    profiler = MemoryProfiler(profile_dir, keep_oom_history=keep_oom_history)
    try:
        try:
            yield
        except torch.OutOfMemoryError:
            profiler.step(exit_ctx=True)
            raise
        else:
            profiler.step(exit_ctx=False)
    finally:
        # close() flushes and releases profiler resources on all exit paths.
        # Without it, consecutive memory profiling runs may overwrite each other's output.
        profiler.close()
