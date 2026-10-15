from contextlib import contextmanager
from pathlib import Path

from xtuner.v1.utils import get_device


if get_device() == "cuda":
    from .cuda_profile import profiling_memory, profiling_time
elif get_device() == "npu":
    from .npu_profile import profiling_memory, profiling_time

else:

    @contextmanager
    def profiling_time(profile_dir: Path):
        yield

    @contextmanager
    def profiling_memory(profile_dir: Path, *, keep_oom_history: bool = False):
        yield


__all__ = [
    "profiling_time",
    "profiling_memory",
]
