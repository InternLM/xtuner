from contextlib import contextmanager
from pathlib import Path

from xtuner.v1.utils import get_device


if get_device() == "cuda":
    from .cuda_profile import profilling_memory, profilling_time
elif get_device() == "npu":
    from .npu_profile import profilling_memory, profilling_time

else:

    @contextmanager
    def profilling_time(profile_dir: Path):
        yield

    @contextmanager
    def profilling_memory(profile_dir: Path):
        yield


__all__ = [
    "profilling_time",
    "profilling_memory",
]
