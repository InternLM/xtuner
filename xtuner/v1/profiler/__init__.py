from contextlib import contextmanager
from pathlib import Path

from xtuner.utils.device import get_device


if str(get_device()) == "cuda":
    from .cuda_profile import profilling_memory, profilling_time
elif str(get_device()) == "npu":
    raise NotImplementedError
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
