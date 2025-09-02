from contextlib import contextmanager
from pathlib import Path

import torch


if not torch.accelerator.is_available():

    @contextmanager
    def profilling_time(profile_dir: Path):
        yield

    @contextmanager
    def profilling_memory(profile_dir: Path):
        yield

elif torch.accelerator.current_accelerator().type == "cuda":
    from .cuda_profile import profilling_memory, profilling_time

elif torch.accelerator.current_accelerator().type == "npu":
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
