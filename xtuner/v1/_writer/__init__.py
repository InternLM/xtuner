from pathlib import Path
from typing import Any, Protocol

from .jsonl_writer import JsonlWriter
from .swanlab_writer import SwanLabConfig, SwanLabWriter
from .tb_writer import TensorboardWriter


class Writer(Protocol):
    def __init__(
        self,
        log_dir: str | Path | None = None,
    ): ...

    def add_scalar(
        self,
        *,
        tag: str,
        scalar_value: float,
        global_step: int,
    ): ...

    def add_scalars(
        self,
        *,
        tag_scalar_dict: dict[str, float],
        global_step: int,
    ): ...

    def close(self) -> None: ...


class NoOpWriter:
    """Ignore experiment metrics on non-logging ranks."""

    def add_scalar(self, *, tag: str, scalar_value: float, global_step: int):
        pass

    def add_scalars(self, *, tag_scalar_dict: dict[str, float], global_step: int):
        pass

    def close(self) -> None:
        pass


def get_writer(
    *,
    writer_type: str,
    log_dir: str | Path | None = None,
    swanlab_config: SwanLabConfig | None = None,
    resume_run: dict[str, Any] | None = None,
) -> Writer:
    if writer_type == "jsonl":
        return JsonlWriter(log_dir=log_dir)
    elif writer_type == "tensorboard":
        return TensorboardWriter(log_dir=log_dir)
    elif writer_type == "swanlab":
        return SwanLabWriter(log_dir=log_dir, config=swanlab_config, resume_run=resume_run)
    else:
        raise ValueError(f"Unknown writer type: {writer_type}")


__all__ = ["JsonlWriter", "NoOpWriter", "SwanLabConfig", "SwanLabWriter", "TensorboardWriter"]
