from pathlib import Path
from typing import Protocol

from .jsonl_writer import JsonlWriter
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


def get_writer(
    *,
    writer_type: str,
    log_dir: str | Path | None = None,
) -> Writer:
    if writer_type == "jsonl":
        return JsonlWriter(log_dir=log_dir)
    elif writer_type == "tensorboard":
        return TensorboardWriter(log_dir=log_dir)
    else:
        raise ValueError(f"Unknown writer type: {writer_type}")


__all__ = ["JsonlWriter", "TensorboardWriter"]
