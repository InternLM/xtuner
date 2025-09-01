from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class TensorboardWriter:
    def __init__(
        self,
        log_dir: str | Path | None = None,
    ):
        if log_dir is None:
            log_dir = Path()

        if isinstance(log_dir, str):
            log_dir = Path(log_dir)

        self._writer = SummaryWriter(log_dir=log_dir)

    def add_scalar(
        self,
        *,
        tag: str,
        scalar_value: float,
        global_step: int,
    ):
        self._writer.add_scalar(tag, scalar_value, global_step)

    def add_scalars(
        self,
        *,
        tag_scalar_dict: dict[str, float],
        global_step: int,
    ):
        for tag, scalar_value in tag_scalar_dict.items():
            self._writer.add_scalar(tag, scalar_value, global_step)

    def close(self):
        self._writer.close()
