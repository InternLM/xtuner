from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, model_validator

from xtuner.v1.config.base_model import TransformerConfig
from xtuner.v1.config.fsdp import FSDPConfig
from xtuner.v1.config.optim import LRConfig, OptimConfig


if TYPE_CHECKING:
    from .loss import GRPOLossContext


class LossConfig(BaseModel):
    model_config = ConfigDict(title="Worker config", extra="allow", arbitrary_types_allowed=True)
    cliprange_low: float
    cliprange_high: float
    ignore_idx: int = -100

    def build(self) -> "GRPOLossContext":
        from .loss import GRPOLossContext

        return GRPOLossContext(
            cliprange_low=self.cliprange_low,
            cliprange_high=self.cliprange_high,
            ignore_idx=self.ignore_idx,
        )


class WorkerConfig(BaseModel):
    model_config = ConfigDict(title="Worker config", extra="allow", arbitrary_types_allowed=True)
    model_cfg: TransformerConfig
    optim_cfg: OptimConfig
    loss_cfg: LossConfig
    lr_cfg: LRConfig
    fsdp_cfg: FSDPConfig
    work_dir: Path
    load_from: str | Path | None = None
    tokenizer_path: str | Path
    global_batch_size: int | None = None
    sp_size: int = 1
    offload_optimizer: bool = False

    @model_validator(mode="after")
    def _convert_work_dir(self):
        if isinstance(self.work_dir, str):
            self.work_dir = Path(self.work_dir)
        elif self.work_dir is None:
            self.work_dir = Path.cwd()
        return self
