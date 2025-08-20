from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from xtuner.v1.config.base_model import TransformerConfig
from xtuner.v1.config.fsdp import FSDPConfig
from xtuner.v1.config.optim import LRConfig, OptimConfig


if TYPE_CHECKING:
    from .loss import GRPOLoss


class LossConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    policy_loss_cfg: dict[str, Any]
    ignore_idx: int = -100
    use_kl_loss: bool = False
    kl_loss_coef: float = 0.001
    kl_loss_type: Literal["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"] | None = None
    mode: Literal["eager", "chunk"] = "eager"
    chunk_size: int | None = None

    def build(self) -> "GRPOLoss":
        from .loss import GRPOLoss

        return GRPOLoss(
            policy_loss_cfg=self.policy_loss_cfg,
            ignore_idx=self.ignore_idx,
            use_kl_loss=self.use_kl_loss,
            kl_loss_coef=self.kl_loss_coef,
            kl_loss_type=self.kl_loss_type,
            mode=self.mode,
            chunk_size=self.chunk_size,
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

    ref_model_fsdp_cfg: FSDPConfig | None = None

    @model_validator(mode="after")
    def _convert_work_dir(self):
        if isinstance(self.work_dir, str):
            self.work_dir = Path(self.work_dir)
        elif self.work_dir is None:
            self.work_dir = Path.cwd()
        return self
