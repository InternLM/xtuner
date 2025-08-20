from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from ..grpo.config import WorkerConfig as GRPOWorkerConfig


if TYPE_CHECKING:
    from xtuner.v1.rl.grpo.oreal_loss import OrealLoss


class LossConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    policy_loss_cfg: dict[str, Any]
    ignore_idx: int = -100
    use_kl_loss: bool = False
    kl_loss_coef: float = 0.001
    kl_loss_type: Literal["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"] | None = None
    positive_loss_factor: float = 1.0
    pos_sft_loss_weight: float = 1.0
    pos_policy_loss_weight: float = 1.0
    negative_loss_factor: float = 1.0
    mode: Literal["eager", "chunk"] = "eager"
    chunk_size: int | None = None

    def build(self) -> "OrealLoss":
        from xtuner.v1.rl.grpo.oreal_loss import OrealLoss

        return OrealLoss(
            policy_loss_cfg=self.policy_loss_cfg,
            ignore_idx=self.ignore_idx,
            positive_loss_factor=self.positive_loss_factor,
            pos_sft_loss_weight=self.pos_sft_loss_weight,
            pos_policy_loss_weight=self.pos_policy_loss_weight,
            negative_loss_factor=self.negative_loss_factor,
            mode=self.mode,
            chunk_size=self.chunk_size,
            use_kl_loss=self.use_kl_loss,
            kl_loss_coef=self.kl_loss_coef,
            kl_loss_type=self.kl_loss_type,
        )


class WorkerConfig(GRPOWorkerConfig):
    pass
