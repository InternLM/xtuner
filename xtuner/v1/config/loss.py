from typing import TYPE_CHECKING

from pydantic import BaseModel


if TYPE_CHECKING:
    from xtuner.v1.loss import BalancingLoss, ZLoss


class BalancingLossConfig(BaseModel):
    balancing_loss_alpha: float = 0.001
    balancing_loss_global_average: bool = True

    def build(self, router_scoring_func) -> "BalancingLoss":
        from xtuner.v1.loss import BalancingLoss

        return BalancingLoss(
            self.balancing_loss_alpha,
            self.balancing_loss_global_average,
            router_scoring_func=router_scoring_func,
        )


class ZLossConfig(BaseModel):
    z_loss_alpha: float = 0.001
    z_loss_global_average: bool = True

    def build(self) -> "ZLoss":
        from xtuner.v1.loss import ZLoss

        return ZLoss(
            self.z_loss_alpha,
            self.z_loss_global_average,
        )
