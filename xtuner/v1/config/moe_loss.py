from typing import Literal

from pydantic import BaseModel


class MoELossConfig(BaseModel):
    balancing_loss_type: Literal["sigmoid", "softmax"]
    balancing_loss_alpha: float = 0.001
    balancing_loss_global_average: bool = True
    z_loss_alpha: float = 0.001
    z_loss_global_average: bool = True
