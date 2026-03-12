from .base_loss_ctx import BaseLossConfig, BaseLossContext, BaseLossKwargs
from .ce_loss import CELossConfig, CELossContext
from .chunk_loss import ChunkLoss
from .moe_loss import (
    BalancingLoss,
    BalancingLossConfig,
    BalancingLossContext,
    BalancingLossKwargs,
    ZLoss,
    ZLossConfig,
    ZLossContext,
    ZLossKwargs,
)


__all__ = [
    "BalancingLoss",
    "BalancingLossConfig",
    "BalancingLossContext",
    "BalancingLossKwargs",
    "ZLoss",
    "ZLossConfig",
    "ZLossContext",
    "ZLossKwargs",
    "CELossContext",
    "CELossConfig",
    "ChunkLoss",
    "BaseLossConfig",
    "BaseLossContext",
    "BaseLossKwargs",
]

import torch

from xtuner.v1.utils import get_device


if get_device() == "cuda":
    from .liger_with_weights import LigerFusedLinearCrossEntropyLossWithWeights

    __all__.append("LigerFusedLinearCrossEntropyLossWithWeights")
