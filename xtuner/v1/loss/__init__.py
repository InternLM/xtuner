from .base_loss_ctx import BaseLossConfig, BaseLossContext, BaseLossKwargs
from .ce_loss import CELossConfig, CELossContext
from .chunk_loss import ChunkLoss
from .moe_loss import BalancingLoss, ZLoss


__all__ = [
    "BalancingLoss",
    "ZLoss",
    "CELossContext",
    "CELossConfig",
    "LigerFusedLinearCrossEntropyLossWithWeights",
    "ChunkLoss",
    "BaseLossConfig",
    "BaseLossContext",
    "BaseLossKwargs",
]

import torch


if torch.accelerator.is_available() and torch.accelerator.current_accelerator().type == "cuda":
    from .liger_with_weights import LigerFusedLinearCrossEntropyLossWithWeights

    __all__.append("LigerFusedLinearCrossEntropyLossWithWeights")
