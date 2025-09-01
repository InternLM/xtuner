from .ce_loss import BaseChunkLoss, CEForwardItem, CELossContext, ChunkCELoss, CrossEntropyLoss
from .chunk_loss import ChunkLoss
from .moe_loss import BalancingLoss, ZLoss


__all__ = [
    "BalancingLoss",
    "ZLoss",
    "BaseChunkLoss",
    "CrossEntropyLoss",
    "CELossContext",
    "ChunkCELoss",
    "LigerFusedLinearCrossEntropyLossWithWeights",
    "CEForwardItem",
    "ChunkLoss",
]

import torch


if torch.accelerator.is_available() and torch.accelerator.current_accelerator().type == "cuda":
    from .liger_with_weights import LigerFusedLinearCrossEntropyLossWithWeights

    __all__.append("LigerFusedLinearCrossEntropyLossWithWeights")
