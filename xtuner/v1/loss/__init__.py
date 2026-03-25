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
from .mtp_loss import MTPLossContext
from .rl_loss import LogProbConfig, LogProbContext


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
    "LMHeadLossContext",
    "MTPLossContext",
    "LogProbConfig",
    "LogProbContext",
]

import torch

from xtuner.v1.utils import get_device


if get_device() == "cuda":
    from .liger_with_weights import LigerFusedLinearCrossEntropyLossWithWeights

    __all__.append("LigerFusedLinearCrossEntropyLossWithWeights")
