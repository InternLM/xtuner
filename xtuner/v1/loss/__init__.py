from .aux_loss import AuxLossConfig, AuxLossContext
from .base_loss_ctx import BaseLossConfig, BaseLossContext, BaseLossKwargs
from .ce_loss import CELossConfig, CELossContext, LMHeadLossContext
from .chunk_loss import ChunkLoss
from .moe_loss import (
    BalancingLossConfig,
    BalancingLossContext,
    BalancingLossKwargs,
    ZLossConfig,
    ZLossContext,
    ZLossKwargs,
)
from .mtp_loss import MTPE2ETVLossContext, MTPLossContext
from .rl_loss import LogProbConfig, LogProbContext, TopKLogProbConfig, TopKLogProbContext


__all__ = [
    "BalancingLossConfig",
    "BalancingLossContext",
    "BalancingLossKwargs",
    "AuxLossConfig",
    "AuxLossContext",
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
    "MTPE2ETVLossContext",
    "LogProbConfig",
    "LogProbContext",
    "TopKLogProbConfig",
    "TopKLogProbContext",
]

import torch

from xtuner.v1.utils import get_device


if get_device() == "cuda":
    from .liger_with_weights import LigerFusedLinearCrossEntropyLossWithWeights

    __all__.append("LigerFusedLinearCrossEntropyLossWithWeights")


def get_flce_loss_cls():
    """Return the fused-linear-cross-entropy loss class for this device.

    NPU returns the in-package implementation (``.liger_npu``) because
    liger-ascend's stock plain CE kernel miscomputes ``loss_1d`` in
    multi-rank training; CUDA returns liger_kernel's stock class. Other
    devices have no fused path and fail loudly here. Dispatch is by device
    only, no env knob.
    """
    device = get_device()
    if device == "npu":
        from .liger_npu import NPUFusedLinearCrossEntropyLoss

        return NPUFusedLinearCrossEntropyLoss
    if device == "cuda":
        from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss

        return LigerFusedLinearCrossEntropyLoss
    raise NotImplementedError(f"Fused-linear-cross-entropy loss is not implemented on {device}")

