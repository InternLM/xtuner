# Copyright (c) OpenMMLab. All rights reserved.

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from transformers.activations import ACT2FN
from xtuner.v1.ops.rms_norm import npu_rms_norm


class NpuRMSNormGated(nn.Module):
    """NPU implementation of RMSNorm followed by a configurable gate."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, activation: str = "silu") -> None:
        super().__init__()
        if activation not in ("silu", "swish", "sigmoid"):
            raise ValueError(f"NPU gated RMSNorm does not support activation {activation!r}")
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.act = ACT2FN[activation]

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        if isinstance(weight, DTensor):
            weight = weight.to_local()
        return npu_rms_norm(x, weight, self.eps) * self.act(g)
