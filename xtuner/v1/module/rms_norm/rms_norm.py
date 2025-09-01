# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn
from torch.distributed.tensor import DTensor

from xtuner.v1.ops import rms_norm


class RMSNorm(nn.Module):
    weight: torch.Tensor

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """RMSNorm is equivalent to T5LayerNorm."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if isinstance(self.weight, DTensor):
            weight = self.weight.to_local()
        else:
            weight = self.weight

        return rms_norm(hidden_states, weight, epsilon=self.variance_epsilon)

    def init_weights(self):
        self.weight.data.fill_(1.0)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
