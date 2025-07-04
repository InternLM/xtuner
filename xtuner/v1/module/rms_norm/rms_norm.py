# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn
from torch.distributed.tensor import DTensor, Partial


class RMSNorm(nn.Module):
    weight: torch.Tensor

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """RMSNorm is equivalent to T5LayerNorm."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if isinstance(self.weight, DTensor):
            weight = self.weight.to_local(grad_placements=(Partial("avg"),))
        else:
            weight = self.weight

        # return F.rms_norm(hidden_states, weight.shape, weight, self.variance_epsilon)
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
