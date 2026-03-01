# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn
from typing import Literal
from torch.distributed.tensor import DTensor

from xtuner.v1.ops import rms_norm, zero_centered_rms_norm


class RMSNorm(nn.Module):
    weight: torch.Tensor

    def __init__(self, hidden_size: int, eps: float = 1e-6, type: Literal['default', 'zero_centered'] = 'default'):
        """RMSNorm is equivalent to T5LayerNorm."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self._type = type

        if type == 'default':
            self.rms_norm_fn = rms_norm
        elif type == 'zero_centered':
            self.rms_norm_fn = zero_centered_rms_norm
        else:
            raise ValueError(f'Unsupported RMSNorm type: {type}')

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if isinstance(self.weight, DTensor):
            weight = self.weight.to_local()
        else:
            weight = self.weight

        # just for align
        # input_dtype = hidden_states.dtype
        # hidden_states = hidden_states.to(torch.float32)
        # variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # return (weight * hidden_states).to(input_dtype)  # gpt_oss
        # return weight * hidden_states.to(input_dtype)  # Llama
        return self.rms_norm_fn(hidden_states, weight, epsilon=self.variance_epsilon)  # type: ignore[operator]

    def init_weights(self):
        self.weight.data.fill_(1.0)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, type={self._type}, eps={self.variance_epsilon}"
