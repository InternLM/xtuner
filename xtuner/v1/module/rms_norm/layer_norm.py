# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from xtuner.v1.utils.dtensor import materialize_full


class LayerNorm(nn.Module):
    """``nn.LayerNorm`` that unwraps DTensor parameters before the functional
    call.

    ``F.layer_norm`` is not DTensor-aware, so the weight and bias have to be materialized on
    each forward -- see :func:`~xtuner.v1.utils.dtensor.materialize_full`. Used by the DSA
    indexers, whose ``k_norm`` is a LayerNorm rather than the RMSNorm the rest of the stack uses.

    Args:
        hidden_size (int): Normalized dimension.
        eps (float): Numerical stabilizer.
    """

    weight: torch.Tensor
    bias: torch.Tensor

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.normalized_shape = (hidden_size,)
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        weight = materialize_full(self.weight, name="layer_norm.weight")
        bias = materialize_full(self.bias, name="layer_norm.bias")
        return torch.nn.functional.layer_norm(hidden_states, self.normalized_shape, weight, bias, self.eps)

    def init_weights(self) -> None:
        self.weight.data.fill_(1.0)
        self.bias.data.zero_()

    def extra_repr(self):
        return f"{self.normalized_shape}, eps={self.eps}"
