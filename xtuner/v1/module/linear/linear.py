import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor
from torch.nn import functional as F


class _Linear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of the linear layer."""
        if isinstance(self.weight, DTensor):
            w = self.weight.to_local()
            if self.bias is not None:
                assert isinstance(self.bias, DTensor), "Bias should be a DTensor if weight is a DTensor"
                b = self.bias.to_local()
            else:
                b = None
        else:
            w = self.weight
            b = self.bias
        return F.linear(input, w, b)
