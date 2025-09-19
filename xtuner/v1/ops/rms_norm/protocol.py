from typing import Protocol

import torch


class RMSNormProtocol(Protocol):
    def __call__(self, x: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor: ...
