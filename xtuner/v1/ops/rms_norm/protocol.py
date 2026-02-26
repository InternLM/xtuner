from typing import Protocol
from typing_extensions import Literal

import torch


class RMSNormProtocol(Protocol):
    def __call__(self, x: torch.Tensor, weight: torch.Tensor, epsilon: float, type: Literal['default', 'zero_centered']) -> torch.Tensor: ...
