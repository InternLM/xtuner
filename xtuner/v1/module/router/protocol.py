from typing import Protocol

import torch
from typing_extensions import TypedDict

from xtuner.v1.config import MoEConfig


class RouterResults(TypedDict):
    logits: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    topkens_per_expert: torch.Tensor


class RouterProtocol(Protocol):
    def __init__(self, config: MoEConfig):
        """Initialize the router with the given MoE configuration."""
        ...

    def forward(self, logits: torch.Tensor) -> RouterResults: ...

    def __call__(self, logits: torch.Tensor) -> RouterResults: ...
