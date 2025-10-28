from typing import Protocol

import torch
from typing_extensions import TypedDict


class RouterResults(TypedDict):
    logits: torch.Tensor
    router_weights: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    topkens_per_expert: torch.Tensor


class RouterProtocol(Protocol):
    def forward(self, logits: torch.Tensor) -> RouterResults: ...

    def __call__(self, logits: torch.Tensor) -> RouterResults: ...
