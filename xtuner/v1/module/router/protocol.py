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
    # `input_ids` is keyword-only so existing positional calls (logits, rollout_routed_experts)
    # remain backward-compatible. Only HashRouter consumes it; other routers ignore it.
    def forward(
        self,
        logits: torch.Tensor,
        rollout_routed_experts: torch.Tensor | None = None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> RouterResults: ...

    def __call__(
        self,
        logits: torch.Tensor,
        rollout_routed_experts: torch.Tensor | None = None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> RouterResults: ...
