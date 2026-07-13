from typing import Any, TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from typing_extensions import overload

from xtuner.v1.loss.ce_loss import LMHeadLossContext


Loss: TypeAlias = torch.Tensor
Logits: TypeAlias = torch.Tensor
Weight: TypeAlias = torch.Tensor | DTensor
Bias: TypeAlias = torch.Tensor | DTensor | None
HiddenStates: TypeAlias = torch.Tensor
Labels: TypeAlias = torch.Tensor


class LMHead(nn.Linear):
    @overload  # type: ignore[override]
    def forward(
        self, hidden_states: HiddenStates, loss_ctx: None = None, *, skip_all_reduce: bool = False
    ) -> tuple[None, tuple[Logits | None, dict[str, Any]]]: ...

    @overload  # type: ignore[override]
    def forward(
        self, hidden_states: HiddenStates, loss_ctx: LMHeadLossContext, *, skip_all_reduce: bool = False
    ) -> tuple[Loss, tuple[Logits | None, dict[str, Any]]]: ...

    def forward(  # type: ignore[override]
        self, hidden_states: torch.Tensor, loss_ctx: LMHeadLossContext | None = None, *, skip_all_reduce: bool = False
    ) -> tuple[Loss | None, tuple[Logits | None, dict[str, Any]]]:
        """Forward pass of the language model head."""
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
        if loss_ctx is None:
            logits = F.linear(hidden_states, w, b)
            return None, (logits.float(), {})
        else:
            return loss_ctx.forward(hidden_states, w, b, skip_all_reduce=skip_all_reduce)

    @overload  # type: ignore
    def __call__(
        self, hidden_states: HiddenStates, loss_ctx: None = None, *, skip_all_reduce: bool = False
    ) -> tuple[None, tuple[Logits | None, dict[str, Any]]]: ...

    @overload  # type: ignore
    def __call__(
        self, hidden_states: HiddenStates, loss_ctx: LMHeadLossContext, *, skip_all_reduce: bool = False
    ) -> tuple[Loss, tuple[Logits | None, dict[str, Any]]]: ...

    __call__ = nn.Module.__call__
