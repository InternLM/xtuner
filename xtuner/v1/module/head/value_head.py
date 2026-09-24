from typing import Any, TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from typing_extensions import overload

from xtuner.v1.rl.loss.critic_loss import CriticLossContext


Loss: TypeAlias = torch.Tensor
Values: TypeAlias = torch.Tensor
Weight: TypeAlias = torch.Tensor | DTensor
Bias: TypeAlias = torch.Tensor | DTensor | None
HiddenStates: TypeAlias = torch.Tensor


class ValueHead(nn.Linear):
    """Scalar value head for critic models.

    ValueHead is an nn.Linear(hidden_size, 1) aligned with LMHead: loss_ctx is None returns fp32 values (forward-only /
    GAE), otherwise the clipped value loss from CriticLossContext.

    The runtime module name is still lm_head; the HuggingFace disk name is value_head.
    """

    def __init__(self, hidden_size: int, bias: bool = False) -> None:
        super().__init__(hidden_size, 1, bias=bias)

    @overload  # type: ignore[override]
    def forward(
        self, hidden_states: HiddenStates, loss_ctx: None = None
    ) -> tuple[None, tuple[Values | None, dict[str, Any]]]: ...

    @overload  # type: ignore[override]
    def forward(
        self, hidden_states: HiddenStates, loss_ctx: CriticLossContext
    ) -> tuple[Loss, tuple[Values | None, dict[str, Any]]]: ...

    def forward(  # type: ignore[override]
        self, hidden_states: torch.Tensor, loss_ctx: CriticLossContext | None = None
    ) -> tuple[Loss | None, tuple[Values | None, dict[str, Any]]]:
        """Forward pass of the scalar value head."""
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
            values = F.linear(hidden_states, w, b)
            return None, (values.float(), {})
        else:
            return loss_ctx.forward(hidden_states, w, b)

    @overload  # type: ignore
    def __call__(
        self, hidden_states: HiddenStates, loss_ctx: None = None
    ) -> tuple[None, tuple[Values | None, dict[str, Any]]]: ...

    @overload  # type: ignore
    def __call__(
        self, hidden_states: HiddenStates, loss_ctx: CriticLossContext
    ) -> tuple[Loss, tuple[Values | None, dict[str, Any]]]: ...

    __call__ = nn.Module.__call__
