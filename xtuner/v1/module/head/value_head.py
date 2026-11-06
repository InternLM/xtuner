from typing import Any, TypeAlias

import torch
import torch.nn as nn
from typing_extensions import overload

from xtuner.v1.loss.critic_loss import CriticLossContext

from .lm_head import LMHead


Loss: TypeAlias = torch.Tensor
Values: TypeAlias = torch.Tensor
HiddenStates: TypeAlias = torch.Tensor


class ValueHead(LMHead):
    """Scalar value head sharing the LM head's projection and loss protocol."""

    def __init__(self, hidden_size: int, bias: bool = False) -> None:
        super().__init__(hidden_size, 1, bias=bias)

    @overload  # type: ignore[override]
    def forward(
        self, hidden_states: HiddenStates, loss_ctx: None = None
    ) -> tuple[None, tuple[Values | None, dict[str, Any]]]: ...

    @overload
    def forward(
        self, hidden_states: HiddenStates, loss_ctx: CriticLossContext
    ) -> tuple[Loss, tuple[Values | None, dict[str, Any]]]: ...

    def forward(  # type: ignore[override]
        self, hidden_states: torch.Tensor, loss_ctx: CriticLossContext | None = None
    ) -> tuple[Loss | None, tuple[Values | None, dict[str, Any]]]:
        return super().forward(hidden_states, loss_ctx)  # type: ignore[arg-type]

    @overload  # type: ignore
    def __call__(
        self, hidden_states: HiddenStates, loss_ctx: None = None
    ) -> tuple[None, tuple[Values | None, dict[str, Any]]]: ...

    @overload  # type: ignore
    def __call__(
        self, hidden_states: HiddenStates, loss_ctx: CriticLossContext
    ) -> tuple[Loss, tuple[Values | None, dict[str, Any]]]: ...

    __call__ = nn.Module.__call__
