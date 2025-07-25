from typing import Callable, TypeAlias

import torch
import torch.nn as nn
from xtuner.v1.data_proto.loss_context import CELossForwardItem, LossContext
from torch.distributed.tensor import DTensor
import torch.nn.functional as F


Loss: TypeAlias = torch.Tensor
Logits: TypeAlias = torch.Tensor
Weight: TypeAlias = torch.Tensor | DTensor
Bias: TypeAlias = torch.Tensor | DTensor | None
HiddenStates: TypeAlias = torch.Tensor
Labels: TypeAlias = torch.Tensor


LossFn = Callable[[HiddenStates, Weight, CELossForwardItem, Bias], tuple[Loss | None, Logits | None]]


class LMHead(nn.Linear):
    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        loss_ctx: LossContext | None = None
    ) -> tuple[Loss | None, Logits | None]:
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
            return None, logits
        else:
            return loss_ctx.forward(hidden_states, w, b)

    __call__: Callable[[HiddenStates, LossContext], tuple[Loss | None, Logits | None]]
