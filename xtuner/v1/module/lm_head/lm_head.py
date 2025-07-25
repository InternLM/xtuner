from typing import Callable, Optional, TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor


Loss: TypeAlias = torch.Tensor
Logits: TypeAlias = torch.Tensor
Weight: TypeAlias = torch.Tensor | DTensor
Bias: TypeAlias = torch.Tensor | DTensor | None
HiddenStates: TypeAlias = torch.Tensor
Labels: TypeAlias = torch.Tensor


LossFn = Callable[[Weight, Bias, HiddenStates, Labels], tuple[Loss, Logits | None]]


def ce_loss_fn(weight: Weight, bias: Bias, hidden_states: HiddenStates, labels: Labels) -> tuple[Loss, Logits]:
    logits = F.linear(hidden_states, weight, bias)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
    )
    return loss, logits


class LMHead(nn.Linear):
    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        loss_fn: LossFn = ce_loss_fn,
    ) -> tuple[Loss, Logits | None]:
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
        return loss_fn(w, b, hidden_states, labels)

    __call__: Callable[[HiddenStates, Labels, Optional[LossFn]], tuple[Loss, Logits | None]]
