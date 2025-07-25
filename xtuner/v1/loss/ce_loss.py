import torch
import torch.nn.functional as F
from mmengine.dist import all_reduce
from torch import nn

from ..data_proto.loss_context import CELossConfig, CELossForwardItem


try:
    from liger_kernel.transformers.fused_linear_cross_entropy import (
        LigerFusedLinearCrossEntropyLoss,
    )
except ImportError:
    LigerFusedLinearCrossEntropyLoss = None

from .liger_with_weights import LigerFusedLinearCrossEntropyLossWithWeights


class CrossEntropyLoss(nn.Module):
    def __init__(self, loss_cfg: CELossConfig) -> None:
        super().__init__()
        self.loss_reduction = loss_cfg.loss_reduction
        self.label_shifted = loss_cfg.label_shifted

        if self.loss_reduction == "global":
            self.loss_fct = torch.nn.CrossEntropyLoss()
        else:
            self.loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        loss_forward_item: CELossForwardItem,
        head_bias: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = F.linear(hidden_states, head_weight, head_bias)

        grad_accumulation_steps = loss_forward_item.grad_accumulation_steps
        labels = loss_forward_item.labels
        loss_weights = loss_forward_item.loss_weight

        assert grad_accumulation_steps > 0
        assert logits.shape[:-1] == labels.shape, (
            f"Logits shape {logits.shape} does not match labels shape {labels.shape}"
        )

        if self.loss_reduction in ["token", "sample", "square"]:
            assert labels.shape == loss_weights.shape, (
                f"Labels shape {labels.shape} does not match loss weights shape {loss_weights.shape}"
            )
        else:
            assert loss_weights.shape == (), (
                f"Loss weights shape {loss_weights.shape} should be a scalar for reduction {self.loss_reduction}"
            )

        if self.label_shifted:
            shift_logits = logits
            shift_labels = labels
        else:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)

        if self.loss_reduction == "global":
            loss = self.loss_fct(shift_logits, shift_labels) * loss_weights
        else:
            loss_weights = loss_weights.view(-1).to(logits.device)
            loss = self.loss_fct(shift_logits, shift_labels)

            loss_weights_sum = loss_weights.sum()
            all_reduce(loss_weights_sum, op="mean")

            loss = loss * loss_weights
            loss = loss.sum() / (loss_weights_sum + 1e-8)
            loss = loss / grad_accumulation_steps
        return loss, logits


class LigerCrossEntropyLoss(nn.Module):
    def __init__(self, loss_cfg: CELossConfig) -> None:
        super().__init__()
        self.loss_reduction = loss_cfg.loss_reduction
        self.label_shifted = loss_cfg.label_shifted

        if LigerFusedLinearCrossEntropyLoss is None:
            raise ImportError(
                "LigerFusedLinearCrossEntropyLoss is not available. Please install liger_kernel package."
            )

        if self.loss_reduction == "global":
            self.loss_fct = LigerFusedLinearCrossEntropyLoss()
        else:
            self.loss_fct = LigerFusedLinearCrossEntropyLossWithWeights(reduction="sum")

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        loss_forward_item: CELossForwardItem,
        head_bias: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        grad_accumulation_steps = loss_forward_item.grad_accumulation_steps
        labels = loss_forward_item.labels
        loss_weights = loss_forward_item.loss_weight

        assert grad_accumulation_steps > 0

        if self.loss_reduction in ["token", "sample", "square"]:
            assert labels.shape == loss_weights.shape, (
                f"Labels shape {labels.shape} does not match loss weights shape {loss_weights.shape}"
            )
        else:
            assert loss_weights.shape == (), (
                f"Loss weights shape {loss_weights.shape} should be a scalar for reduction {self.loss_reduction}"
            )

        if self.label_shifted:
            shift_hidden_states = hidden_states
            shift_labels = labels
        else:
            shift_hidden_states = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

        shift_hidden_states = shift_hidden_states.view(-1, shift_hidden_states.size(-1))
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_hidden_states.device)

        if self.loss_reduction == "global":
            loss = self.loss_fct(head_weight, shift_hidden_states, shift_labels, head_bias) * loss_weights
        else:
            loss_weights = loss_weights.view(-1).to(shift_hidden_states.device)
            loss_weights_sum = loss_weights.sum()
            all_reduce(loss_weights_sum, op="mean")

            loss = self.loss_fct(
                head_weight,
                shift_hidden_states,
                shift_labels,
                head_bias,
                loss_weights,
                loss_weights_sum,
                grad_accumulation_steps,
            )
        return loss, None
