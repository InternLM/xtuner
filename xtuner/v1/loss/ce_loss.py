from typing import Literal

import torch
import torch.nn.functional as F
from mmengine.dist import all_reduce, dist
from pydantic import BaseModel
from torch import nn
from typing_extensions import Self

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.loss.base_chunk_loss import BaseChunkLoss


class CEForwardItem(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    labels: torch.Tensor
    loss_reduction: str
    loss_weight: torch.Tensor
    global_sum_loss_weight: torch.Tensor | None = None
    grad_accumulation_steps: int = 1
    chunk_size: int = 1024


class CELossContext(BaseModel):
    loss_reduction: Literal["global", "token", "sample", "square"] = "global"
    label_shifted: bool = True
    ignore_idx: int = -100
    loss_class: Literal["cross_entropy", "liger_cross_entropy", "chunk_cross_entropy"] = "cross_entropy"
    chunk_size: int = 1024
    chunk_loss_fn: Literal["chunk_ce_loss"] = "chunk_ce_loss"

    forward_item: CEForwardItem | None = None

    def build_loss_fn(self) -> "CrossEntropyLoss | LigerCrossEntropyLoss | BaseChunkLoss":
        if self.loss_class == "cross_entropy":
            return CrossEntropyLoss(ctx=self)
        elif self.loss_class == "liger_cross_entropy":
            return LigerCrossEntropyLoss(ctx=self)
        elif self.loss_class == "chunk_cross_entropy":
            if self.chunk_loss_fn == "chunk_ce_loss":
                chunk_loss_fn = chunk_ce_loss
            else:
                raise NotImplementedError
            # TODO: 这个类只能定义基本结构，无法保证任何场景都不自定义
            return BaseChunkLoss(ctx=self, chunk_loss_class=ChunkCELoss, chunk_loss_fn=chunk_loss_fn)
        else:
            raise NotImplementedError

    def build_forward_item(
        self,
        seq_ctx: SequenceContext,
        labels: torch.Tensor,
        grad_accumulation_steps: int,
        global_grad_tokens: torch.IntTensor | None = None,
    ) -> Self:
        device = seq_ctx.cu_seq_lens_q.device
        if self.loss_reduction == "global":
            rank_grad_tokens = (labels >= 0).sum()
            assert global_grad_tokens is not None, "Global grad tokens must be provided for global reduction."
            loss_weights = rank_grad_tokens / global_grad_tokens * dist.get_world_size()
            celoss_forward_item = CEForwardItem(
                labels=labels.to(device),
                loss_reduction=self.loss_reduction,
                loss_weight=loss_weights.to(device),
                grad_accumulation_steps=grad_accumulation_steps,
                chunk_size=self.chunk_size,
            )
            loss_ctx = self.__class__(
                loss_reduction=self.loss_reduction,
                label_shifted=self.label_shifted,
                forward_item=celoss_forward_item,
            )
        else:
            labels = labels.to(device)
            num_tokens = seq_ctx.cu_seq_lens_q[1:] - seq_ctx.cu_seq_lens_q[:-1]
            labels_list = torch.split(labels, num_tokens.tolist(), dim=1)
            loss_weights_list = []
            for labels in labels_list:
                num_effective_tokens = (labels != self.ignore_id).sum().item()
                loss_weight = len2weight(num_effective_tokens, self.loss_reduction)
                loss_weights_list.append(torch.full(labels.shape, loss_weight, device=labels.device))
            loss_weights = torch.cat(loss_weights_list, dim=1)

            global_sum_loss_weight = loss_weights.sum()
            all_reduce(global_sum_loss_weight, op="mean")

            forward_item = CEForwardItem(
                labels=labels,
                loss_reduction=self.loss_reduction,
                loss_weight=loss_weights,
                global_sum_loss_weight=global_sum_loss_weight,
                grad_accumulation_steps=grad_accumulation_steps,
                chunk_size=self.chunk_size,
            )
            loss_ctx = self.__class__(
                loss_reduction=self.loss_reduction,
                label_shifted=self.label_shifted,
                forward_item=forward_item,
            )
        return loss_ctx

    # TODO: 没有测试 sp 逻辑
    def split(self, sequence_parallel_mesh) -> Self:
        raise NotImplementedError()

    def forward(
        self,
        hidden_states: torch.Tensor | None = None,
        head_weight: torch.Tensor | None = None,
        head_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        loss_fn = self.build_loss_fn()
        return loss_fn(
            hidden_states=hidden_states,
            head_weight=head_weight,
            head_bias=head_bias,
            forward_item=self.forward_item,
        )


class CrossEntropyLoss(nn.Module):
    def __init__(self, ctx: CELossContext) -> None:
        super().__init__()
        self.loss_reduction = ctx.loss_reduction
        self.label_shifted = ctx.label_shifted

        if self.loss_reduction == "global":
            self.loss_fct = torch.nn.CrossEntropyLoss()
        else:
            self.loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        forward_item: CEForwardItem,
        head_bias: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = F.linear(hidden_states, head_weight, head_bias)

        grad_accumulation_steps = forward_item.grad_accumulation_steps
        labels = forward_item.labels
        loss_weights = forward_item.loss_weight

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
            loss_weights_sum = forward_item.global_sum_loss_weight
            assert loss_weights_sum is not None
            loss = loss * loss_weights
            loss = loss.sum() / (loss_weights_sum + 1e-8)
            loss = loss / grad_accumulation_steps
        return loss, logits


class LigerCrossEntropyLoss(nn.Module):
    def __init__(self, ctx: CELossContext) -> None:
        super().__init__()
        self.loss_reduction = ctx.loss_reduction
        self.label_shifted = ctx.label_shifted

        try:
            from liger_kernel.transformers.fused_linear_cross_entropy import (
                LigerFusedLinearCrossEntropyLoss,
            )

            from .liger_with_weights import LigerFusedLinearCrossEntropyLossWithWeights
        except ImportError as e:
            raise ImportError(
                f"`LigerFusedLinearCrossEntropyLoss` is not available for {e}. Please install liger_kernel package."
            )

        if self.loss_reduction == "global":
            self.loss_fct = LigerFusedLinearCrossEntropyLoss()
        else:
            self.loss_fct = LigerFusedLinearCrossEntropyLossWithWeights(reduction="sum")

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        forward_item: CEForwardItem,
        head_bias: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        grad_accumulation_steps = forward_item.grad_accumulation_steps
        labels = forward_item.labels
        loss_weights = forward_item.loss_weight

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
            loss_weights_sum = forward_item.global_sum_loss_weight
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


class ChunkCELoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, head_weight, froward_item: CEForwardItem, head_bias, loss_fn):
        chunk_size = froward_item.chunk_size
        labels = froward_item.labels
        loss_weight = froward_item.loss_weight

        device = hidden_states.device
        accumulated_loss = torch.tensor(0.0, device=device)
        grad_inputs = torch.empty_like(hidden_states)
        grad_weight = torch.zeros_like(head_weight)

        grad_inputs_chunks = torch.split(grad_inputs, chunk_size, dim=1)
        hidden_states_chunks = torch.split(hidden_states, chunk_size, dim=1)
        labels_chunks = torch.split(labels, chunk_size, dim=1)

        if loss_weight.shape != ():
            loss_weight_chunks = torch.split(loss_weight, chunk_size, dim=1)
        else:
            loss_weight_chunks = [loss_weight] * len(hidden_states_chunks)  # type: ignore[assignment]

        for i in range(len(hidden_states_chunks)):
            hidden_states_chunk = hidden_states_chunks[i]
            labels_chunk = labels_chunks[i]
            loss_weight_chunk = loss_weight_chunks[i]
            grad_inputs_chunk = grad_inputs_chunks[i]

            chunk_forward_item = CEForwardItem(
                loss_reduction=froward_item.loss_reduction,
                loss_weight=loss_weight_chunk,
                grad_accumulation_steps=froward_item.grad_accumulation_steps,
                global_sum_loss_weight=froward_item.global_sum_loss_weight,
                chunk_size=chunk_size,
            )
            chunk_loss, chunk_grad_input, chunk_grad_weight = accumulate_chunk(
                hidden_states_chunk,
                labels_chunk,
                head_weight,
                loss_fn,
                forward_item=chunk_forward_item,
            )
            accumulated_loss.add_(chunk_loss)
            grad_inputs_chunk.copy_(chunk_grad_input)
            grad_weight.add_(chunk_grad_weight)

        ctx.save_for_backward(grad_inputs, grad_weight)
        return accumulated_loss, None

    @staticmethod
    def backward(ctx, *grad_output):
        grad_input, grad_weight = ctx.saved_tensors
        if torch.ne(grad_output[0], torch.tensor(1.0, device=grad_output[0].device)):
            grad_input = grad_input * grad_output[0]
            grad_weight = grad_weight * grad_output[0]

        return grad_input, grad_weight, None, None, None


def chunk_ce_loss(logits, labels, forward_item: CEForwardItem):
    if forward_item.loss_reduction == "global":
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels) * forward_item.loss_weight
    else:
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(logits, labels)

        loss = loss * forward_item.loss_weight
        assert forward_item.global_sum_loss_weight is not None
        loss = loss.sum() / (forward_item.global_sum_loss_weight + 1e-8)
        loss = loss / forward_item.grad_accumulation_steps
    return loss


def len2weight(x, loss_reduction):
    if x == 0:
        return x
    if loss_reduction == "token":
        return 1
    if loss_reduction == "sample":
        return 1 / x
    if loss_reduction == "square":
        return 1 / (x**0.5)
    raise NotImplementedError(loss_reduction)


def _chunk_loss(hidden_states_chunk, labels_chunk, head_weight, loss_fn, forward_item: CEForwardItem):
    logits_chunk = hidden_states_chunk @ head_weight.t()
    return loss_fn(logits_chunk.float().view(-1, logits_chunk.shape[-1]), labels_chunk.view(-1), forward_item)


def accumulate_chunk(hidden_states_chunk, labels_chunk, head_weight, loss_fn, forward_item: CEForwardItem):
    (chunk_grad_input, chunk_grad_weight), chunk_loss = torch.func.grad_and_value(
        _chunk_loss, argnums=(0, 2), has_aux=False
    )(hidden_states_chunk, labels_chunk, head_weight, loss_fn, forward_item)
    return chunk_loss, chunk_grad_input, chunk_grad_weight
