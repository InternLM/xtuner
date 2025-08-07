from typing import Callable, Literal, cast

import torch
import torch.nn.functional as F
from mmengine.dist import get_world_size
from pydantic import BaseModel
from torch import distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from typing_extensions import Self

from ..data_proto.utils import pad_to_multiple_of, split_for_sequence_parallel
from .utils import cal_global_grad_tokens, cal_global_sum_loss_weight


class CEForwardItem(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    labels: torch.Tensor
    loss_reduction: str
    loss_weight: torch.Tensor
    chunk_size: int = 1024
    sequence_parallel_mesh: DeviceMesh | None = None


# TODO: 如何更通用
class BaseChunkLoss(nn.Module):
    def __init__(self, ctx, chunk_loss_class, chunk_loss_fn) -> None:
        super().__init__()
        self.ctx = ctx
        self.chunk_loss_class = chunk_loss_class
        self.chunk_loss_fn = chunk_loss_fn

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        forward_item: CEForwardItem,
        head_bias: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # 注意： 不能用 kwargs 传参数
        loss, logits = self.chunk_loss_class.apply(
            hidden_states, head_weight, forward_item, head_bias, self.chunk_loss_fn
        )
        sp_mesh = forward_item.sequence_parallel_mesh
        if sp_mesh and sp_mesh.size() > 1:
            loss = dist.nn.all_reduce(loss, group=sp_mesh.get_group())
        return loss, logits


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

    def build_list_ctx(self, data_batch: list, data_mesh=None, device=None) -> list:
        if data_mesh is None:
            dp_size = get_world_size()
            sp_mesh = None
        else:
            dp_mesh = data_mesh["dp"]
            dp_size = dp_mesh.size()
            # TODO: 需要判断 sp_mesh 是否和 seq_ctx 里面是否一致，不一致要么报错或者赋予 sp？
            sp_mesh = data_mesh["sp"]

        if device is None:
            device = data_batch[0]["seq_ctx"].device

        label_list = []
        seq_ctx_list = []
        num_tokens_list = []
        for data in data_batch:
            labels = data["labels"]
            data["labels"] = labels.to(device)
            data["seq_ctx"] = data["seq_ctx"].to(device)
            label_list.append(data["labels"])
            seq_ctx_list.append(data["seq_ctx"])
            num_tokens = data["seq_ctx"].cu_seq_lens_q[1:] - data["seq_ctx"].cu_seq_lens_q[:-1]
            num_tokens_list.append(num_tokens)

        if self.loss_reduction == "global":
            global_grad_tokens = cal_global_grad_tokens(label_list, sp_mesh)
            loss_weights = 1 / (global_grad_tokens + 1e-8) * dp_size
            batch_loss_weights = [loss_weights] * len(label_list)
        else:
            global_grad_tokens, batch_loss_weights = cal_global_sum_loss_weight(
                label_list, num_tokens_list, self.loss_reduction, sp_mesh
            )
            for i in range(len(batch_loss_weights)):
                batch_loss_weights[i] = batch_loss_weights[i] / (global_grad_tokens + 1e-8) * dp_size

        loss_ctx_list = []
        for seq_ctx, labels, loss_weights in zip(seq_ctx_list, label_list, batch_loss_weights):
            forward_item = CEForwardItem(
                labels=labels,
                loss_reduction=self.loss_reduction,
                loss_weight=loss_weights,
                chunk_size=self.chunk_size,
            )
            loss_ctx = self.__class__(
                loss_class=self.loss_class,
                loss_reduction=self.loss_reduction,
                label_shifted=self.label_shifted,
                forward_item=forward_item,
            )
            loss_ctx_list.append(loss_ctx)

        assert len(loss_ctx_list) == len(data_batch)
        ret_data_batch = [
            {"seq_ctx": seq_ctx, "loss_ctx": loss_ctx} for seq_ctx, loss_ctx in zip(seq_ctx_list, loss_ctx_list)
        ]

        if sp_mesh is not None:
            for data in ret_data_batch:
                loss_ctx = data["loss_ctx"]
                seq_ctx = data["seq_ctx"]

                seq_ctx = seq_ctx.split(sp_mesh)
                loss_ctx = loss_ctx.split(sp_mesh)
                data["seq_ctx"] = seq_ctx
                data["loss_ctx"] = loss_ctx

        return ret_data_batch

    def split(self, sequence_parallel_mesh=None) -> Self:
        if sequence_parallel_mesh is None:
            return self

        forward_item = self.forward_item
        assert forward_item is not None
        forward_item.sequence_parallel_mesh = sequence_parallel_mesh
        labels = forward_item.labels
        loss_weight = forward_item.loss_weight

        multiple_of = sequence_parallel_mesh.size()
        pad_labels = pad_to_multiple_of(labels, -100, multiple_of, 1)
        sp_labels = cast(
            torch.LongTensor, split_for_sequence_parallel(pad_labels, dim=1, sp_mesh=sequence_parallel_mesh)
        )
        if loss_weight.shape != ():
            pad_loss_weight = pad_to_multiple_of(loss_weight, 0, multiple_of, 1)
            sp_loss_weight = cast(
                torch.Tensor, split_for_sequence_parallel(pad_loss_weight, dim=1, sp_mesh=sequence_parallel_mesh)
            )
        else:
            sp_loss_weight = loss_weight

        forward_item.labels = sp_labels
        forward_item.loss_weight = sp_loss_weight
        return self

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
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
            self.loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
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
        logits = F.linear(hidden_states, head_weight, head_bias).float()

        labels = forward_item.labels
        loss_weights = forward_item.loss_weight

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
            rank_grad_tokens = (shift_labels >= 0).sum()
            if rank_grad_tokens == 0:
                loss = shift_logits.sum() * 0
            else:
                loss = self.loss_fct(shift_logits, shift_labels) * loss_weights
        else:
            loss_weights = loss_weights.view(-1).to(logits.device)
            loss = self.loss_fct(shift_logits, shift_labels)
            loss = loss * loss_weights
            loss = loss.sum()

        sequence_parallel_mesh = forward_item.sequence_parallel_mesh
        if sequence_parallel_mesh is not None and sequence_parallel_mesh.size() > 1:
            loss = dist.nn.all_reduce(loss, group=sequence_parallel_mesh.get_group())
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
            self.loss_fct = LigerFusedLinearCrossEntropyLoss(reduction="sum")
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
        labels = forward_item.labels
        loss_weights = forward_item.loss_weight

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
            loss_weights = loss_weights.view(-1)
            loss = self.loss_fct(head_weight, shift_hidden_states, shift_labels, head_bias, loss_weights)
        sequence_parallel_mesh = forward_item.sequence_parallel_mesh
        if sequence_parallel_mesh is not None and sequence_parallel_mesh.size() > 1:
            loss = dist.nn.all_reduce(loss, group=sequence_parallel_mesh.get_group())
        return loss, None


class ChunkCELoss(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        forward_item: CEForwardItem,
        head_bias: torch.Tensor | None,
        loss_fn: Callable,
    ):
        chunk_size = forward_item.chunk_size
        labels = forward_item.labels
        loss_weight = forward_item.loss_weight

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
                loss_reduction=forward_item.loss_reduction,
                loss_weight=loss_weight_chunk,
                labels=labels_chunk,
                chunk_size=chunk_size,
            )
            chunk_loss, chunk_grad_input, chunk_grad_weight = accumulate_chunk(
                hidden_states_chunk,
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


def _chunk_loss(hidden_states_chunk, head_weight, loss_fn, forward_item: CEForwardItem):
    logits_chunk = hidden_states_chunk @ head_weight.t()
    return loss_fn(logits_chunk, forward_item)


def accumulate_chunk(hidden_states_chunk, head_weight, loss_fn, forward_item: CEForwardItem):
    (chunk_grad_input, chunk_grad_weight), chunk_loss = torch.func.grad_and_value(
        _chunk_loss, argnums=(0, 1), has_aux=False
    )(hidden_states_chunk, head_weight, loss_fn, forward_item)
    return chunk_loss, chunk_grad_input, chunk_grad_weight


def chunk_ce_loss(logits: torch.Tensor, forward_item: CEForwardItem):
    logits = logits.float().reshape(-1, logits.shape[-1])
    labels = forward_item.labels.reshape(-1)
    if forward_item.loss_reduction == "global":
        loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
        rank_grad_tokens = (labels >= 0).sum()
        if rank_grad_tokens == 0:
            loss = logits.sum() * 0
        else:
            loss = loss_fct(logits, labels) * forward_item.loss_weight
    else:
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(logits, labels)

        loss = loss * forward_item.loss_weight
        loss = loss.sum()
    return loss
