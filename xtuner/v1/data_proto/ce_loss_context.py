# Copyright (c) OpenMMLab. All rights reserved.
from typing import TYPE_CHECKING, Literal

import torch
from mmengine.dist import all_reduce, dist
from pydantic import BaseModel
from typing_extensions import Self


if TYPE_CHECKING:
    from xtuner.v1.loss.base_chunk_loss import BaseChunkLoss
    from xtuner.v1.loss.ce_loss import CrossEntropyLoss, LigerCrossEntropyLoss

    from . import SequenceContext


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
        from xtuner.v1.loss.base_chunk_loss import BaseChunkLoss
        from xtuner.v1.loss.ce_loss import CrossEntropyLoss, LigerCrossEntropyLoss
        from xtuner.v1.loss.chunk_ce_loss import ChunkCELoss, chunk_ce_loss

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
        seq_ctx: "SequenceContext",
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
