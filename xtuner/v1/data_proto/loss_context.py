# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mmengine.dist import dist
from typing_extensions import Self

from ..config.loss import CELossConfig


if TYPE_CHECKING:
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


# TODO: 自定义 loss 的话，这个 item 也会自定义
@dataclass
class CELossForwardItem:
    labels: torch.Tensor
    loss_reduction: str
    loss_weight: torch.Tensor
    grad_accumulation_steps: int = 1
    chunk_size: int = 1024


# TODO: 自定义 loss 的话，这个可能也有继承
@dataclass
class LossContext:
    loss_cfg: CELossConfig
    loss_forward_item: CELossForwardItem | None = None

    def build_item(
        self,
        seq_ctx: "SequenceContext",
        labels: torch.Tensor,
        grad_accumulation_steps: int,
        global_grad_tokens: torch.IntTensor | None = None,
    ) -> Self:
        device = seq_ctx.cu_seq_lens_q.device
        if self.loss_cfg.loss_reduction == "global":
            rank_grad_tokens = (labels >= 0).sum()
            assert global_grad_tokens is not None, "Global grad tokens must be provided for global reduction."
            loss_weights = rank_grad_tokens / global_grad_tokens * dist.get_world_size()
            celoss_forward_item = CELossForwardItem(
                labels=labels.to(device),
                loss_reduction=self.loss_cfg.loss_reduction,
                loss_weight=loss_weights.to(device),
                grad_accumulation_steps=grad_accumulation_steps,
                chunk_size=self.loss_cfg.chunk_size,
            )
            loss_ctx = self.__class__(
                loss_cfg=self.loss_cfg,
                loss_forward_item=celoss_forward_item,
            )
        else:
            num_tokens = seq_ctx.cu_seq_lens_q[1:] - seq_ctx.cu_seq_lens_q[:-1]
            labels_list = torch.split(labels, num_tokens.tolist(), dim=1)
            loss_weights_list = []
            for labels in labels_list:
                num_effective_tokens = (labels != self.loss_cfg.ignore_id).sum().item()
                loss_weight = len2weight(num_effective_tokens, self.loss_cfg.loss_reduction)
                loss_weights_list.append(torch.full(labels.shape, loss_weight, device=labels.device))
            loss_weights = torch.cat(loss_weights_list, dim=1)
            celoss_forward_item = CELossForwardItem(
                labels=labels.to(device),
                loss_reduction=self.loss_cfg.loss_reduction,
                loss_weight=loss_weights.to(device),
                grad_accumulation_steps=grad_accumulation_steps,
                chunk_size=self.loss_cfg.chunk_size,
            )
            loss_ctx = self.__class__(
                loss_cfg=self.loss_cfg,
                loss_forward_item=celoss_forward_item,
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
        loss_fn = self.loss_cfg.build()
        return loss_fn(
            hidden_states=hidden_states,
            head_weight=head_weight,
            head_bias=head_bias,
            loss_forward_item=self.loss_forward_item,
        )
