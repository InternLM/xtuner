# Copyright (c) OpenMMLab. All rights reserved.
from typing import TYPE_CHECKING, Any, TypedDict, cast

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

# from mmengine.dist import dist
from pydantic import BaseModel
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.nn.functional import all_reduce

from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.data_proto.utils import pad_to_multiple_of, split_for_sequence_parallel


class LossContextInputItem(TypedDict):
    seq_ctx: SequenceContext
    shift_labels: torch.LongTensor
    advantage: torch.Tensor
    old_logprobs: torch.Tensor  # old_logprobs in train worker's input is None and will be set afterwards


if TYPE_CHECKING:
    from xtuner.v1.rl.grpo.engine import EngineInputItem


def cal_global_grad_tokens(labels: list[torch.Tensor], sp_mesh: DeviceMesh | None = None):
    # calculate global token number which is used for loss scaling
    assert len(labels) > 0, "labels should not be empty"
    rank_grad_tokens = torch.tensor(0, dtype=labels[0].dtype, device=labels[0].device)
    for label in labels:
        rank_grad_tokens += (label >= 0).sum()
    dist.all_reduce(rank_grad_tokens)
    if sp_mesh:
        # data in different sp ranks are replicated
        global_grad_tokens = rank_grad_tokens / sp_mesh.size()
    else:
        global_grad_tokens = rank_grad_tokens
    return global_grad_tokens


class GRPOForwardItem(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    old_logprobs: torch.Tensor
    shift_labels: torch.Tensor
    advantage: torch.Tensor
    loss_weight: torch.Tensor
    global_loss_weight_sum: torch.Tensor


class GRPOLossContext(BaseModel):
    cliprange_low: float
    cliprange_high: float
    ignore_idx: int = -100
    forward_item: GRPOForwardItem | None = None

    def build_loss_fn(self):
        return GRPOLoss(ctx=self)

    def _sp_split(
        self,
        tensor,
        sp_mesh: DeviceMesh,
        split_dim: int,
        padding_value: Any,
    ):
        tensor = pad_to_multiple_of(tensor, padding_value, sp_mesh.size(), split_dim)
        tensor = split_for_sequence_parallel(tensor, dim=split_dim, sp_mesh=sp_mesh)
        return tensor

    @torch.no_grad()
    def _compute_loss_weight(self, label_list: list[torch.Tensor]):
        loss_weight_list = []
        for shift_labels in label_list:
            loss_weight = torch.ones_like(shift_labels, dtype=torch.float32)
            loss_weight[shift_labels == self.ignore_idx] = 0.0
            loss_weight_list.append(loss_weight)
        return loss_weight_list

    @torch.no_grad()
    def _compute_global_loss_weight_sum(self, loss_weights: list[torch.Tensor], dp_mesh: DeviceMesh | None = None):
        global_loss_weight_sum = sum([loss_weight.sum() for loss_weight in loss_weights])
        global_loss_weight_sum = all_reduce(
            global_loss_weight_sum, op=dist.ReduceOp.SUM, group=dp_mesh.get_group() if dp_mesh else dist.group.WORLD
        )
        return global_loss_weight_sum

    def build_list_ctx(
        self,
        data_batch: list[LossContextInputItem],
        data_mesh: DeviceMesh | None = None,
        device=None,
    ) -> list["EngineInputItem"]:
        # Do loss calibration among dp, sp and grad accumulation:
        # Suppose we have sp = 2, grad acc = 2
        #                     rank0         rank1
        # iter0 loss        l00, l01      l02, l03
        #       loss weight w00, w01      w02, w03
        # iter1 loss        l10, l11      l12, l13
        #       loss weight w10, w11      w12, w13
        # There are 2 steps to compute the calibrated loss:
        # 1. Compute the global loss weight sum among dp, sp and grad accumulation:
        #    global_loss_weight_sum = all_reduce(sum([loss_weight.sum() for loss_weight in loss_weights_grad_acc]), op=dist.ReduceOp.SUM, group=world)
        #                           = (w00 + w01 + w02 + w03 + w10 + w11 + w12 + w13)
        # 2. Compute the iter loss, take rank0 iter0 as an example:
        #    a. loss_{rank0iter0} = (l00 * w00 + l01 * w01)
        #    b. loss_{rank0iter0} = all_reduce_autograd(loss_{rank0iter0}, op=dist.ReduceOp.SUM, group=world)
        #                         = (l00 * w00 + l01 * w01 + l02 * w02 + l03 * w03)
        #    c. loss_{rank0iter0} = loss_{rank0iter0} / global_loss_weight_sum
        #                         = (l00 * w00 + l01 * w01 + l02 * w02 + l03 * w03) / (w00 + w01 + w02 + w03 + w10 + w11 + w12 + w13)
        # 3. Compute the step loss:
        #    step_loss = loss_{rank0iter0} + loss_{rank0iter1}
        #              = (l00 * w00 + l01 * w01 + l02 * w02 + l03 * w03 + l10 * w10 + l11 * w11 + l12 * w12 + l13 * w13) /
        #                (w00 + w01 + w02 + w03 + w10 + w11 + w12 + w13)
        #    It's equivalent to loss calculation in sp1, dp1 and grad acc 1.
        if data_mesh is None:
            dp_mesh = None
            sp_mesh = None
        else:
            dp_mesh = data_mesh["dp"]
            # TODO: 需要判断 sp_mesh 是否和 seq_ctx 里面是否一致，不一致要么报错或者赋予 sp？
            sp_mesh = data_mesh["sp"]

        if device is None:
            device = data_batch[0]["seq_ctx"].device

        label_list = [data["shift_labels"] for data in data_batch]

        loss_weight_list = self._compute_loss_weight(label_list)  # type: ignore
        global_loss_weight_sum = self._compute_global_loss_weight_sum(
            loss_weight_list,
            dp_mesh=dp_mesh,
        )

        loss_ctx_list = []
        for data, loss_weight in zip(data_batch, loss_weight_list):
            advantage = data["advantage"]  # sequence level advantage
            # cu_seq_lens_q is not affected by sequence parallel
            cu_seq_lens_q = data["seq_ctx"].cu_seq_lens_q
            num_tokens = cu_seq_lens_q[1:] - cu_seq_lens_q[:-1]
            advantage = torch.repeat_interleave(advantage, num_tokens, dim=1)

            shift_labels = data["shift_labels"]
            if sp_mesh is not None:
                # seq_ctx has been split for sequence parallel
                # so old_logprobs is computed by splited seq_ctx
                shift_labels = self._sp_split(
                    shift_labels, sp_mesh=sp_mesh, split_dim=1, padding_value=self.ignore_idx
                )
                advantage = self._sp_split(advantage, sp_mesh=sp_mesh, split_dim=1, padding_value=0)
                loss_weight = self._sp_split(loss_weight, sp_mesh=sp_mesh, split_dim=1, padding_value=0.0)

            forward_item = GRPOForwardItem(
                old_logprobs=data["old_logprobs"],
                shift_labels=shift_labels,
                advantage=advantage,
                loss_weight=loss_weight,
                global_loss_weight_sum=global_loss_weight_sum,
            )
            loss_ctx = self.__class__(
                cliprange_low=self.cliprange_low,
                cliprange_high=self.cliprange_high,
                ignore_idx=self.ignore_idx,
                forward_item=forward_item,
            )
            loss_ctx_list.append(loss_ctx)

        seq_ctx_list = [data["seq_ctx"] for data in data_batch]
        ret_data_batch = [
            {"seq_ctx": seq_ctx, "loss_ctx": loss_ctx} for seq_ctx, loss_ctx in zip(seq_ctx_list, loss_ctx_list)
        ]
        return cast(list["EngineInputItem"], ret_data_batch)

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


class GRPOLoss(nn.Module):
    def __init__(self, ctx: GRPOLossContext) -> None:
        super().__init__()
        self.cliprange_low = ctx.cliprange_low
        self.cliprange_high = ctx.cliprange_high

    def _gather_logprobs(self, shifted_logits, shifted_labels):
        shift_logprobs = F.log_softmax(shifted_logits, dim=-1)
        shift_logprobs = shift_logprobs.gather(dim=-1, index=shifted_labels.clip(min=0).unsqueeze(-1)).squeeze(-1)
        return shift_logprobs

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        forward_item: GRPOForwardItem,
        head_bias: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = F.linear(hidden_states, head_weight, head_bias)
        old_logprobs = forward_item.old_logprobs
        advantage = forward_item.advantage
        loss_weight = forward_item.loss_weight
        shift_labels = forward_item.shift_labels
        global_loss_weight_sum = forward_item.global_loss_weight_sum
        packed_shift_logprobs = self._gather_logprobs(logits, shift_labels.clip(0))

        ratio = (packed_shift_logprobs - old_logprobs.detach()).exp()
        loss1 = -ratio * advantage
        loss2 = -ratio.clamp(1 - self.cliprange_low, 1 + self.cliprange_high) * advantage
        loss_max = torch.max(loss1, loss2)
        # 2.a
        loss = (loss_max * loss_weight.to(loss_max.dtype)).sum()
        print(f"loss_weight.sum(): {loss_weight.sum()}")
        print(f"2a loss: {loss}")
        # 2.b
        loss = all_reduce(loss, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
        print(f"2b loss: {loss}")
        # 2.c
        loss = loss / global_loss_weight_sum.to(loss.dtype)
        print(f"2c loss: {loss}")
        print(loss)
        return loss, logits
