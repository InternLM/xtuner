# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from typing import Generic, Literal, TypeVar

import torch
import torch.distributed as dist
import torch.nn as nn
from pydantic import BaseModel, ConfigDict
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.nn.functional import all_reduce

from .chunk_loss import ChunkLoss


# Do loss calibration among dp, sp and grad accumulation:
# Suppose we have sp = 2, grad acc = 2
#                             rank0         rank1
# iter0 loss                 l00, l01      l02, l03
#       loss weight          w00, w01      w02, w03
#       loss mask (0 or 1)   m00, m01      m02, m03
# iter1 loss                 l10, l11      l12, l13
#       loss weight          w10, w11      w12, w13
#       loss mask (0 or 1)   m10, m11      m12, m13
# There are 2 steps to compute the calibrated loss:
# 1. Compute the global loss mask sum among dp, sp and grad accumulation:
#    global_loss_mask_sum = all_reduce(sum([loss_mask.sum() for loss_mask in loss_masks_grad_acc]), op=dist.ReduceOp.SUM, group=world)
#                           = (m00 + m01 + m02 + m03 + m10 + m11 + m12 + m13)
# 2. Compute the iter loss, take rank0 iter0 as an example:
#    a. loss_{rank0iter0} = (l00 * w00 * m00 + l01 * w01 * m01)
#    b. loss_{rank0iter0} = loss_{rank0iter0} / global_loss_mask_sum
#                         = (l00 * w00 * m00 + l01 * w01 * m01) /
#                           (m00 + m01 + m02 + m03 + m10 + m11 + m12 + m13)
#    c. loss_{rank0iter0} = all_reduce_autograd(loss_{rank0iter0}, op=dist.ReduceOp.SUM, group=world)
#                         = (l00 * w00 * m00 + l01 * w01 * m01 + l02 * w02 * m02 + l03 * w03 * m03) /
#                           (m00 + m01 + m02 + m03 + m10 + m11 + m12 + m13)
# 3. Compute the step loss:
#    step_loss = loss_{rank0iter0} + loss_{rank0iter1}
#              = (l00 * w00 * m00 + l01 * w01 * m01 + l02 * w02 * m02 + l03 * w03 * m03 +
#                 l10 * w10 * m10 + l11 * w11 * m11 + l12 * w12 * m12 + l13 * w13 * m13) /
#                (m00 + m01 + m02 + m03 + m10 + m11 + m12 + m13)
#    It's equivalent to loss calculation in sp1, dp1 and grad acc 1.


class BaseLossKwargs(BaseModel):
    """Everything needed to compute the loss."""

    model_config = ConfigDict(title="RL loss keyword arguments", extra="allow", arbitrary_types_allowed=True)
    shifted_labels: torch.Tensor

    def chunk(self, chunk_size) -> list["BaseLossKwargs"]:
        tensor_fields: dict[str, tuple[torch.Tensor, ...]] = {}
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, torch.Tensor):
                tensor_fields[field_name] = torch.split(field_value, chunk_size, dim=1)

        assert len(tensor_fields) > 0, "At least one field should be a tensor to chunk."

        num_chunks = len(next(iter(tensor_fields.values())))
        chunks = []
        for i in range(num_chunks):
            chunk_dict = {}
            for field_name, splits in tensor_fields.items():
                chunk_dict[field_name] = splits[i]
            chunks.append(type(self)(**chunk_dict))
        return chunks


class BaseLossConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    ignore_idx: int = -100
    mode: Literal["eager", "chunk"] = "eager"
    chunk_size: int | None = None

    @property
    def loss_ctx_cls(self) -> type["BaseLossContext"]:
        raise NotImplementedError


LossContextInputItem = TypeVar("LossContextInputItem")


class BaseLossContext(nn.Module, ABC, Generic[LossContextInputItem]):
    def __init__(self, loss_cfg: BaseLossConfig, loss_kwargs: BaseLossKwargs):
        super().__init__()
        self.loss_cfg = loss_cfg
        self.loss_kwargs = loss_kwargs

    @classmethod
    @abstractmethod
    def build_batches_loss_kwargs(
        cls,
        data_batches: list[LossContextInputItem],
        loss_cfg: BaseLossConfig,
        # The following two parameters need to be passed in only when sp is enabled
        # and calculating loss_kwargs requires the complete shifted_labels.
        # (For example, the sample-wise loss)
        cu_seq_lens_list: list[torch.Tensor] | None = None,
        sp_mesh: DeviceMesh | None = None,
    ) -> list[BaseLossKwargs]: ...

    @abstractmethod
    def loss_fn(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None,
        loss_kwargs: BaseLossKwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Step 2.a and 2.b in the loss calculation."""
        ...

    def eager_mode(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None,
        loss_kwargs: BaseLossKwargs,
    ):
        return self.loss_fn(hidden_states, head_weight, head_bias, loss_kwargs)

    def chunk_mode(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None,
        loss_kwargs: BaseLossKwargs,
    ):
        assert self.loss_cfg.chunk_size is not None, "chunk_size must be set in chunk mode"

        chunks = loss_kwargs.chunk(self.loss_cfg.chunk_size)
        loss = ChunkLoss.apply(hidden_states, head_weight, head_bias, self.loss_fn, chunks, self.loss_cfg.chunk_size)
        return loss, None

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert self.loss_kwargs is not None, "loss_kwargs must be set before calling forward"
        if head_bias is not None:
            raise NotImplementedError("RL Loss does not support head_bias yet.")

        if self.loss_cfg.mode == "eager":
            loss, logits = self.eager_mode(hidden_states, head_weight, head_bias, self.loss_kwargs)
        else:
            loss, logits = self.chunk_mode(hidden_states, head_weight, head_bias, self.loss_kwargs)

        # Step 2.c in the loss calculation
        if dist.is_initialized():
            loss = all_reduce(loss, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
        return loss, logits
