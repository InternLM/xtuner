# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from typing import Annotated, Any, Literal, Self, TypeVar

import torch
import torch.distributed as dist
import torch.nn as nn
from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.nn.functional import all_reduce

from xtuner.v1.loss.utils import sp_split

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
# There are 3 steps to compute the calibrated loss:
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

    model_config = ConfigDict(title="loss keyword arguments", extra="forbid", arbitrary_types_allowed=True)
    shifted_labels: torch.Tensor

    def sp_split(self, sp_mesh: DeviceMesh) -> Self:
        self.shifted_labels = sp_split(self.shifted_labels, sp_mesh=sp_mesh, split_dim=1, padding_value=-100)
        return self

    def to(self, device: torch.device | str) -> Self:
        self.shifted_labels = self.shifted_labels.to(device)
        return self

    def chunk(self, chunk_size) -> list["BaseLossKwargs"]:
        tensor_fields: dict[str, tuple[torch.Tensor, ...]] = {}
        non_tensor_fields: dict[str, Any] = {}
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, torch.Tensor) and field_value.dim() > 0:
                tensor_fields[field_name] = torch.split(field_value, chunk_size, dim=1)
            else:  # scalar or 0-dim tensor such as global_grad_tokens
                non_tensor_fields[field_name] = field_value

        assert len(tensor_fields) > 0, "At least one field should be a tensor to chunk."

        num_chunks = len(next(iter(tensor_fields.values())))
        chunks = []
        for i in range(num_chunks):
            chunk_dict = {}
            for field_name, splits in tensor_fields.items():
                chunk_dict[field_name] = splits[i]
            for field_name, field_value in non_tensor_fields.items():
                chunk_dict[field_name] = field_value
            chunks.append(type(self)(**chunk_dict))
        return chunks

    @classmethod
    def cat(cls, chunks: list["BaseLossKwargs"]) -> "BaseLossKwargs":
        assert len(chunks) > 0, "chunks must not be empty."

        # Collect all tensor field names (based on chunk[0]'s fields; pydantic extra=forbid also requires fields to be consistent)
        first = chunks[0]
        tensor_field_names: list[str] = []
        non_tensor_fields: dict[str, Any] = {}
        for field_name, field_value in first.__dict__.items():
            if isinstance(field_value, torch.Tensor) and field_value.dim() > 0:
                tensor_field_names.append(field_name)
            else:
                non_tensor_fields[field_name] = field_value

        assert len(tensor_field_names) > 0, "At least one field should be a tensor to cat."

        cat_dict: dict[str, torch.Tensor] = {}
        for field_name in tensor_field_names:
            tensors = [getattr(c, field_name) for c in chunks]
            cat_dict[field_name] = torch.cat(tensors, dim=1)

        cat_dict.update(non_tensor_fields)

        return cls(**cat_dict)


class BaseLossConfig(BaseModel):
    model_config = ConfigDict(title="BaseLossConfig", extra="forbid", arbitrary_types_allowed=True)
    ignore_idx: Annotated[int, Parameter(help="ignore index for loss calculation")] = -100
    mode: Annotated[Literal["eager", "chunk"], Parameter(help="loss calculation mode")] = "eager"
    chunk_size: Annotated[int | None, Parameter(help="chunk size when mode is chunk")] = 1024

    @property
    def loss_ctx_cls(self) -> type["BaseLossContext"]:
        raise NotImplementedError

    @property
    def _loss_kwargs_cls(self) -> type["BaseLossKwargs"]:
        raise NotImplementedError

    def build(self, *args, **kwargs) -> "BaseLossContext":
        raise NotImplementedError


# NOTE: Self type for BaseLossContext subclasses (F-bounded polymorphism)
_BaseLossContextT = TypeVar("_BaseLossContextT", bound="BaseLossContext")


class BaseLossContext(nn.Module, ABC):
    def __init__(self, loss_cfg: BaseLossConfig, loss_kwargs: BaseLossKwargs, sp_mesh: DeviceMesh | None = None):
        # LossContext需要负责几个功能：
        # 1. sequence parallel, 借助LossKwargs.sp_split 实现
        # 2. batch内的loss全局校准, 借助 LossContext.build_batches 实现
        # 3. chunk loss计算，借助 LossKwargs.chunk 实现
        # 其中，因为LossContext负责batch内的loss 全局校准，提供 build_batches接口，
        # 在build_batches中统计batch内的loss 全局校准参数（例如global_grad_tokens）
        super().__init__()
        self.loss_cfg = loss_cfg
        self.loss_kwargs = loss_kwargs
        self.sp_mesh = sp_mesh

    @staticmethod
    @abstractmethod
    def build_batches(loss_ctx_list: list[_BaseLossContextT], *args, **kwargs) -> list[_BaseLossContextT]: ...

    @abstractmethod
    def loss_fn(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None,
        loss_kwargs: BaseLossKwargs,
        enable_chunk_linear: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor | None, dict[str, Any]]]:
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
        loss, extra_info = ChunkLoss.apply(
            hidden_states, head_weight, head_bias, self.loss_fn, chunks, self.loss_cfg.chunk_size
        )
        return loss, (None, extra_info)

    def _run_mode(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None,
        loss_kwargs: BaseLossKwargs,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor | None, dict[str, Any]]]:
        if self.loss_cfg.mode == "eager":
            return self.eager_mode(hidden_states, head_weight, head_bias, loss_kwargs)
        else:
            return self.chunk_mode(hidden_states, head_weight, head_bias, loss_kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor | None, dict[str, Any]]]:
        assert self.loss_kwargs is not None, "loss_kwargs must be set before calling forward"
        if head_bias is not None:
            raise NotImplementedError("Loss does not support head_bias yet.")

        loss, (logits, extra_info) = self._run_mode(hidden_states, head_weight, head_bias, self.loss_kwargs)

        extra_info["local_base_loss"] = loss.detach().clone()

        # Step 2.c in the loss calculation: reduce the loss over all ranks using all_reduce with autograd support
        if dist.is_initialized():
            loss = all_reduce(loss, op=dist.ReduceOp.SUM, group=dist.group.WORLD)

        return loss, (logits, extra_info)

    @classmethod
    def cat(cls: type[_BaseLossContextT], chunks: list[_BaseLossContextT]) -> _BaseLossContextT:
        assert len(chunks) > 0, "chunks must not be empty."

        first = chunks[0]
        loss_cfg = first.loss_cfg
        loss_kwargs_chunks = [c.loss_kwargs for c in chunks]
        loss_kwargs = type(first.loss_kwargs).cat(loss_kwargs_chunks)

        return cls(loss_cfg, loss_kwargs)
