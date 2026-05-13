# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from typing import Annotated, Any, Generic, Literal, TypeVar

import torch
import torch.nn as nn
from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict
from torch.distributed.device_mesh import DeviceMesh


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
    """Everything needed to compute the loss.

    Subclasses should implement sp_split() and to() methods if they contain tensors that need to be split across
    sequence parallel mesh or moved to device.
    """

    model_config = ConfigDict(title="loss keyword arguments", extra="forbid", arbitrary_types_allowed=True)

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
    @abstractmethod
    def loss_ctx_cls(self) -> type["BaseLossContext"]:
        raise NotImplementedError

    # TODO: private property maybe not a good idea
    @property
    @abstractmethod
    def _loss_kwargs_cls(self) -> type["BaseLossKwargs"]:
        raise NotImplementedError

    @abstractmethod
    def build(
        self,
        data: dict,
        sp_mesh: "DeviceMesh | None" = None,
    ) -> "BaseLossContext | None":
        """Build loss context from data dict.

        Subclasses should extract required fields from data dict and construct loss_kwargs.

        Args:
            data (dict): Data dict containing all possible loss-related fields.
                Different loss configs extract different fields as needed.
            sp_mesh (DeviceMesh | None): Sequence parallel mesh.

        Returns:
            BaseLossContext: Built loss context.
        """
        ...


# NOTE: Self type for BaseLossContext subclasses (F-bounded polymorphism)
_BaseLossContextT = TypeVar("_BaseLossContextT", bound="BaseLossContext")
LossContextInputItem = TypeVar("LossContextInputItem")


class BaseLossContext(nn.Module, ABC, Generic[LossContextInputItem]):
    def __init__(self, loss_cfg: BaseLossConfig, loss_kwargs: BaseLossKwargs):
        # LossContext需要负责几个功能：
        # 1. sequence parallel, 借助LossKwargs.sp_split 实现
        # 2. batch内的loss全局校准, 借助 LossContext.build_batches 实现
        # 3. chunk loss计算，借助 LossKwargs.chunk 实现
        # 其中，因为LossContext负责batch内的loss 全局校准，提供 build_batches接口，
        # 在build_batches中统计batch内的loss 全局校准参数（例如global_grad_tokens）
        super().__init__()
        self.loss_cfg = loss_cfg
        self.loss_kwargs = loss_kwargs
        self._batch_size = 1

    @staticmethod
    def build_batches(loss_ctx_list: list[_BaseLossContextT], *args, **kwargs) -> list[_BaseLossContextT]:
        for ctx in loss_ctx_list:
            ctx._batch_size = len(loss_ctx_list)
        return loss_ctx_list

    @classmethod
    def cat(cls: type[_BaseLossContextT], chunks: list[_BaseLossContextT]) -> _BaseLossContextT:
        assert len(chunks) > 0, "chunks must not be empty."

        first = chunks[0]
        loss_cfg = first.loss_cfg
        loss_kwargs_chunks = [c.loss_kwargs for c in chunks]
        loss_kwargs = type(first.loss_kwargs).cat(loss_kwargs_chunks)

        return cls(loss_cfg, loss_kwargs)

    @property
    def batch_size(self) -> int:
        return self._batch_size
