import enum
from typing import Dict, Literal

import torch


class ScalingGranularity(enum.Enum):
    """Defines the granularity of scaling strategies for casting to float8."""

    # use one scale for each 1x128 tile
    TILEWISE = "tilewise"
    # use one scale for each 128x128 block
    BLOCKWISE = "blockwise"
    # use one scale for the whole tensor
    TENSORWISE = "tensorwise"


class Float8Tensor(torch.Tensor):
    _data: torch.Tensor
    _scale: torch.Tensor
    _orig_dtype: torch.dtype
    _scaling_granularity: Literal[
        ScalingGranularity.TILEWISE, ScalingGranularity.BLOCKWISE, ScalingGranularity.TENSORWISE
    ]
    _group_size: int  # -1 for tensorwise
    __slots__ = [
        "_data",
        "_scale",
        "_orig_dtype",
        "_scaling_granularity",
        "_group_size",
    ]

    def __new__(
        cls,
        data: torch.Tensor,
        scale: torch.Tensor,
        orig_dtype: torch.dtype,
        scaling_granularity: Literal[
            ScalingGranularity.TILEWISE, ScalingGranularity.BLOCKWISE, ScalingGranularity.TENSORWISE
        ],
        group_size: int = 128,
    ):
        self = torch.Tensor._make_wrapper_subclass(  # type: ignore
            cls,
            data.size(),
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            dtype=orig_dtype,
            layout=data.layout,
            requires_grad=data.requires_grad,
            device=data.device,
        )
        self._data = data
        self._scale = scale
        self._orig_dtype = orig_dtype
        self._scaling_granularity = scaling_granularity
        assert scaling_granularity in (
            ScalingGranularity.TILEWISE,
            ScalingGranularity.BLOCKWISE,
            ScalingGranularity.TENSORWISE,
        )
        if group_size != 128:
            raise NotImplementedError("group_size != 128 is not supported yet.")
        self._group_size = group_size

        return self

    def __repr__(self):
        return (
            f"Float8Tensor(\n\t dtype={self._data.dtype}, \n\t data={self._data}, \n\t scale={self._scale}, "
            f"\n\t scaling_granularity={self._scaling_granularity}\n\t group_size={self._group_size}\n)"
        )

    def __tensor_flatten__(self):
        ctx = {
            "_orig_dtype": self._orig_dtype,
            "_scaling_granularity": self._scaling_granularity,
            "_group_size": self._group_size,
        }
        return ["_data", "_scale"], ctx

    @staticmethod
    def __tensor_unflatten__(inner_tensors: Dict, metadata, outer_size, outer_stride):
        assert len(inner_tensors) == 2
        return Float8Tensor(
            inner_tensors["_data"],
            inner_tensors["_scale"],
            metadata["_orig_dtype"],
            metadata["_scaling_granularity"],
            metadata["_group_size"],
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        # 1. tracing through __torch_function__ logic is not supported yet in
        # PT2.0, so we explicitly disallow it here for callsites from user code.
        # 2. We do need to handle a couple of ops in order for
        # TorchDynamo tracing to succeed.

        # Lazy import to avoid circular dependency
        from .float8_ops import FLOAT8_OPS_TABLE

        # All ops in the FLOAT8_OPS_TABLE expect Float8Tensor as inputs
        # And don't support mixed tensor subclasses. This will trigger the handler for
        # the next type in the dispatch list
        def allowed_subclasses(type):
            return (
                issubclass(cls, type)
                or issubclass(torch._subclasses.fake_tensor.FakeTensor, type)
                or issubclass(torch._subclasses.functional_tensor.FunctionalTensor, type)
            )

        if not all(allowed_subclasses(t) for t in types):
            return NotImplemented

        if func in FLOAT8_OPS_TABLE:
            return FLOAT8_OPS_TABLE[func](func, args, kwargs)
        raise NotImplementedError(f"attempting to run {func}, this is not supported")

    # Do not force the Float8Tensor type on the returned tensor
    __torch_function__ = torch._C._disabled_torch_function_impl  # type: ignore
