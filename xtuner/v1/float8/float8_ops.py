import math
from typing import Any, Dict

import torch

from xtuner.v1.float8.float8_tensor import Float8Tensor, ScalingGranularity


aten = torch.ops.aten
c10d_functional = torch.ops.c10d_functional
_c10d_functional = torch.ops._c10d_functional
FLOAT8_OPS_TABLE: Dict[Any, Any] = {}


def implements(aten_ops):
    """Register aten ops to the float8 op table."""

    def decorator(func):
        for op in aten_ops:
            FLOAT8_OPS_TABLE[op] = func
        return func

    return decorator


@implements(
    [
        aten._unsafe_view.default,
        aten.clone.default,
        aten.slice.Tensor,
        aten.fill_.Scalar,
        aten.reshape.default,
    ]
)
def float8_desugar_op(aten_op, args, kwargs=None):
    assert args[0]._scaling_granularity == ScalingGranularity.TENSORWISE, (
        f"{aten_op} with {args[0]._scaling_granularity} scaling granularity is not supported. "
    )
    new_data = aten_op(args[0]._data, *args[1:], **kwargs)
    return Float8Tensor(
        new_data,
        args[0]._scale,
        args[0]._orig_dtype,
        args[0]._linear_mm_config,
        args[0]._gemm_input_role,
    )


@implements([aten.as_strided.default])
def float8_as_strided(aten_op, args, kwargs=None):
    if args[0]._scaling_granularity == ScalingGranularity.TENSORWISE:
        new_data = aten_op(args[0]._data, *args[1:], **kwargs)
        return Float8Tensor(
            new_data,
            args[0]._scale,
            args[0]._orig_dtype,
            args[0]._scaling_granularity,
            args[0]._group_size,
        )
    elif args[0]._scaling_granularity == ScalingGranularity.BLOCKWISE:
        # Used in fsdp2
        assert args[0]._data.is_contiguous() and args[0]._scale.is_contiguous(), (
            f"{aten_op} with {args[0]._scaling_granularity} needs contiguous tensors. "
        )
        assert args[0]._data.ndim == 2 and args[0]._scale.ndim == 2, (
            f"{aten_op} with {args[0]._scaling_granularity} needs 2D tensors. "
        )
        group_size = args[0]._group_size
        new_data = aten_op(args[0]._data, *args[1:], **kwargs)
        keywords = dict(size=None, stride=None, storage_offset=None)
        for key, arg in zip(keywords.keys(), args[1:]):
            if key in ["size", "stride"] and isinstance(arg, int):
                arg = (arg,)
            keywords[key] = arg
        keywords.update(kwargs or {})
        assert keywords["storage_offset"] == 0, (
            f"{aten_op} with {args[0]._scaling_granularity} does not support storage_offset != 0. "
        )
        keywords["size"] = tuple(math.ceil(size / group_size) for size in keywords["size"])
        keywords["stride"] = tuple(math.ceil(stride / group_size) for stride in keywords["stride"])
        keywords["storage_offset"] = 0
        new_scale = aten_op(args[0]._scale, **keywords)
        return Float8Tensor(
            new_data,
            new_scale,
            args[0]._orig_dtype,
            args[0]._scaling_granularity,
            args[0]._group_size,
        )
    else:
        # Currently, we do not have to apply as_strided for tilewise scaling granularity.
        raise NotImplementedError(
            f"{aten_op} with {args[0]._scaling_granularity} scaling granularity is not supported. "
        )


@implements([aten.view.default])
def float8_view(aten_op, args, kwargs=None):
    if args[0]._scaling_granularity == ScalingGranularity.TENSORWISE:
        new_data = aten_op(args[0]._data, *args[1:])
        return Float8Tensor(
            new_data,
            args[0]._scale,
            args[0]._orig_dtype,
            args[0]._scaling_granularity,
            args[0]._group_size,
        )
    elif args[0]._scaling_granularity == ScalingGranularity.BLOCKWISE:
        group_size = args[0]._group_size
        for shape_i in args[0]._data.shape:
            assert shape_i % group_size == 0, (
                f"{aten_op} with {args[0]._scaling_granularity} requires the shape to be divisible by group_size. "
            )
        size = args[1]
        for shape_i in size[-2:]:
            assert isinstance(shape_i, int) and shape_i % group_size == 0, (
                f"{aten_op} with {args[0]._scaling_granularity} requires the new shape to be divisible by group_size. Got {size}."
            )
        new_data = aten_op(args[0]._data, size)
        new_scale_shape = (*new_data.shape[:-2], new_data.shape[-2] // group_size, new_data.shape[-1] // group_size)
        new_scale = args[0]._scale.view(new_scale_shape)
        return Float8Tensor(
            new_data,
            new_scale,
            args[0]._orig_dtype,
            args[0]._scaling_granularity,
            args[0]._group_size,
        )
    elif args[0]._scaling_granularity == ScalingGranularity.TILEWISE:
        group_size = args[0]._group_size
        assert args[0]._data.shape[-1] % group_size == 0, (
            f"{aten_op} with {args[0]._scaling_granularity} requires the last dimension to be divisible by group_size. "
        )
        size = args[1]
        assert size[-1] % group_size == 0, (
            f"{aten_op} with {args[0]._scaling_granularity} requires the new shape to be divisible by group_size. Got {size}."
        )
        new_data = aten_op(args[0]._data, size)
        new_scale_shape = (*new_data.shape[:-1], new_data.shape[-1] // group_size)
        new_scale = args[0]._scale.view(new_scale_shape)
        return Float8Tensor(
            new_data,
            new_scale,
            args[0]._orig_dtype,
            args[0]._scaling_granularity,
            args[0]._group_size,
        )
    else:
        raise NotImplementedError(
            f"{aten_op} with {args[0]._scaling_granularity} scaling granularity is not supported. "
        )


@implements(
    [
        aten.detach.default,
    ]
)
def float8_desugar_data_and_scale_op(aten_op, args, kwargs=None):
    new_data = aten_op(args[0]._data, *args[1:], **kwargs)
    new_scale = aten_op(args[0]._scale, *args[1:], **kwargs)
    return Float8Tensor(
        new_data,
        new_scale,
        args[0]._orig_dtype,
        args[0]._scaling_granularity,
        args[0]._group_size,
    )
