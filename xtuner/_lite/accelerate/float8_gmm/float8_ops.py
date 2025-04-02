# Copyright (c) OpenMMLab. All rights reserved.
# Modified from https://github.com/pytorch/ao/blob/v0.8.0/torchao/float8/float8_ops.py
from typing import Any, Dict, Tuple

import torch
from torch.utils._pytree import tree_map

from xtuner._lite.accelerate.float8_gmm.float8_python_api import addmm_float8_unwrapped
from xtuner._lite.accelerate.float8_gmm.float8_tensor import (
    Float8Tensor,
    choose_scaled_mm_config,
)
from xtuner._lite.accelerate.float8_gmm.float8_utils import (
    is_row_major,
    pad_tensor_for_matmul,
)

aten = torch.ops.aten
c10d_functional = torch.ops.c10d_functional
_c10d_functional = torch.ops._c10d_functional
FLOAT8_OPS_TABLE: Dict[Any, Any] = {}


def _assert_tensorwise_scale(aten_op, scale):
    assert (
        # TODO(future PR): figure out why tensorwise scaling can have
        # both rank 0 and rank 1
        len(scale.shape)
        in (0, 1)
    ), f"{aten_op} with axiswise scaling is not supported yet"


def implements(aten_ops):
    """Register aten ops to the float8 op table."""

    def decorator(func):
        for op in aten_ops:
            FLOAT8_OPS_TABLE[op] = func
        return func

    return decorator


@implements(
    [
        # aten.view.default,
        aten._unsafe_view.default,
        # aten.as_strided.default,
        aten.clone.default,
        aten.slice.Tensor,
        aten.fill_.Scalar,
        aten.reshape.default,
    ]
)
def float8_desugar_op(aten_op, args, kwargs=None):
    _assert_tensorwise_scale(aten_op, args[0]._scale)
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
    data = args[0]._data
    assert args[3] == 0
    new_data = aten_op(data, *args[1:], **kwargs)
    return Float8Tensor(
        new_data,
        args[0]._scale,
        args[0]._orig_dtype,
        args[0]._linear_mm_config,
        args[0]._gemm_input_role,
    )


@implements([aten.view.default])
def float8_view(aten_op, args, kwargs=None):
    # 目前只支持 view _data 不 view scale
    if len(args[0]._scale.shape) < 2:
        # tensorwise scaling
        return float8_desugar_op(aten_op, args, kwargs)

    t, new_shape = args[0], args[1]
    new_data = aten_op(t._data, new_shape, **kwargs)
    return Float8Tensor(
        new_data,
        t._scale,
        t._orig_dtype,
        t._linear_mm_config,
        t._gemm_input_role,
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
        args[0]._linear_mm_config,
        args[0]._gemm_input_role,
    )


@implements(
    [
        aten.t.default,
        aten.transpose.int,
    ]
)
def float8_transpose(aten_op, args, kwargs=None):
    new_data = aten_op(args[0]._data, *args[1:], **kwargs)
    if args[0]._scale.ndim > 1:
        new_scale = aten_op(args[0]._scale, *args[1:], **kwargs)
    else:
        new_scale = args[0]._scale

    if aten_op == aten.transpose.int:
        _assert_tensorwise_scale(aten_op, args[0]._scale)

    old_axiswise_dim = args[0]._axiswise_dim
    new_axiswise_dim = old_axiswise_dim
    if old_axiswise_dim is not None:
        if old_axiswise_dim == 0:
            new_axiswise_dim == -1
        else:
            new_axiswise_dim == 0

    return Float8Tensor(
        new_data,
        new_scale,
        args[0]._orig_dtype,
        args[0]._linear_mm_config,
        args[0]._gemm_input_role,
        new_axiswise_dim,
    )


@implements([aten.split.Tensor])
def float8_split(aten_op, args, kwargs=None):
    new_data_tensors = aten_op(args[0]._data, *args[1:], **kwargs)
    _assert_tensorwise_scale(aten_op, args[0]._scale)

    def make_float8(data):
        return Float8Tensor(
            data,
            args[0]._scale,
            args[0]._orig_dtype,
            args[0]._linear_mm_config,
            args[0]._gemm_input_role,
        )

    out = map(make_float8, new_data_tensors)
    return list(out)


# Errors can't `cat_cuda float8 e4m3fn`
@implements([aten.cat.default])
def float8_cat(aten_op, args, kwargs=None):
    chunked_tensors: Tuple[Float8Tensor] = args[0]

    orig_dtype = chunked_tensors[0]._orig_dtype
    scale = chunked_tensors[0]._scale
    mm_config = chunked_tensors[0]._linear_mm_config
    fp8_dtype = chunked_tensors[0]._data.dtype
    gemm_input_role = chunked_tensors[0]._gemm_input_role
    chunk_data = []
    for chunk in chunked_tensors:
        assert isinstance(
            chunk, Float8Tensor
        ), "Expecting all chunks to be of type Float8Tensor"
        assert (
            chunk._orig_dtype == orig_dtype
        ), "Expecting all chunks to be of the same dtype"
        assert (
            chunk._scale is scale
        ), "Expecting all chunks to have thee same scale as a result of a split"
        assert (
            chunk._linear_mm_config is mm_config
        ), "Expecting all chunks to have thee same mm config as a result of a split"
        assert (
            chunk._data.dtype == fp8_dtype
        ), "Expecting all chunks to be of the same dtype as a result of a split"
        assert (
            chunk._gemm_input_role is gemm_input_role
        ), "Expecting all chunks to have the same gemm_input_role as a result of a split"
        _assert_tensorwise_scale(aten_op, chunk._scale)
        chunk_data.append(chunk._data.view(torch.uint8))

    new_data = aten_op(chunk_data, *args[1:], **kwargs)
    new_data = new_data.view(fp8_dtype)
    return Float8Tensor(new_data, scale, orig_dtype, mm_config, gemm_input_role)


@implements([aten.sum.dim_IntList])
def float8_cast_up_op(aten_op, args, kwargs=None):
    """Be careful with this function, this is a "fallback" op that casts the
    output of the op to the original precision. And performs the op.

    We currently need this to support the backward for admmm bias. "addmm" -> out "hp_gradBias" <-"sum" <- "identity"
    <- gradOut <- "hp_gradOut"
    """
    _assert_tensorwise_scale(aten_op, args[0]._scale)

    def unwrap(x):
        if isinstance(x, Float8Tensor):
            return x.to_original_precision()
        return x

    new_args = tree_map(unwrap, args)
    new_kwargs = tree_map(unwrap, kwargs)
    return aten_op(*new_args, **new_kwargs)


def preprocess_addmm(a: Float8Tensor, b: Float8Tensor):
    a_data = a._data
    a_scale = a._scale
    b_data = b._data

    scaled_mm_config = choose_scaled_mm_config(
        a._gemm_input_role,
        a._linear_mm_config,
        b._gemm_input_role,
        b._linear_mm_config,
    )

    if scaled_mm_config.pad_inner_dim:
        assert a._data.size(1) == b._data.size(
            0
        ), f"Inner dims must match for mm, got {a._data.size(1)} and {b._data.size(0)}"
        a_data = pad_tensor_for_matmul(a_data, dims=1)
        b_data = pad_tensor_for_matmul(b_data, dims=0)

    if not is_row_major(a_data.stride()):
        a_data = a_data.contiguous()
    if is_row_major(b_data.stride()):
        b_data = b_data.t().contiguous().t()
    b_scale = b._scale

    # Today, torch._scaled_mm only supports both operands using the
    # same granularity. The code below checks for cases where one
    # operand is scaled axiswise and one tensorwise. If this case is found,
    # we reshape the tensorwise scale to be repeat along the needed axis,
    # so that torch._scaled_mm can call the axiswise-axiswise kernel.
    # Note: using shape/size info does not work with compile here, which is
    # why we are using inferring scaling type from the presence of
    # axiswise_dim.
    if a._axiswise_dim is None and b._axiswise_dim is not None:
        a_scale = a_scale.repeat(a_data.shape[0]).reshape(-1, 1)
    elif a._axiswise_dim is not None and b._axiswise_dim is None:
        b_scale = b_scale.repeat(b_data.shape[1]).reshape(1, -1)

    return a_data, a_scale, b_data, b_scale


@implements([aten.mm.default, aten.matmul.default])
def float8_mm(aten_op, args, kwargs=None):
    a = args[0]
    b = args[1]

    assert isinstance(a, Float8Tensor) and isinstance(
        b, Float8Tensor
    ), "Expecting  both Float8Tensor for mm inputs but found {} and {}".format(
        type(a), type(b)
    )
    a_data, a_scale, b_data, b_scale = preprocess_addmm(a, b)
    output_dtype = a._orig_dtype
    scaled_mm_config = choose_scaled_mm_config(
        a._gemm_input_role,
        a._linear_mm_config,
        b._gemm_input_role,
        b._linear_mm_config,
    )
    if scaled_mm_config.emulate:
        return torch.mm(a._data.float() / a._scale, b._data.float() / b._scale).to(
            output_dtype
        )
    tensor_out = addmm_float8_unwrapped(
        a_data,
        a_scale,
        b_data,
        b_scale,
        output_dtype,
        output_scale=None,
        bias=None,
        use_fast_accum=scaled_mm_config.use_fast_accum,
    )
    return tensor_out


@implements([aten.addmm.default])
def float8_addmm(aten_op, args, kwargs=None):
    assert (
        isinstance(args[0], torch.Tensor)
        and isinstance(args[1], Float8Tensor)
        and isinstance(args[2], Float8Tensor)
    )
    bias = args[0]
    a = args[1]
    b = args[2]
    a_data, a_scale, b_data, b_scale = preprocess_addmm(a, b)
    output_dtype = a._orig_dtype
    assert bias.dtype == output_dtype, "bias dtype must match output dtype"
    scaled_mm_config = choose_scaled_mm_config(
        a._gemm_input_role,
        a._linear_mm_config,
        b._gemm_input_role,
        b._linear_mm_config,
    )
    if scaled_mm_config.emulate:
        out = torch.mm(a._data.float() / a._scale, b._data.float() / b._scale).to(
            output_dtype
        )
        return out + bias
    tensor_out = addmm_float8_unwrapped(
        a_data,
        a_scale,
        b_data,
        b_scale,
        output_dtype,
        output_scale=None,
        bias=bias,
        use_fast_accum=scaled_mm_config.use_fast_accum,
    )
    return tensor_out


@implements([aten.is_same_size.default])
def float8_is_same_size(aten_op, args, kwargs=None):
    _assert_tensorwise_scale(aten_op, args[0]._scale)
    return args[0].shape == args[1].shape


@implements([aten._to_copy.default])
def autocast_to_copy(aten_op, args, kwargs=None):
    """This gets called when running matmul under autocast when the input is a
    Float8Tensor, presenting as a fp32 tensor."""
    _assert_tensorwise_scale(aten_op, args[0]._scale)
    assert isinstance(args[0], Float8Tensor)
    assert (
        len(kwargs) == 1 and "dtype" in kwargs
    ), "Only support dtype kwarg for autocast"
    assert kwargs["dtype"] in {
        torch.float16,
        torch.bfloat16,
    }, "Only support floating point conversion for autocast w/ Float8Tensor"
    return Float8Tensor(
        args[0]._data,
        args[0]._scale,
        kwargs["dtype"],
        args[0]._linear_mm_config,
        args[0]._gemm_input_role,
    )


@implements(
    [
        c10d_functional.all_gather_into_tensor.default,
        _c10d_functional.all_gather_into_tensor.default,
    ]
)
def allgather_fp8(aten_op, args, kwargs=None):
    """Override funcol with FP8 handling."""
    _assert_tensorwise_scale(aten_op, args[0]._scale)
    fp8_input = args[0]
    assert isinstance(
        fp8_input, Float8Tensor
    ), f"expecting a Float8Tensor for allgather but found {type(fp8_input)}"

    fp8_data = fp8_input._data
    fp8_data = fp8_data.contiguous()
    fp8_out = aten_op(fp8_data, *args[1:], **kwargs)
    return Float8Tensor(
        fp8_out,
        fp8_input._scale,
        fp8_input._orig_dtype,
        fp8_input._linear_mm_config,
        fp8_input._gemm_input_role,
    )


@implements([c10d_functional.wait_tensor.default, _c10d_functional.wait_tensor.default])
def wait_tensor_fp8(aten_op, args, kwargs=None):
    _assert_tensorwise_scale(aten_op, args[0]._scale)
    fp8_input = args[0]
    assert isinstance(fp8_input, Float8Tensor)

    fp8_data = fp8_input._data
    fp8_out = aten_op(fp8_data, *args[1:], **kwargs)
    return Float8Tensor(
        fp8_out,
        fp8_input._scale,
        fp8_input._orig_dtype,
        fp8_input._linear_mm_config,
        fp8_input._gemm_input_role,
    )


@implements([aten.index_put_.default])
def index_put_fp8(aten_op, args, kwargs=None):
    fp8_self = args[0]
    fp8_values = args[2]
    assert isinstance(fp8_self, Float8Tensor)
    assert isinstance(fp8_values, Float8Tensor)
    _assert_tensorwise_scale(fp8_self, args[0]._scale)
    assert fp8_self._scale == fp8_values._scale
    assert fp8_self.dtype == fp8_values.dtype
    assert fp8_self._orig_dtype == fp8_values._orig_dtype

    fp8_data = fp8_self._data
    fp8_values_data = fp8_values._data
    fp8_out = aten_op(fp8_data, args[1], fp8_values_data, *args[3:], **kwargs)
    return Float8Tensor(
        fp8_out,
        fp8_self._scale,
        fp8_self._orig_dtype,
        fp8_self._linear_mm_config,
        fp8_self._gemm_input_role,
    )


@implements([aten.copy_.default])
def copy_fp8(aten_op, args, kwargs=None):
    # For a copy op with Float8Tensors involved, only the following combinations are allowed:
    # 1. self is a high precision (hp) tensor, src is a Float8Tensor:
    #    in this case src is upcasted and unscaled to go into the hp tensor
    # 2. self and src are Float8Tensors:
    #    the copy is only allowed if all the Float8Tensor properties are equal (a la torch.cat)
    # Every other combination is banned as the semantics are not well defined

    self = args[0]
    src = args[1]

    if not isinstance(self, Float8Tensor) and isinstance(src, Float8Tensor):
        src_hp = src.to_original_precision()
        _assert_tensorwise_scale(aten_op, src._scale)
        return aten_op(self, src_hp, *args[2:], **kwargs)
    elif isinstance(self, Float8Tensor) and isinstance(src, Float8Tensor):
        _assert_tensorwise_scale(aten_op, src._scale)
        assert (
            self._orig_dtype == src._orig_dtype
        ), "Expecting both Float8Tensors to be of the same dtype"
        assert (
            self._scale == src._scale
        ), "Expecting both Float8Tensors to have thee same scale"
        assert (
            self._linear_mm_config == src._linear_mm_config
        ), "Expecting both Float8Tensors to have thee same mm config"
        assert (
            self._data.dtype == src._data.dtype
        ), "Expecting both Float8Tensors to be of the same dtypet"
        assert (
            self._gemm_input_role == src._gemm_input_role
        ), "Expecting both Float8Tensors to have the same gemm_input_role"
        fp8_out = aten_op(self._data, src._data, *args[2:], **kwargs)
        return Float8Tensor(
            fp8_out,
            self._scale,
            self._orig_dtype,
            self._linear_mm_config,
            self._gemm_input_role,
        )
    else:
        raise RuntimeError("Unsupported semantics for copy_ in Float8Tensor")
