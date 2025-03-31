# Copyright (c) OpenMMLab. All rights reserved.
# Modified from https://github.com/pytorch/ao/blob/v0.8.0/torchao/float8/float8_tensor.py
# 1. Support tile-wise Float8Tensor
import enum
from typing import Dict, NamedTuple, Optional

import torch
from torch.distributed._tensor import DTensor

from xtuner._lite.accelerate.float8_gmm.float8_utils import to_fp8_saturated

aten = torch.ops.aten

#
# A note on configuration of float8 logic in a linear
# TODO(future): move all the configs to separate file
# TODO(future): change this to input/weight/grad_output notation,
#   can be separate PR because none of this is user facing
#
# There are three gemms in a forward + backward of a Linear layer:
#
# 1.       input @ weight_t    = output     (forward pass)
# 2. grad_output @ weight      = grad_input (backward pass)
# 3.     input_t @ grad_output = grad_weight (backward pass)
#
# In the formulas above, there are:
# A. six input tensors (input, input_t, weight, weight_t, grad_output, grad_output_t).
#    - Note that grad_output_t is implied because of memory format requirements
#      of float8 gemms
# B. three output tensors (output, grad_input, grad_weight)
#
# We want each input tensor, gemm, and output tensor to be configurable.
# The state of this configuration today is:
#
# i. pairs of input tensors (non-t and t variants) have their scaling
#    configurable via the scaling_type_* arguments to Float8Linear
# ii. each gemm + output is configurable via ScaledMMConfig, which is not user facing
# iii. LinearMMConfig is a container for the three ScaledMMConfig objects needed
#    to configure all three gemms, also not user facing


class ScaledMMConfig(NamedTuple):
    """Configuration for the scaled_mm in the forward and backward pass.

    Attributes:
        emulate (bool): Whether to emulate the matmuls in fp32.
        use_fast_accum (bool): Whether to use the fast-accumulation option for scaled_mm.
        fp8_output (bool): Whether to output the result of the scaled_mm in fp8.
        pad_inner_dim (bool): Whether to pad the inner dimension of a and b with 0s.
                              This is needed for matmuls not aligned to 16.
    """

    emulate: bool = False
    use_fast_accum: bool = False
    fp8_output: bool = False
    pad_inner_dim: bool = False


class LinearMMConfig(NamedTuple):
    """Configuration for different gemm operations in LinearMM.

    This configuration is not user-facing and exists for convenience,
    allowing Float8Tensor to use the right config based on which gemm
    from gemms with outputs `output`, `grad_input`, `grad_weight` is being called.

    Attributes:
        output (ScaledMMConfig): Configuration for the output gemm.
        grad_input (ScaledMMConfig): Configuration for the grad_input gemm.
        grad_weight (ScaledMMConfig): Configuration for the grad_weight gemm.
    """

    output: ScaledMMConfig = ScaledMMConfig(False, True, False, False)
    grad_input: ScaledMMConfig = ScaledMMConfig(False, False, False, False)
    grad_weight: ScaledMMConfig = ScaledMMConfig(False, False, False, False)


class GemmInputRole(enum.Enum):
    """Given a Float8Tensor, the enum below describes the expected role of this
    tensor in the three gemms present in the fw + bw pass of a Linear layer.

    This is used to choose the right config for a float8 gemm when the gemm is performed.
    """

    INPUT = "input"
    WEIGHT = "weight"
    GRAD_OUTPUT = "grad_output"


# choose which scaled_mm_config to use based on gemm inputs
def choose_scaled_mm_config(
    a_role: GemmInputRole,
    a_linear_mm_config: LinearMMConfig,
    b_role: GemmInputRole,
    b_linear_mm_config: LinearMMConfig,
):
    if a_role is GemmInputRole.INPUT and b_role is GemmInputRole.WEIGHT:
        assert (
            a_linear_mm_config.output == b_linear_mm_config.output
        ), f"linear_mm_config.output mismatch: {a_linear_mm_config.output} vs {b_linear_mm_config.output}"
        return a_linear_mm_config.output
    elif a_role is GemmInputRole.GRAD_OUTPUT and b_role is GemmInputRole.WEIGHT:
        assert (
            a_linear_mm_config.grad_input == b_linear_mm_config.grad_input
        ), f"linear_mm_config.grad_input mismatch: {a_linear_mm_config.grad_input} vs {b_linear_mm_config.grad_input}"
        return a_linear_mm_config.grad_input
    elif a_role is GemmInputRole.GRAD_OUTPUT and b_role is GemmInputRole.INPUT:
        assert a_linear_mm_config.grad_weight == b_linear_mm_config.grad_weight, (
            f"linear_mm_config.grad_weight mismatch: "
            f"{a_linear_mm_config.grad_weight} vs {b_linear_mm_config.grad_weight}"
        )
        return a_linear_mm_config.grad_weight
    else:
        raise AssertionError(f"unexpected a_role {a_role} and b_role {b_role}")


@torch._dynamo.allow_in_graph
class _ToFloat8ConstrFunc(torch.autograd.Function):
    """A differentiable conversion to fp8.

    * forward: convert from high precision to float8
    * backward: pass the gradient without changes
    """

    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        scale: torch.Tensor,
        float8_dtype: torch.dtype,
        linear_mm_config: Optional[LinearMMConfig] = None,
        gemm_input_role: Optional[GemmInputRole] = GemmInputRole.INPUT,
        axiswise_dim: Optional[int] = None,
    ):
        """This function will apply the scaling, and then convert to a
        Float8Tensor.

        Note:
        We will call this function with a DTensor subclass. Ideally this would be an aten OP
        that DTensor could overload to ensure proper semantics. There are some technical issues
        with that composing with FakeTensor, so we special case here.

        DTensor Invariant: DTensor must always be the outer most tensor subclass
        """
        # Note: when the line below is compiled with `torch.compile`, `tensor` is automatically
        # upcasted to `float32` to multiply with the scale
        # In order to match numerics between eager and compile, we upcast manually here.
        tensor_scaled = tensor.to(torch.float32) / scale
        bits_fp8 = to_fp8_saturated(tensor_scaled, float8_dtype)

        if isinstance(bits_fp8, DTensor):
            assert isinstance(
                scale, DTensor
            ), "Expected Float8 scale to be a DTensor if bits_fp8 is a DTensor"
            bits_mesh = bits_fp8.device_mesh
            bits_placements = bits_fp8.placements
            local_bits = bits_fp8.to_local()
            local_scale = scale.to_local()
            inner_float8_tensor = Float8Tensor(
                local_bits,
                local_scale,
                tensor.dtype,
                linear_mm_config=linear_mm_config,
                gemm_input_role=gemm_input_role,
                axiswise_dim=axiswise_dim,
            )
            return DTensor.from_local(
                inner_float8_tensor,
                bits_mesh,
                bits_placements,
                run_check=False,
                shape=bits_fp8.size(),
                stride=bits_fp8.stride(),
            )

        return Float8Tensor(
            bits_fp8,
            scale,
            tensor.dtype,
            linear_mm_config=linear_mm_config,
            gemm_input_role=gemm_input_role,
            axiswise_dim=axiswise_dim,
        )

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None, None, None


@torch._dynamo.allow_in_graph
class _FromFloat8ConstrFunc(torch.autograd.Function):
    """A differentiable conversion from fp8.

    * forward: convert from float8 to high precision
    * backward: pass the gradient without changes
    """

    @staticmethod
    def forward(ctx, tensor):
        return tensor._data.to(tensor._orig_dtype) / tensor._scale

    @staticmethod
    def backward(ctx, g):
        return g, None, None


def hp_tensor_and_scale_to_float8(
    hp_tensor: torch.Tensor,
    s: torch.Tensor,
    float8_dtype: torch.dtype,
    linear_mm_config: Optional[LinearMMConfig] = None,
    gemm_input_role: Optional[GemmInputRole] = GemmInputRole.INPUT,
    axiswise_dim: Optional[int] = None,
):
    """Given a high precision tensor `hp_tensor` and a precalculated scale `s`,
    scales `hp_tensor` by `s` and returns a `Float8Tensor` of the result.

    Autograd-aware, the derivative is pass-through.
    DTensor-aware, if the input is a DTensor the output will be DTensor(Float8Tensor).

    Args:
        hp_tensor: the tensor to convert
        s: the scale to use to convert the tensor
        float8_dtype: the float8 dtype to use
        linear_mm_config: Defines the configuration for the scaled_mm for
          the 3 fwd/bwd gemms of linear
        gemm_input_role: Defines the role of this tensor (input, weight or grad_output) in
          the 3 fwd/bwd gemms of linear
        axiswise_dim: for rowwise scaling, contains the axis scaled across
    """
    return _ToFloat8ConstrFunc.apply(
        hp_tensor, s, float8_dtype, linear_mm_config, gemm_input_role, axiswise_dim
    )


class Float8Tensor(torch.Tensor):
    """
    Note: this is **not** a public API and is only intended to be used
    inside of this repository. Please file an issue if you would benefit
    from this being a public API.

    A Python-only Float8 tensor subclass.  Contains:
    * `_data`: the underlying e4m3 or e5m2 data
    * `_scale`: the scale used to scale the original fp32 tensor. We multiply
      by scale to go from fp32 range to fp8 range, and divide by scale to go
      from fp8 range to fp32 range. Scale is guaranteed to have a shape compatible
      with `_data`. For example:
      - if scaling is tensorwise, `_scale` is a scalar tensor
      - if scaling is axiswise and _data.shape is [3, 5], `_scale` could have
        shape [1, 5] or [3, 1]. `axiswise_dim` defines the scaling axis.
      - if scaling is axiswise and _data.shape is [2, 3, 5], `_scale` could have
        shape [1, 1, 5] or [2, 1, 1]. `axiswise_dim` defines the scaling
        axis. Non-one entries which are not the first or last element are not
        supported.
    * `_orig_dtype`: the original dtype of the tensor used to create this
      tensor.
    * `_axiswise_dim`: for axiswise scaling only, contains the axis scales
      across. Only values of 0 or -1 are supported.

    Intended usage of this abstraction:
    1. to bundle raw data + fp8 metadata together for easy passing through
       Python PyTorch systems.
    2. Float8-aware user code can use the private fields on these tensors
       to call into float8 operations.
    3. Float8-agnostic user code can use these tensors as is - they will
       convert to original precision in `__torch_dispatch__`.
    """

    _data: torch.Tensor
    _scale: torch.Tensor
    _orig_dtype: torch.dtype
    _linear_mm_config: LinearMMConfig
    _gemm_input_role: GemmInputRole
    _axiswise_dim: Optional[int]
    __slots__ = [
        "_data",
        "_scale",
        "_orig_dtype",
        "_linear_mm_config",
        "_gemm_input_role",
        "_axiswise_dim",
    ]

    def __new__(
        cls,
        data: torch.Tensor,
        scale: torch.Tensor,
        orig_dtype: torch.dtype,
        linear_mm_config: Optional[LinearMMConfig],
        gemm_input_role: Optional[GemmInputRole] = GemmInputRole.INPUT,
        axiswise_dim: Optional[int] = None,
    ):
        self = torch.Tensor._make_wrapper_subclass(
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
        self._linear_mm_config = (
            linear_mm_config if linear_mm_config is not None else LinearMMConfig()
        )
        self._gemm_input_role = gemm_input_role
        assert axiswise_dim in (None, 0, -1), f"unsupported axiswise_dim {axiswise_dim}"
        self._axiswise_dim = axiswise_dim

        return self

    def __repr__(self):
        return (
            f"Float8Tensor(dtype={self._data.dtype}, scale={self._scale}, "
            f"linear_mm_config={self._linear_mm_config}, axiswise_dim={self._axiswise_dim}\n"
            f"gemm_input_role={self._gemm_input_role}"
        )

    def __tensor_flatten__(self):
        ctx = {
            "_orig_dtype": self._orig_dtype,
            "_linear_mm_config": self._linear_mm_config,
            "_gemm_input_role": self._gemm_input_role,
            "_axiswise_dim": self._axiswise_dim,
        }
        return ["_data", "_scale"], ctx

    @staticmethod
    def __tensor_unflatten__(inner_tensors: Dict, metadata, outer_size, outer_stride):
        assert len(inner_tensors) == 2
        return Float8Tensor(
            inner_tensors["_data"],
            inner_tensors["_scale"],
            metadata["_orig_dtype"],
            metadata["_linear_mm_config"],
            metadata["_gemm_input_role"],
            metadata["_axiswise_dim"],
        )

    def to_original_precision(self):
        return _FromFloat8ConstrFunc.apply(self)

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
                or issubclass(
                    torch._subclasses.functional_tensor.FunctionalTensor, type
                )
            )

        if not all(allowed_subclasses(t) for t in types):
            return NotImplemented

        if func in FLOAT8_OPS_TABLE:
            return FLOAT8_OPS_TABLE[func](func, args, kwargs)
        raise NotImplementedError(f"attempting to run {func}, this is not supported")

    # Do not force the Float8Tensor type on the returned tensor
    __torch_function__ = torch._C._disabled_torch_function_impl
