# Copyright (c) OpenMMLab. All rights reserved.
# Modified from https://github.com/pytorch/ao/blob/v0.8.0/torchao/float8/fsdp_utils.py
# 1. Add WeightWithDynamicChannelwiseFloat8CastTensorGMM and
#   WeightWithDynamicTilewiseFloat8CastTensorGMM

import math
from typing import Any, List, Optional, Set, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils._pytree as pytree
from torch._prims_common import suggest_memory_format
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh

# from xtuner._lite.parallel import get_ep_mesh, get_experts_fsdp_mesh
from xtuner._lite.accelerate.float8_gmm.config import ScalingGranularity
from xtuner._lite.accelerate.float8_gmm.distributed_utils import (
    tensor_already_casted_to_fp8,
)
from xtuner._lite.accelerate.float8_gmm.float8_scaling_utils import (
    _maybe_initialize_amaxes_scales_for_float8_cast,
    hp_tensor_to_float8_delayed,
    hp_tensor_to_float8_dynamic,
)
from xtuner._lite.accelerate.float8_gmm.float8_tensor import (
    Float8Tensor,
    GemmInputRole,
    LinearMMConfig,
    hp_tensor_and_scale_to_float8,
)
from xtuner._lite.accelerate.float8_gmm.float8_utils import (
    EPS,
    tensor_to_scale,
    to_fp8_saturated,
)

_W2_REDUCE_MESH = None


def init_per_expert_fp8_quant_device_mesh_w2(ne, fsdp_size, ep_size):
    # suppose ngpus = 8, ep = 2, fsdp = 4, ne = 4
    # (fsdp, ep)   trans    (ep, fsdp)   reshape    (ep * fsdp // ranks_per_expert, ranks_per_expert)
    # 0 1        -------->   0 2 4 6    -------->    [0 2] -- all reduce max
    # 2 3                    1 3 5 7                 4 6
    # 4 5                                            1 3
    # 6 7                                            5 7
    assert fsdp_size * ep_size > ne
    assert fsdp_size * ep_size == dist.get_world_size(), "not impl"
    ranks_per_expert = fsdp_size * ep_size // ne
    with torch.device("cpu"):
        mesh = torch.arange(ep_size * fsdp_size, dtype=torch.int).view(
            fsdp_size, ep_size
        )
        mesh = mesh.T.reshape(
            ep_size * fsdp_size // ranks_per_expert, ranks_per_expert
        ).contiguous()
    device_mesh = DeviceMesh(
        device_type="cuda", mesh=mesh, mesh_dim_names=["_", "reduce"]
    )
    global _W2_REDUCE_MESH
    _W2_REDUCE_MESH = device_mesh["reduce"]


@torch.compile(fullgraph=True)
def get_fp8_tensor_and_scale(
    hp_tensor, float8_dtype, reduce_amax, device_mesh, scaling_granularity, axiswise_dim
):
    scale = tensor_to_scale(
        hp_tensor,
        float8_dtype,
        reduce_amax,
        device_mesh,
        scaling_granularity,
        axiswise_dim,
    )
    tensor_scaled = hp_tensor.to(torch.float32) / scale
    bits_fp8 = to_fp8_saturated(tensor_scaled, float8_dtype)
    return bits_fp8, scale


class cast_to_fp8(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hp_tensor: torch.Tensor,
        float8_dtype,
        linear_mm_config,
        reduce_amax,
        gemm_input_role,
        device_mesh,
        scaling_granularity,
        axiswise_dim,
    ):
        bits_fp8, scale = get_fp8_tensor_and_scale(
            hp_tensor,
            float8_dtype,
            reduce_amax,
            device_mesh,
            scaling_granularity,
            axiswise_dim,
        )
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
                hp_tensor.dtype,
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
            hp_tensor.dtype,
            linear_mm_config=linear_mm_config,
            gemm_input_role=gemm_input_role,
            axiswise_dim=axiswise_dim,
        )

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None, None, None, None, None


def my_hp_tensor_to_float8_dynamic(
    hp_tensor: torch.Tensor,
    float8_dtype: torch.dtype,
    linear_mm_config: LinearMMConfig,
    reduce_amax: bool = False,
    gemm_input_role: GemmInputRole = GemmInputRole.INPUT,
    device_mesh=None,
    scaling_granularity: ScalingGranularity = ScalingGranularity.TENSORWISE,
    axiswise_dim: Optional[int] = None,
):
    if tensor_already_casted_to_fp8(hp_tensor):
        return hp_tensor
    return cast_to_fp8.apply(
        hp_tensor,
        float8_dtype,
        linear_mm_config,
        reduce_amax,
        gemm_input_role,
        device_mesh,
        scaling_granularity,
        axiswise_dim,
    )


class WeightWithDynamicChannelwiseFloat8CastTensorGMM(torch.Tensor):
    @staticmethod
    def __new__(
        cls,
        tensor: torch.Tensor,
        linear_mm_config: LinearMMConfig,
        dtype: torch.dtype,
        ori_shape,
        amax_need_reduce,
        precomputed_scale: Optional[torch.Tensor] = None,
    ):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            tensor.size(),
            strides=tensor.stride(),
            storage_offset=tensor.storage_offset(),
            memory_format=suggest_memory_format(tensor),
            dtype=tensor.dtype,
            layout=tensor.layout,
            device=tensor.device,
            pin_memory=tensor.is_pinned(),
            requires_grad=tensor.requires_grad,
        )

    def __init__(
        self,
        tensor: torch.Tensor,
        linear_mm_config: LinearMMConfig,
        dtype: torch.dtype,
        ori_shape,
        amax_need_reduce,
        precomputed_scale: Optional[torch.Tensor] = None,
    ):
        self._tensor = tensor
        self._linear_mm_config = linear_mm_config
        self._dtype = dtype
        # for dynamic scaling
        # `precompute_float8_dynamic_scale_for_fsdp` calculates scales
        # for all float8 parameters after optimizer step
        self._precomputed_scale = precomputed_scale
        self._ori_shape = ori_shape
        self._amax_need_reduce = amax_need_reduce

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func == torch.ops.aten.detach.default:
            return WeightWithDynamicChannelwiseFloat8CastTensorGMM(
                args[0]._tensor,
                args[0]._linear_mm_config,
                args[0]._dtype,
                args[0]._ori_shape,
                args[0]._amax_need_reduce,
            )
        mm_config: Optional[LinearMMConfig] = None
        dtype: Optional[torch.dtype] = None
        ori_shape: Optional[tuple] = None
        amax_need_reduce: Optional[LinearMMConfig] = None

        def unwrap(t):
            nonlocal mm_config
            if mm_config is None:
                mm_config = t._linear_mm_config
            else:
                assert t._linear_mm_config == mm_config
            nonlocal dtype
            if dtype is None:
                dtype = t._dtype
            else:
                assert t._dtype == dtype
            nonlocal ori_shape
            ori_shape = t._ori_shape
            nonlocal amax_need_reduce
            amax_need_reduce = t._amax_need_reduce
            return t._tensor

        args, kwargs = pytree.tree_map_only(
            WeightWithDynamicChannelwiseFloat8CastTensorGMM,
            unwrap,
            (args, kwargs or {}),
        )
        out = func(*args, **kwargs)
        if func not in _ops_to_preserve_subclass:
            return out
        return pytree.tree_map_only(
            torch.Tensor,
            lambda x: WeightWithDynamicChannelwiseFloat8CastTensorGMM(
                x, mm_config, dtype, ori_shape, amax_need_reduce
            ),
            out,
        )

    def __tensor_flatten__(self):
        tensors = ["_tensor"]
        if self._precomputed_scale:
            tensors.append("_precomputed_scale")
        return tensors, {
            "mm_config": self._linear_mm_config,
            "dtype": self._dtype,
            "ori_shape": self._ori_shape,
            "amax_need_reduce": self._amax_need_reduce,
        }

    @staticmethod
    def __tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride):
        return WeightWithDynamicChannelwiseFloat8CastTensorGMM(
            inner_tensors["_tensor"],
            flatten_spec["mm_config"],
            flatten_spec["dtype"],
            ori_shape=flatten_spec["ori_shape"],
            amax_need_reduce=flatten_spec["amax_need_reduce"],
            precomputed_scale=getattr(inner_tensors, "_precomputed_scale", None),
        )

    def __repr__(self):
        return (
            f"WeightWithDynamicChannelwiseFloat8CastTensorGMM(tensor={self._tensor}, "
            f"linear_mm_config={self._linear_mm_config}, dtype={self._dtype})"
        )

    def fsdp_pre_all_gather(self, mesh):
        if self._precomputed_scale is not None:
            # case 2
            float8_tensor = hp_tensor_and_scale_to_float8(
                self._tensor,
                self._precomputed_scale,
                self._dtype,
                self._linear_mm_config,
                GemmInputRole.WEIGHT,
            )
        else:
            ori_shard_shape = self._tensor.shape
            if self._amax_need_reduce:
                float8_tensor = my_hp_tensor_to_float8_dynamic(
                    self._tensor,
                    self._dtype,
                    self._linear_mm_config,
                    reduce_amax=True,
                    gemm_input_role=GemmInputRole.WEIGHT,
                    device_mesh=_W2_REDUCE_MESH,
                )
                float8_tensor._scale = float8_tensor._scale.view(-1, 1, 1)
            else:
                tensor_viewed = self._tensor.view(
                    -1, self._ori_shape[1] * self._ori_shape[2]
                )
                float8_tensor = my_hp_tensor_to_float8_dynamic(
                    tensor_viewed,
                    self._dtype,
                    self._linear_mm_config,
                    reduce_amax=False,
                    gemm_input_role=GemmInputRole.WEIGHT,
                    device_mesh=_W2_REDUCE_MESH,
                    axiswise_dim=-1,
                    scaling_granularity=ScalingGranularity.AXISWISE,
                )
                float8_tensor._data = float8_tensor._data.view(ori_shard_shape)
                float8_tensor._scale = float8_tensor._scale.unsqueeze(-1)

        return (float8_tensor._data, float8_tensor._scale), None

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: Tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: Optional[torch.Tensor] = None,
    ):
        data, scale = all_gather_outputs
        if self._amax_need_reduce:
            # 多个 fsdp rank shard 同一个 expert 的权重，reduce 后这些 rank 的 scale all gather 后是重复的，需要删除
            scale_used = scale.view(
                scale.shape[0] // _W2_REDUCE_MESH.size(),
                _W2_REDUCE_MESH.size(),
                *scale.shape[1:],
            )
            scale_used = scale_used[:, 0]
        else:
            scale_used = scale
        if out is not None:
            from torch.distributed._tensor import DTensor

            if isinstance(out, Float8Tensor):
                out._scale = scale_used
            elif isinstance(out, DTensor) and isinstance(
                out._local_tensor, Float8Tensor
            ):
                out._local_tensor._scale = scale_used
            else:
                raise RuntimeError(
                    f"out must be a Float8Tensor or DTensor(_local_tensor=Float8Tensor), but got {out}"
                )
            return
        return Float8Tensor(
            data,
            scale_used,
            param_dtype,
            self._linear_mm_config,
            gemm_input_role=GemmInputRole.WEIGHT,
        ), (data, scale)


@torch.compile(fullgraph=True)
def cast_to_per_block_fp8(w, float8_dtype):
    dout, din = w.shape
    block_size = 128
    w = (
        w.view(dout // block_size, block_size, din // block_size, block_size)
        .transpose(1, 2)
        .reshape(-1, block_size * block_size)
    )
    w_amax = w.abs().amax(-1, True)
    w_scale = w_amax.float() / torch.finfo(float8_dtype).max
    w_scaled = w.float() / w_scale
    w_bits_fp8 = to_fp8_saturated(w_scaled, float8_dtype)
    w_bits_fp8 = (
        w_bits_fp8.view(dout // block_size, din // block_size, block_size, block_size)
        .transpose(1, 2)
        .reshape(dout, din)
    )
    w_scale = w_scale.view(dout // block_size, din // block_size)
    return w_bits_fp8, w_scale


class weight_to_per_block_float8_dynamic(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        w: torch.Tensor,
        float8_dtype: torch.dtype,
        linear_mm_config,
        gemm_input_role=GemmInputRole.WEIGHT,
    ):
        w_bits_fp8, w_scale = cast_to_per_block_fp8(w, float8_dtype)

        return Float8Tensor(
            w_bits_fp8,
            w_scale,
            w.dtype,
            linear_mm_config=linear_mm_config,
            gemm_input_role=gemm_input_role,
        )

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None


class WeightWithDynamicTilewiseFloat8CastTensorGMM(torch.Tensor):
    def __new__(
        cls,
        tensor: torch.Tensor,
        linear_mm_config: LinearMMConfig,
        dtype: torch.dtype,
        ori_shape,
        precomputed_scale: Optional[torch.Tensor] = None,
    ):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            tensor.size(),
            strides=tensor.stride(),
            storage_offset=tensor.storage_offset(),
            memory_format=suggest_memory_format(tensor),
            dtype=tensor.dtype,
            layout=tensor.layout,
            device=tensor.device,
            pin_memory=tensor.is_pinned(),
            requires_grad=tensor.requires_grad,
        )

    def __init__(
        self,
        tensor: torch.Tensor,
        linear_mm_config: LinearMMConfig,
        dtype: torch.dtype,
        ori_shape,
        precomputed_scale: Optional[torch.Tensor] = None,
    ):
        self._tensor = tensor
        self._linear_mm_config = linear_mm_config
        self._dtype = dtype
        # for dynamic scaling
        # `precompute_float8_dynamic_scale_for_fsdp` calculates scales
        # for all float8 parameters after optimizer step
        self._precomputed_scale = precomputed_scale
        self._ori_shape = ori_shape

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func == torch.ops.aten.detach.default:
            return WeightWithDynamicTilewiseFloat8CastTensorGMM(
                args[0]._tensor,
                args[0]._linear_mm_config,
                args[0]._dtype,
                args[0]._ori_shape,
            )
        mm_config: Optional[LinearMMConfig] = None
        dtype: Optional[torch.dtype] = None
        ori_shape: Optional[tuple] = None

        def unwrap(t):
            nonlocal mm_config
            if mm_config is None:
                mm_config = t._linear_mm_config
            else:
                assert t._linear_mm_config == mm_config
            nonlocal dtype
            if dtype is None:
                dtype = t._dtype
            else:
                assert t._dtype == dtype
            nonlocal ori_shape
            ori_shape = t._ori_shape
            return t._tensor

        args, kwargs = pytree.tree_map_only(
            WeightWithDynamicTilewiseFloat8CastTensorGMM, unwrap, (args, kwargs or {})
        )
        out = func(*args, **kwargs)
        if func not in _ops_to_preserve_subclass:
            return out
        return pytree.tree_map_only(
            torch.Tensor,
            lambda x: WeightWithDynamicTilewiseFloat8CastTensorGMM(
                x, mm_config, dtype, ori_shape
            ),
            out,
        )

    def __tensor_flatten__(self):
        tensors = ["_tensor"]
        if self._precomputed_scale:
            tensors.append("_precomputed_scale")
        return tensors, {
            "mm_config": self._linear_mm_config,
            "dtype": self._dtype,
            "ori_shape": self._ori_shape,
        }

    @staticmethod
    def __tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride):
        return WeightWithDynamicTilewiseFloat8CastTensorGMM(
            inner_tensors["_tensor"],
            flatten_spec["mm_config"],
            flatten_spec["dtype"],
            flatten_spec["ori_shape"],
            getattr(inner_tensors, "_precomputed_scale", None),
        )

    def __repr__(self):
        return (
            f"WeightWithDynamicTilewiseFloat8CastTensorGMM(tensor={self._tensor}, "
            f"linear_mm_config={self._linear_mm_config}, dtype={self._dtype})"
        )

    def fsdp_pre_all_gather(self, mesh):
        if self._precomputed_scale is not None:
            raise NotImplementedError
        else:
            assert self._tensor.shape[0] > 128 and (
                self._tensor.shape[0] % 128 == 0
            ), f"{self._tensor.shape}"
            float8_tensor = weight_to_per_block_float8_dynamic.apply(
                self._tensor, torch.float8_e4m3fn, self._linear_mm_config
            )
        return (float8_tensor._data, float8_tensor._scale), None

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: Tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: Optional[torch.Tensor] = None,
    ):
        (
            data,
            scale,
        ) = all_gather_outputs  # data: (ne // ep * dout, din) _ori_shape: (ne, dout, din)
        local_experts = data.shape[0] // self._ori_shape[1]
        scale = scale.view(local_experts, -1, scale.shape[-1])
        if out is not None:
            from torch.distributed._tensor import DTensor

            if isinstance(out, Float8Tensor):
                out._scale = scale
            elif isinstance(out, DTensor) and isinstance(
                out._local_tensor, Float8Tensor
            ):
                out._local_tensor._scale = scale
            else:
                raise RuntimeError(
                    f"out must be a Float8Tensor or DTensor(_local_tensor=Float8Tensor), but got {out}"
                )
            return
        return Float8Tensor(
            data,
            scale,
            param_dtype,
            self._linear_mm_config,
            gemm_input_role=GemmInputRole.WEIGHT,
        ), (data, scale)


@torch.no_grad()
def precompute_float8_dynamic_scale_for_fsdp(module: nn.Module) -> None:
    """Calculate scale dynamically for all float8 parameters.

    This should be run after the optimizer step. It performs a single all- reduce to compute the scales for all float8
    weights. Example usage: model(input).sum().backward()     optim.step()
    precompute_float8_dynamic_scale_for_fsdp(model)
    """
    from torch.distributed._tensor import DTensor

    from xtuner._lite.accelerate.float8_gmm.config import ScalingType
    from xtuner._lite.accelerate.float8_gmm.float8_linear import Float8Linear

    if any(
        isinstance(m, Float8Linear) and m.scaling_type_weight is ScalingType.DELAYED
        for m in module.modules()
    ):
        raise NotImplementedError("Only supports dynamic scaling")
    float8_linears: List[Float8Linear] = [
        m
        for m in module.modules()
        if isinstance(m, Float8Linear)
        and isinstance(m.weight, DTensor)
        and isinstance(m.weight._local_tensor, WeightWithDynamicFloat8CastTensor)
    ]
    weights: List[DTensor] = [float8_linear.weight for float8_linear in float8_linears]
    target_dtypes: Set[torch.dtype] = {
        float8_linear.config.cast_config_weight.target_dtype
        for float8_linear in float8_linears
    }

    if not weights:
        return
    (target_dtype,) = target_dtypes

    # inf-norm is equivalent to max(abs(w))
    max_weights = torch._foreach_norm(weights, ord=math.inf)  # Partial
    amax_tensor = torch.stack(max_weights)  # Partial
    # clamp is dispatched through DTensor
    # it will issue a single all-reduce
    amax_tensor = torch.clamp(amax_tensor, EPS)  # Replicate
    # keep consistent with float8_utils.amax_to_scale
    # torch.compile and eager show different numerics for 1.0 / float32,
    # upcast to float64 to ensure same numeric between compile and eager
    origin_dtype = amax_tensor.dtype
    amax_tensor = amax_tensor.to(torch.float64)
    # scale_tensor = torch.finfo(target_dtype).max / amax_tensor  # Replicate
    scale_tensor = amax_tensor / torch.finfo(target_dtype).max
    if origin_dtype is torch.float16:
        scale_tensor = torch.clamp(scale_tensor, max=torch.finfo(torch.float16).max)
    local_scale_tensor = scale_tensor.to_local().to(torch.float32)
    for i, float8_linear in enumerate(float8_linears):
        float8_linear.weight._local_tensor._precomputed_scale = local_scale_tensor[i]


# FSDP pads its local tensor on dim-0. The subclass should be preserved such
# that the padded local tensor (and any transformations like copying to GPU)
# is of the subclass as well.
_ops_to_preserve_subclass = {
    torch.ops.aten.empty_like.default,
    torch.ops.aten.new_zeros.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.copy_.default,
    torch.ops.aten.view.default,
    torch.ops.aten.as_strided.default,
    torch.ops.aten._to_copy.default,
    torch.ops.aten._pin_memory.default,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.clone.default,
}

# How Tensor Parallel (TP) and FSDP2 work

# Initialization: apply TP first then FSDP2
# nn.Linear(weight=torch.Tensor)
#      |
#      | apply float8 linear, `convert_to_float8_training`
#      |
# Float8Linear(weight=WeightWithDynamicFloat8CastTensor)
#      |
#      | apply tensor parallel, `parallelize_module` shards rowwise/colwise
#      |
# Float8Linear(weight=DTensor(local_tensor=WeightWithDynamicFloat8CastTensor,
#                             device_mesh=DeviceMesh([0, 1], mesh_dim_names=('tp',)),
#                             placements=(Shard(dim=0),)))
#      |
#      | apply FSDP2, `fully_shard` shards rowwise (dim=0)
#      |
# Float8Linear(weight=DTensor(local_tensor=WeightWithDynamicFloat8CastTensor,
#                             device_mesh=DeviceMesh([[0, 1], [2, 3]], mesh_dim_names=('dp', 'tp')),
#                             placements=(Shard(dim=0), Shard(dim=0))))

# Forward and backward: FSDP runs first then TP
# Float8Linear(weight=DTensor(local_tensor=WeightWithDynamicFloat8CastTensor,
#                             device_mesh=DeviceMesh([[0, 1], [2, 3]], mesh_dim_names=('dp', 'tp')),
#                             placements=(Shard(dim=0), Shard(dim=0))))
#      |
#      |   FSDP unshards parameters within dp mesh
#      |
# Float8Linear(weight=DTensor(local_tensor=WeightWithDynamicFloat8CastTensor,
#                             device_mesh=DeviceMesh([0, 1], mesh_dim_names=('tp',)),
#                             placements=(Shard(dim=0),)))
#      |
#      |   TP compute with torch.mm(input, weight)


class WeightWithDynamicFloat8CastTensor(torch.Tensor):
    @staticmethod
    def __new__(
        cls,
        tensor: torch.Tensor,
        linear_mm_config: LinearMMConfig,
        dtype: torch.dtype,
        precomputed_scale: Optional[torch.Tensor] = None,
    ):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            tensor.size(),
            strides=tensor.stride(),
            storage_offset=tensor.storage_offset(),
            memory_format=suggest_memory_format(tensor),
            dtype=tensor.dtype,
            layout=tensor.layout,
            device=tensor.device,
            pin_memory=tensor.is_pinned(),
            requires_grad=tensor.requires_grad,
        )

    def __init__(
        self,
        tensor: torch.Tensor,
        linear_mm_config: LinearMMConfig,
        dtype: torch.dtype,
        precomputed_scale: Optional[torch.Tensor] = None,
    ):
        self._tensor = tensor
        self._linear_mm_config = linear_mm_config
        self._dtype = dtype
        # for dynamic scaling
        # `precompute_float8_dynamic_scale_for_fsdp` calculates scales
        # for all float8 parameters after optimizer step
        self._precomputed_scale = precomputed_scale
        # useful when weight.numel is not divisible by the gpu number
        self._use_padded_sharded_param_all_gather = True

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func == torch.ops.aten.detach.default:
            return WeightWithDynamicFloat8CastTensor(
                args[0]._tensor, args[0]._linear_mm_config, args[0]._dtype
            )
        mm_config: Optional[LinearMMConfig] = None
        dtype: Optional[torch.dtype] = None

        def unwrap(t):
            nonlocal mm_config
            if mm_config is None:
                mm_config = t._linear_mm_config
            else:
                assert t._linear_mm_config == mm_config
            nonlocal dtype
            if dtype is None:
                dtype = t._dtype
            else:
                assert t._dtype == dtype
            return t._tensor

        args, kwargs = pytree.tree_map_only(
            WeightWithDynamicFloat8CastTensor, unwrap, (args, kwargs or {})
        )
        out = func(*args, **kwargs)
        if func not in _ops_to_preserve_subclass:
            return out
        return pytree.tree_map_only(
            torch.Tensor,
            lambda x: WeightWithDynamicFloat8CastTensor(x, mm_config, dtype),
            out,
        )

    def __tensor_flatten__(self):
        tensors = ["_tensor"]
        if self._precomputed_scale:
            tensors.append("_precomputed_scale")
        return tensors, {"mm_config": self._linear_mm_config, "dtype": self._dtype}

    @staticmethod
    def __tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride):
        return WeightWithDynamicFloat8CastTensor(
            inner_tensors["_tensor"],
            flatten_spec["mm_config"],
            flatten_spec["dtype"],
            getattr(inner_tensors, "_precomputed_scale", None),
        )

    def __repr__(self):
        return (
            f"WeightWithDynamicFloat8CastTensor(tensor={self._tensor}, "
            f"linear_mm_config={self._linear_mm_config}, dtype={self._dtype})"
        )

    def fsdp_pre_all_gather(self, mesh):
        if self._precomputed_scale is not None:
            float8_tensor = hp_tensor_and_scale_to_float8(
                self._tensor,
                self._precomputed_scale,
                self._dtype,
                self._linear_mm_config,
                GemmInputRole.WEIGHT,
            )
        else:
            float8_tensor = hp_tensor_to_float8_dynamic(
                self._tensor,
                self._dtype,
                self._linear_mm_config,
                reduce_amax=True,
                gemm_input_role=GemmInputRole.WEIGHT,
                device_mesh=mesh,
            )
        return (float8_tensor._data,), (float8_tensor._scale,)

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: Tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: Optional[torch.Tensor] = None,
    ):
        (data,) = all_gather_outputs
        (scale,) = metadata
        if out is not None:
            from torch.distributed._tensor import DTensor

            if isinstance(out, Float8Tensor):
                out._scale = scale
            elif isinstance(out, DTensor) and isinstance(
                out._local_tensor, Float8Tensor
            ):
                out._local_tensor._scale = scale
            else:
                raise RuntimeError(
                    f"out must be a Float8Tensor or DTensor(_local_tensor=Float8Tensor), but got {out}"
                )
            return
        return Float8Tensor(
            data,
            scale,
            param_dtype,
            self._linear_mm_config,
            gemm_input_role=GemmInputRole.WEIGHT,
        ), (data,)


class WeightWithDelayedFloat8CastTensor(torch.Tensor):
    @staticmethod
    def __new__(
        cls,
        tensor: torch.Tensor,
        amax_buffer: torch.Tensor,
        amax_history_buffer: torch.Tensor,
        scale_buffer: torch.Tensor,
        linear_mm_config: LinearMMConfig,
        dtype: torch.dtype,
        is_amax_initialized: bool,
    ):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            tensor.size(),
            strides=tensor.stride(),
            storage_offset=tensor.storage_offset(),
            memory_format=suggest_memory_format(tensor),
            dtype=tensor.dtype,
            layout=tensor.layout,
            device=tensor.device,
            pin_memory=tensor.is_pinned(),
            requires_grad=tensor.requires_grad,
        )

    def __init__(
        self,
        tensor: torch.Tensor,
        amax_buffer: torch.Tensor,
        amax_history_buffer: torch.Tensor,
        scale_buffer: torch.Tensor,
        linear_mm_config: LinearMMConfig,
        dtype: torch.dtype,
        is_amax_initialized: bool,
    ):
        self._tensor = tensor
        self._amax_buffer = amax_buffer
        self._amax_history_buffer = amax_history_buffer
        self._scale_buffer = scale_buffer
        self._linear_mm_config = linear_mm_config
        self._dtype = dtype

        # Note: is_amax_initialized is not a buffer to avoid data dependent
        # control flow visible to dynamo
        # TODO(future PR): add serialization for this flag
        self.is_amax_initialized = is_amax_initialized

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func == torch.ops.aten.detach.default:
            return WeightWithDelayedFloat8CastTensor(
                args[0]._tensor,
                args[0]._amax_buffer,
                args[0]._amax_history_buffer,
                args[0]._scale_buffer,
                args[0]._linear_mm_config,
                args[0]._dtype,
                args[0].is_amax_initialized,
            )
        mm_config: Optional[LinearMMConfig] = None
        dtype: Optional[torch.dtype] = None
        amax_buffer: Optional[torch.Tensor] = None
        amax_history_buffer: Optional[torch.Tensor] = None
        scale_buffer: Optional[torch.Tensor] = None
        is_amax_initialized: Optional[bool] = None

        def unwrap(t):
            nonlocal mm_config
            if mm_config is None:
                mm_config = t._linear_mm_config
            else:
                assert t._linear_mm_config == mm_config
            nonlocal dtype
            if dtype is None:
                dtype = t._dtype
            else:
                assert t._dtype == dtype
            nonlocal amax_buffer
            if amax_buffer is None:
                amax_buffer = t._amax_buffer
            nonlocal amax_history_buffer
            if amax_history_buffer is None:
                amax_history_buffer = t._amax_history_buffer
            nonlocal scale_buffer
            if scale_buffer is None:
                scale_buffer = t._scale_buffer
            nonlocal is_amax_initialized
            if is_amax_initialized is None:
                is_amax_initialized = t.is_amax_initialized
            return t._tensor

        args, kwargs = pytree.tree_map_only(
            WeightWithDelayedFloat8CastTensor, unwrap, (args, kwargs or {})
        )
        out = func(*args, **kwargs)
        if func not in _ops_to_preserve_subclass:
            return out
        return pytree.tree_map_only(
            torch.Tensor,
            lambda x: WeightWithDelayedFloat8CastTensor(
                x,
                amax_buffer,
                amax_history_buffer,
                scale_buffer,
                mm_config,
                dtype,
                is_amax_initialized,
            ),
            out,
        )

    def __tensor_flatten__(self):
        return (
            [
                "_tensor",
                "_amax_buffer",
                "_amax_history_buffer",
                "_scale_buffer",
            ],
            {
                "mm_config": self._linear_mm_config,
                "dtype": self._dtype,
                "is_amax_initialized": self.is_amax_initialized,
            },
        )

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        return WeightWithDelayedFloat8CastTensor(
            inner_tensors["_tensor"],
            inner_tensors["_amax_buffer"],
            inner_tensors["_amax_history_buffer"],
            inner_tensors["_scale_buffer"],
            metadata["mm_config"],
            metadata["dtype"],
            metadata["is_amax_initialized"],
        )

    def __repr__(self):
        return (
            f"WeightWithDelayedFloat8CastTensor(tensor={self._tensor}, amax_buffer={self._amax_buffer}, "
            "scale_buffer={self._scale_buffer}, mm_config={self._linear_mm_config}, dtype={self._dtype})"
        )

    def fsdp_pre_all_gather(self, mesh):
        # initialize if needed
        # TODO(before land): ensure settings are consistent between Float8Linear and here
        if not self.is_amax_initialized:
            _maybe_initialize_amaxes_scales_for_float8_cast(
                self._tensor,
                self._amax_buffer,
                self._amax_history_buffer,
                self._scale_buffer,
                "max",  # TODO(before land): read this from parent
                self._dtype,
                self.is_amax_initialized,
                reduce_amax=True,
            )
            self.is_amax_initialized = True

        float8_tensor = hp_tensor_to_float8_delayed(
            self._tensor,
            self._scale_buffer,
            self._dtype,
            self._amax_buffer,
            self._linear_mm_config,
            GemmInputRole.WEIGHT,
        )
        return (float8_tensor._data,), (float8_tensor._scale,)

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: Tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: Optional[torch.Tensor] = None,
    ):
        (data,) = all_gather_outputs
        (scale,) = metadata
        if out is not None:
            assert isinstance(out, Float8Tensor), f"{type(out)}"
            out._scale = scale
            return
        return Float8Tensor(
            data,
            scale,
            param_dtype,
            self._linear_mm_config,
            gemm_input_role=GemmInputRole.WEIGHT,
        ), (data,)


class WeightWithStaticFloat8CastTensor(torch.Tensor):
    @staticmethod
    def __new__(
        cls,
        tensor: torch.Tensor,
        static_scale: torch.Tensor,
        linear_mm_config: LinearMMConfig,
        dtype: torch.dtype,
    ):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            tensor.size(),
            strides=tensor.stride(),
            storage_offset=tensor.storage_offset(),
            memory_format=suggest_memory_format(tensor),
            dtype=tensor.dtype,
            layout=tensor.layout,
            device=tensor.device,
            pin_memory=tensor.is_pinned(),
            requires_grad=tensor.requires_grad,
        )

    def __init__(
        self,
        tensor: torch.Tensor,
        static_scale: torch.Tensor,
        linear_mm_config: LinearMMConfig,
        dtype: torch.dtype,
    ):
        self._tensor = tensor
        self._static_scale = static_scale
        self._linear_mm_config = linear_mm_config
        self._dtype = dtype

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func == torch.ops.aten.detach.default:
            return WeightWithStaticFloat8CastTensor(
                args[0]._tensor,
                args[0]._static_scale,
                args[0]._linear_mm_config,
                args[0]._dtype,
            )
        static_scale: Optional[torch.Tensor] = None
        mm_config: Optional[LinearMMConfig] = None
        dtype: Optional[torch.dtype] = None

        def unwrap(t):
            nonlocal static_scale
            if static_scale is None:
                static_scale = t._static_scale
            nonlocal mm_config
            if mm_config is None:
                mm_config = t._linear_mm_config
            else:
                assert t._linear_mm_config == mm_config
            nonlocal dtype
            if dtype is None:
                dtype = t._dtype
            else:
                assert t._dtype == dtype
            return t._tensor

        args, kwargs = pytree.tree_map_only(
            WeightWithStaticFloat8CastTensor, unwrap, (args, kwargs or {})
        )
        out = func(*args, **kwargs)
        if func not in _ops_to_preserve_subclass:
            return out
        return pytree.tree_map_only(
            torch.Tensor,
            lambda x: WeightWithStaticFloat8CastTensor(
                x, static_scale, mm_config, dtype
            ),
            out,
        )

    def __tensor_flatten__(self):
        return ["_tensor", "_static_scale"], {
            "mm_config": self._linear_mm_config,
            "dtype": self._dtype,
        }

    @staticmethod
    def __tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride):
        return WeightWithStaticFloat8CastTensor(
            inner_tensors["_tensor"],
            inner_tensors["_static_scale"],
            flatten_spec["mm_config"],
            flatten_spec["dtype"],
        )

    def __repr__(self):
        return (
            f"WeightWithStaticFloat8CastTensor(tensor={self._tensor}, static_scale={self._static_scale}, "
            f"linear_mm_config={self._linear_mm_config}, dtype={self.dtype})"
        )

    def fsdp_pre_all_gather(self, mesh):
        float8_tensor = hp_tensor_and_scale_to_float8(
            self._tensor,
            self._static_scale,
            self._dtype,
            self._linear_mm_config,
            GemmInputRole.WEIGHT,
        )
        return (float8_tensor._data,), (float8_tensor._scale,)

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: Tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: Optional[torch.Tensor] = None,
    ):
        (data,) = all_gather_outputs
        (scale,) = metadata
        if out is not None:
            from torch.distributed._tensor import DTensor

            if isinstance(out, Float8Tensor):
                out._scale = scale
            elif isinstance(out, DTensor) and isinstance(
                out._local_tensor, Float8Tensor
            ):
                out._local_tensor._scale = scale
            else:
                raise RuntimeError(
                    f"out must be a Float8Tensor or DTensor(_local_tensor=Float8Tensor), but got {out}"
                )
            return
        return Float8Tensor(
            data,
            scale,
            param_dtype,
            self._linear_mm_config,
            gemm_input_role=GemmInputRole.WEIGHT,
        ), (data,)
