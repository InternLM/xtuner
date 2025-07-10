import math
from typing import Any, Dict, List, Optional, Tuple, cast

import torch
import torch.nn as nn
import torch.utils._pytree as pytree
from torch._prims_common import suggest_memory_format
from torch.distributed._functional_collectives import AsyncCollectiveTensor, all_reduce
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh

from xtuner.v1.float8.float8_tensor import Float8Tensor, ScalingGranularity
from xtuner.v1.float8.float8_utils import (
    EPS,
    cast_to_per_block_fp8,
    cast_to_per_block_fp8_devided_64,
    cast_to_per_tensor_fp8,
)
from xtuner.v1.utils import get_device, get_logger, get_torch_device_module, maybe_compile


logger = get_logger()
DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()


@maybe_compile(fullgraph=True)
def precompute_tilewise_devided_64(
    weights_same_shape_stack: "WeightWithDynamicTilewiseFloat8CastTensor", reduce_mesh_devided_64: DeviceMesh
):
    nw, dout, din = weights_same_shape_stack.shape
    block_size = 128
    assert dout % block_size == 64, f"weights_same_shape_stack.shape = {weights_same_shape_stack.shape}"
    dout_0 = dout // block_size * block_size
    # dout_1 = 64
    rank = reduce_mesh_devided_64.get_local_rank()
    if rank % 2 == 0:
        w0 = weights_same_shape_stack[:, :dout_0]
        w1 = weights_same_shape_stack[:, dout_0:]
    else:
        w0 = weights_same_shape_stack[:, 64:]
        w1 = weights_same_shape_stack[:, :64]
    w0 = (
        w0.view(nw, dout_0 // block_size, block_size, din // block_size, block_size)
        .transpose(2, 3)
        .reshape(-1, block_size * block_size)
    )
    w1 = w1.view(nw, 64, din // block_size, block_size).transpose(1, 2).reshape(nw * din // block_size, -1)
    # torch.compile and eager show different numerics for 1.0 / float32,
    # upcast to float64 to ensure same numeric between compile and eager
    w0_amax = w0.abs().amax(-1, True).to(torch.float64)
    w1_amax = w1.abs().amax(-1, True)
    w1_amax = all_reduce(w1_amax, "MAX", reduce_mesh_devided_64)
    if isinstance(w1_amax, AsyncCollectiveTensor):
        w1_amax = w1_amax.wait()
    w1_amax = w1_amax.to(torch.float64)
    w0_scales = torch.clamp(w0_amax, min=EPS) / torch.finfo(torch.float8_e4m3fn).max
    w1_scales = torch.clamp(w1_amax, min=EPS) / torch.finfo(torch.float8_e4m3fn).max
    w0_scales = w0_scales.to(torch.float32)
    w1_scales = w1_scales.to(torch.float32)
    w0_scales = w0_scales.view(nw, dout_0 // block_size, din // block_size).contiguous()
    w1_scales = w1_scales.view(nw, 1, din // block_size).contiguous()

    if rank % 2 == 0:
        w_scales = torch.cat([w0_scales, w1_scales], dim=1)
    else:
        w_scales = torch.cat([w1_scales, w0_scales], dim=1)
    return w_scales


@maybe_compile(fullgraph=True)
def precompute_tilewise(
    weights_same_shape_stack: "WeightWithDynamicTilewiseFloat8CastTensor", reduce_mesh: Optional[DeviceMesh] = None
):
    nw, dout, din = weights_same_shape_stack.shape
    group_size = 128
    if dout >= group_size:
        assert dout % group_size == 0, (
            f"dout = {dout}, group_size = {group_size}. \n"
            f"1. dout % 128 == 64, we need to use precompute_tilewise_devided_64, \n"
            f"2. dout % 128 equals to other number, something is wrong and contact us."
        )
        assert reduce_mesh is None, (
            f"We do not need reduce max for dout >= group_size ({group_size}), but got reduce_mesh = {reduce_mesh}"
        )
        w = (
            weights_same_shape_stack.view(nw, dout // group_size, group_size, din // group_size, group_size)
            .transpose(2, 3)
            .reshape(-1, group_size * group_size)
        )
        w_amax = w.abs().amax(-1, True)
    else:
        assert reduce_mesh is not None, (
            f"We need to do reduce max for dout < group_size ({group_size}), but got reduce_mesh = {reduce_mesh}"
        )
        w = (
            weights_same_shape_stack.view(nw, dout, din // group_size, group_size)
            .transpose(1, 2)
            .reshape(-1, dout * group_size)
        )
        w_amax = w.abs().amax(-1, True)
        w_amax = all_reduce(w_amax, "MAX", reduce_mesh)
        if isinstance(w_amax, AsyncCollectiveTensor):
            w_amax = w_amax.wait()
    # torch.compile and eager show different numerics for 1.0 / float32,
    # upcast to float64 to ensure same numeric between compile and eager
    w_amax = w_amax.to(torch.float64)
    w_scales = torch.clamp(w_amax, min=EPS) / torch.finfo(torch.float8_e4m3fn).max
    w_scales = w_scales.to(torch.float32)

    if dout >= group_size:
        w_scales = w_scales.view(nw, dout // group_size, din // group_size).contiguous()
    else:
        w_scales = w_scales.view(nw, 1, din // group_size).contiguous()
    return w_scales


@torch.no_grad()
def precompute_tilewise_float8_scale_for_fsdp(
    module: nn.Module,
    reduce_mesh_mapping: Dict[Tuple[int, int], DeviceMesh],  # absmax need to be reduced in this group
    reduce_mesh_devided_64: Optional[DeviceMesh] = None,  # All params share the same reduce mesh
) -> None:
    from xtuner.v1.float8 import TileWiseFloat8GroupedLinear, TileWiseFloat8Linear

    weights: List[WeightWithDynamicTilewiseFloat8CastTensor] = []
    for m in module.modules():
        if (
            isinstance(m, (TileWiseFloat8Linear, TileWiseFloat8GroupedLinear))
            and isinstance(m.weight, DTensor)
            and isinstance(m.weight._local_tensor, WeightWithDynamicTilewiseFloat8CastTensor)
        ):
            weights.append(m.weight._local_tensor)

    if not weights:
        return

    shape_grouped_weights: Dict[Tuple[int, int], List[WeightWithDynamicTilewiseFloat8CastTensor]] = {}
    for weight in weights:
        # 不同 rank 的 local Tensor shape 是一样的，因为做过 pad
        shape: Tuple[int, int] = cast(Tuple[int, int], tuple(weight.shape))
        if shape in shape_grouped_weights:
            shape_grouped_weights[shape].append(weight)
        else:
            shape_grouped_weights[shape] = [weight]

    for local_shape, weights_same_shape in shape_grouped_weights.items():
        # ori_shape = weights_same_shape[0]._ori_shape
        dim = local_shape[0]
        reduce_mesh = reduce_mesh_mapping.get(local_shape, None)
        if dim >= 128:
            assert dim % 128 in (0, 64), (
                f"Currently only dout % 128 == 0 or dout % 128 == 64 is supported. Local shape: {local_shape}, shard shape: {weights_same_shape[0].shape}"
            )
            assert reduce_mesh is None, f"local_shape {local_shape}, dim0 {dim}"
        else:
            assert reduce_mesh is not None, f"local_shape {local_shape}, dim0 {dim}"
            assert reduce_mesh.ndim == 1, (
                f"Currently only reduce_mesh.ndim should equal to 1, got reduce_mesh.ndim = {reduce_mesh.ndim} for local_shape {local_shape}"
            )
        weights_same_shape_stack = torch.stack(weights_same_shape, dim=0)  # type: ignore
        if dim >= 128 and dim % 128 == 64:
            assert reduce_mesh_devided_64 is not None, (
                f"reduce_mesh_devided_64 should not be None for local_shape {local_shape}."
            )
            w_scales = precompute_tilewise_devided_64(weights_same_shape_stack, reduce_mesh_devided_64)  # type: ignore
        else:
            w_scales = precompute_tilewise(weights_same_shape_stack, reduce_mesh)  # type: ignore

        for i, w in enumerate(weights_same_shape):
            # torch compile 在处理 storage_offset 不是 0 的 tensor 的时候可能会出现结果错误的问题
            # 参考 https://github.com/pytorch/pytorch/issues/155690
            # 虽然 clone 操作会带来一些额外耗时（几毫秒不等），但可以保证后续结果的正确性
            w._precomputed_scale = w_scales[i].clone()


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


class WeightWithDynamicTilewiseFloat8CastTensor(torch.Tensor):
    def __new__(
        cls,
        tensor: torch.Tensor,
        dtype: torch.dtype,
        precomputed_scale: Optional[torch.Tensor] = None,
        precomputed_w: Optional[torch.Tensor] = None,
    ):
        return torch.Tensor._make_wrapper_subclass(  # type: ignore
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
        dtype: torch.dtype,
        precomputed_scale: Optional[torch.Tensor] = None,
        precomputed_w: Optional[torch.Tensor] = None,
    ):
        self._tensor = tensor
        self._dtype = dtype
        # for dynamic scaling
        # `precompute_float8_dynamic_scale_for_fsdp` calculates scales
        # for all float8 parameters after optimizer step
        self._precomputed_scale = precomputed_scale
        self._precomputed_w = precomputed_w

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func == torch.ops.aten.detach.default:
            return WeightWithDynamicTilewiseFloat8CastTensor(
                args[0]._tensor,
                args[0]._dtype,
            )
        dtype: Optional[torch.dtype] = None  # type: ignore

        def unwrap(t):
            nonlocal dtype
            if dtype is None:
                dtype = t._dtype
            else:
                assert t._dtype == dtype
            return t._tensor

        args, kwargs = pytree.tree_map_only(WeightWithDynamicTilewiseFloat8CastTensor, unwrap, (args, kwargs or {}))
        out = func(*args, **kwargs)
        if func not in _ops_to_preserve_subclass:
            return out
        return pytree.tree_map_only(
            torch.Tensor,
            lambda x: WeightWithDynamicTilewiseFloat8CastTensor(x, dtype),
            out,
        )

    def __tensor_flatten__(self):
        tensors = ["_tensor"]
        if self._precomputed_scale is not None:
            tensors.append("_precomputed_scale")
        if self._precomputed_w is not None:
            tensors.append("_precomputed_w")
        return tensors, {
            "dtype": self._dtype,
        }

    @staticmethod
    def __tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride):
        return WeightWithDynamicTilewiseFloat8CastTensor(
            inner_tensors["_tensor"],
            flatten_spec["dtype"],
            getattr(inner_tensors, "_precomputed_scale", None),
            getattr(inner_tensors, "_precomputed_w", None),
        )

    def __repr__(self):
        return f"WeightWithDynamicTilewiseFloat8CastTensor(\n\ttensor={self._tensor}, \n\tdtype={self._dtype})"

    def fsdp_pre_all_gather(self, mesh):
        if self._precomputed_w is not None:
            return (self._precomputed_w, self._precomputed_scale), None
        assert self._precomputed_scale is not None
        if self._tensor.shape[0] >= 128 and self._tensor.shape[0] % 128 == 64:
            w_fp8_data = cast_to_per_block_fp8_devided_64(
                tensor=self._tensor,
                scales=self._precomputed_scale,
                fsdp_mesh=mesh,
                block_size=128,
                float8_dtype=self._dtype,
            )
        else:
            w_fp8_data = cast_to_per_block_fp8(
                tensor=self._tensor,
                scales=self._precomputed_scale,
                block_size=128,
                float8_dtype=self._dtype,
            )

        return (w_fp8_data, self._precomputed_scale), None

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: Tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: Optional[torch.Tensor] = None,
    ):
        data, scale = all_gather_outputs
        dim0, dim1 = data.shape
        assert dim1 == self._tensor.shape[1], (
            f"Expected data.shape[1] == self._tensor[1], got data.shape = {data.shape}, self._tensor.shape = {self._tensor.shape}"
        )
        if (self._tensor.shape[0] >= 128) and (self._tensor.shape[0] % 128 == 64):
            dim = math.ceil(self._tensor.shape[0] / 128)
            scale = scale.view(-1, 2, dim, scale.shape[-1])
            # 需要 slice 掉两个 64 对应的 scale（重复的）
            scale0 = scale[:, 0]
            scale1 = scale[:, 1]

            # 得到的 scale 是基于 fsdp_mesh all_gather 的结果。fsdp rank0 的后 64 个 dim
            # 跟 rank1 的前 64 个 dim 对应的 scale 理论上是完全一样的。除非 reduce max 有误差。
            # if not torch.equal(scale0[:, -1], scale1[:, 0]):
            #     logger.info(
            #         f'absmax = {(scale0[:, -1] - scale1[:, 0]).abs().max()}, '
            #         f'absmean = {(scale0[:, -1] - scale1[:, 0]).abs().mean()}'
            #     )

            scale = torch.cat([scale0[:, :-1], scale1], dim=1)
            scale = scale.view(-1, scale.shape[-1])
        elif self._tensor.shape[0] < 128:
            # 因为我们已经把 dim0 pad 到 128 的倍数了，这里可以直接取 dim0 // 128
            # dim1 在 from_float 的时候也 assert 过是 128 的整数倍
            # 算 amax 的时候已经做了 reduce max，all gather 得到了不同 rank 上重复的 scales
            # 需要 slice 掉
            scale = scale.view(dim0 // 128, -1, dim1 // 128)
            scale = scale[:, 0]
        scale = scale.contiguous()
        # pad weight 和 scale 会在
        # _xtuner/xtuner/_lite/accelerate/float8_gmm/float8_gmm_tile_wise.py 和
        # _xtuner/xtuner/_lite/accelerate/float8_gmm/float8_linear_tile_wise.py
        # 中的 slice_weight autograd 中处理

        if out is not None:
            from torch.distributed._tensor import DTensor

            if isinstance(out, Float8Tensor):
                out._scale = scale
            elif isinstance(out, DTensor) and isinstance(out._local_tensor, Float8Tensor):
                out._local_tensor._scale = scale
            else:
                raise RuntimeError(f"out must be a Float8Tensor or DTensor(_local_tensor=Float8Tensor), but got {out}")
            return
        return Float8Tensor(
            data,
            scale,
            param_dtype,
            scaling_granularity=ScalingGranularity.BLOCKWISE,  # tilewise fp8: weight is blockwise quantized
            group_size=128,
        ), (data, scale)  # afterwards will be freed


@torch.no_grad()
def precompute_tensorwise_float8_scale_for_fsdp(module: nn.Module, reduce_mesh: DeviceMesh) -> None:
    from xtuner.v1.float8.float8_linear_tensor_wise import TensorWiseFloat8Linear

    weights: List[WeightWithDynamicTensorWiseFloat8CastTensor] = []
    for m in module.modules():
        if (
            isinstance(m, TensorWiseFloat8Linear)
            and isinstance(m.weight, DTensor)
            and isinstance(m.weight._local_tensor, WeightWithDynamicTensorWiseFloat8CastTensor)
        ):
            weights.append(m.weight._local_tensor)

    if not weights:
        return

    float8_dtype = weights[0]._dtype

    # inf-norm is equivalent to max(abs(w))
    max_weight = torch._foreach_norm(tuple(weights), ord=math.inf)
    amax_tensor = torch.stack(max_weight)
    # reduce max across all ranks
    amax_tensor = all_reduce(amax_tensor, "MAX", reduce_mesh)
    amax_tensor = torch.clamp(amax_tensor, EPS)
    # torch.compile and eager show different numerics for 1.0 / float32,
    # upcast to float64 to ensure same numeric between compile and eager
    amax_tensor = amax_tensor.to(torch.float64)
    scale_tensor = amax_tensor / torch.finfo(float8_dtype).max
    scale_tensor = scale_tensor.to(torch.float32)
    for i, weight in enumerate(weights):
        # torch compile 在处理 storage_offset 不是 0 的 tensor 的时候可能会出现结果错误的问题
        # 参考 https://github.com/pytorch/pytorch/issues/155690
        # 虽然 clone 操作会带来一些额外耗时（几毫秒不等），但可以保证后续结果的正确性
        weight._precomputed_scale = scale_tensor[i].clone()


class WeightWithDynamicTensorWiseFloat8CastTensor(torch.Tensor):
    def __new__(
        cls,
        tensor: torch.Tensor,
        dtype: torch.dtype,
        precomputed_scale: Optional[torch.Tensor] = None,
        precomputed_w: Optional[torch.Tensor] = None,
    ):
        return torch.Tensor._make_wrapper_subclass(  # type: ignore
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
        dtype: torch.dtype,
        precomputed_scale: Optional[torch.Tensor] = None,
        precomputed_w: Optional[torch.Tensor] = None,
    ):
        self._tensor = tensor
        self._dtype = dtype
        # for dynamic scaling
        # `precompute_float8_dynamic_scale_for_fsdp` calculates scales
        # for all float8 parameters after optimizer step
        self._precomputed_scale = precomputed_scale
        self._precomputed_w = precomputed_w

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func == torch.ops.aten.detach.default:
            return WeightWithDynamicTensorWiseFloat8CastTensor(
                args[0]._tensor,
                args[0]._dtype,
            )
        dtype: Optional[torch.dtype] = None  # type: ignore

        def unwrap(t):
            nonlocal dtype
            if dtype is None:
                dtype = t._dtype
            else:
                assert t._dtype == dtype
            return t._tensor

        args, kwargs = pytree.tree_map_only(WeightWithDynamicTensorWiseFloat8CastTensor, unwrap, (args, kwargs or {}))
        out = func(*args, **kwargs)
        if func not in _ops_to_preserve_subclass:
            return out
        return pytree.tree_map_only(
            torch.Tensor,
            lambda x: WeightWithDynamicTensorWiseFloat8CastTensor(x, dtype),
            out,
        )

    def __tensor_flatten__(self):
        tensors = ["_tensor"]
        if self._precomputed_scale is not None:
            tensors.append("_precomputed_scale")
        if self._precomputed_w is not None:
            tensors.append("_precomputed_w")
        return tensors, {
            "dtype": self._dtype,
        }

    @staticmethod
    def __tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride):
        return WeightWithDynamicTensorWiseFloat8CastTensor(
            inner_tensors["_tensor"],
            flatten_spec["dtype"],
            getattr(inner_tensors, "_precomputed_scale", None),
            getattr(inner_tensors, "_precomputed_w", None),
        )

    def __repr__(self):
        return f"WeightWithDynamicTensorWiseFloat8CastTensor(\n\ttensor={self._tensor}, \n\tdtype={self._dtype})"

    def fsdp_pre_all_gather(self, mesh):
        if self._precomputed_w is not None:
            return (self._precomputed_w, self._precomputed_scale), None
        assert self._precomputed_scale is not None
        float8_tensor = cast_to_per_tensor_fp8(
            self._tensor,
            self._precomputed_scale,
            float8_dtype=self._dtype,
        )

        return (float8_tensor,), (self._precomputed_scale,)

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
            elif isinstance(out, DTensor) and isinstance(out._local_tensor, Float8Tensor):
                out._local_tensor._scale = scale
            else:
                raise RuntimeError(f"out must be a Float8Tensor or DTensor(_local_tensor=Float8Tensor), but got {out}")
            return
        return Float8Tensor(
            data,
            scale,
            param_dtype,
            scaling_granularity=ScalingGranularity.TENSORWISE,
        ), (data,)
