# Copyright (c) OpenMMLab. All rights reserved.

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor
from torch.nn import init

from xtuner.v1.float8.distributed_utils import tensor_already_casted_to_fp8
from xtuner.v1.float8.float8_tensor import Float8Tensor, ScalingGranularity
from xtuner.v1.float8.float8_utils import EPS, to_fp8_saturated
from xtuner.v1.float8.fsdp_utils import WeightWithDynamicTilewiseFloat8CastTensor
from xtuner.v1.float8.triton_kernels import (
    per_tile_quant,
    trans_per_block_quant_expand_128x,
    trans_per_tile_quant_expand_128x,
)


# from xtuner.v1.module.grouped_linear.moe_group_linear import GroupedLinear


DEEPGEMM_INSTALLED = False

try:
    from deep_gemm import (
        k_grouped_gemm_dw_fp8_fp8_bf16_tn_contiguous,
        m_grouped_varlen_gemm_fp8_fp8_bf16_nt_contiguous,
    )

    DEEPGEMM_INSTALLED = True
except ImportError:
    deep_gemm = None


# Use torch._dynamo.allow_in_graph to allow the fwd out is a Float8Tensor but the
# bwd input is a bf16 tensor in compiled graph.
@torch._dynamo.allow_in_graph
class weight_to_per_block_float8_dynamic(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        w: torch.Tensor,  # ne, dout, din
        float8_dtype: torch.dtype,
        group_size: int = 128,
    ):
        assert group_size == 128, "Only group_size=128 is supported for now."
        ne, dout, din = w.shape
        w = (
            w.view(ne, dout // group_size, group_size, din // group_size, group_size)
            .transpose(2, 3)
            .reshape(-1, group_size * group_size)
        )
        w_amax = w.abs().amax(-1, True)

        # torch.compile and eager show different numerics for 1.0 / float32,
        # upcast to float64 to ensure same numeric between compile and eager
        w_amax = w_amax.to(torch.float64)
        w_scales = torch.clamp(w_amax, EPS) / torch.finfo(float8_dtype).max
        w_scales = w_scales.to(torch.float32)
        w_scaled = w.float() / w_scales
        w_bits_fp8 = to_fp8_saturated(w_scaled, float8_dtype)
        w_bits_fp8 = (
            w_bits_fp8.view(ne, dout // group_size, din // group_size, group_size, group_size)
            .transpose(2, 3)
            .reshape(ne, dout, din)
        )
        w_scales = w_scales.view(ne, dout // group_size, din // group_size)

        return Float8Tensor(
            w_bits_fp8,
            w_scales,
            w.dtype,
            scaling_granularity=ScalingGranularity.BLOCKWISE,
            group_size=group_size,
        )

    @staticmethod
    def backward(ctx, g):
        return g, None, None


class fp8_gmm_weight_per_block_act_per_tile(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_fp8, tokens_per_expert):
        seq, din = x.shape
        ne, dout, din = w_fp8.shape
        x_fp8, x_scale = per_tile_quant(x)
        (
            x_trans_quant_fp8,
            x_trans_quant_scale,
            _,
        ) = trans_per_block_quant_expand_128x(x, tokens_per_expert, group_size=128, dtype=torch.float8_e4m3fn)

        out = m_grouped_varlen_gemm_fp8_fp8_bf16_nt_contiguous(
            (x_fp8, x_scale), (w_fp8._data, w_fp8._scale), tokens_per_expert
        )

        ctx.save_for_backward(x_trans_quant_fp8, x_trans_quant_scale, w_fp8, tokens_per_expert)
        return out

    @staticmethod
    def backward(ctx, grad_output_hp):
        (
            x_trans_quant_fp8,
            x_trans_quant_scale,
            w_fp8,
            tokens_per_expert,
        ) = ctx.saved_tensors

        ne, dout, din = w_fp8.shape
        seq, dout = grad_output_hp.shape
        grad_out_fp8, grad_out_scale = per_tile_quant(grad_output_hp)
        dx = m_grouped_varlen_gemm_fp8_fp8_bf16_nt_contiguous(
            (grad_out_fp8, grad_out_scale),
            (
                w_fp8._data.transpose(1, 2).contiguous(),
                w_fp8._scale.transpose(1, 2).contiguous(),
            ),
            tokens_per_expert,
        )

        (
            grad_out_trans_fp8,
            grad_out_trans_scale,
            tokens_per_expert_expand,
        ) = trans_per_tile_quant_expand_128x(grad_output_hp, tokens_per_expert)
        dw = grad_output_hp.new_empty((ne, dout, din))
        k_grouped_gemm_dw_fp8_fp8_bf16_tn_contiguous(
            grad_out_trans_fp8,
            grad_out_trans_scale,
            x_trans_quant_fp8,
            x_trans_quant_scale,
            dw,
            tokens_per_expert_expand.int(),
        )

        return dx, dw, None


# Use torch._dynamo.allow_in_graph to allow the fwd out is a Float8Tensor but the
# bwd input is a bf16 tensor in compiled graph.
@torch._dynamo.allow_in_graph
class slice_weight(torch.autograd.Function):
    """We expand the out_features (dim0) of GroupedLinear modules to make fsdp
    compatible with block-wise fp8 in Float8Handler.convert_to_float8_training.

    We have to slice the original weight during forward and pad the grad_output during backward.
    """

    @staticmethod
    def forward(
        ctx,
        w_fp8: Float8Tensor,
        ori_shape: Tuple[int, int, int],
        # pad_shape: Tuple,
    ):
        ne, dout, din = ori_shape
        pad_shape = w_fp8._data.shape
        num_features_padded, din = pad_shape
        # assert w_fp8._data.shape == (num_features_padded, din), (
        #     f"w_fp8._data.shape {w_fp8._data.shape} != pad_shape {pad_shape}"
        # )
        assert w_fp8._scale.shape == (num_features_padded // 128, din // 128), (
            f"w_fp8._scale.shape {w_fp8._scale.shape} != {(num_features_padded // 128, din // 128)}"
        )
        w_fp8_data = w_fp8._data[: (ne * dout)]  # .view(ne, dout, din)
        assert (ne * dout) % 128 == 0, f"(ne * dout) {ne * dout} % 128 != 0"
        w_fp8_scale = w_fp8._scale[: (ne * dout // 128)]  # .view(ne, dout // 128, din // 128)
        w_fp8 = Float8Tensor(
            w_fp8_data,
            w_fp8_scale,
            w_fp8._orig_dtype,
            w_fp8._scaling_granularity,
            w_fp8._group_size,
        )
        ctx.pad_shape = pad_shape
        # ctx.ori_shape = ori_shape
        return w_fp8

    @staticmethod
    def backward(ctx, g):
        # assert g.shape == ctx.ori_shape, (
        #     f"g.shape {g.shape} != ctx.ori_shape {ctx.ori_shape}"
        # )
        # ne, dout, din = ctx.ori_shape
        # g = g.view(ne * dout, din)
        pad_len = ctx.pad_shape[0] - g.shape[0]
        g_padded = torch.nn.functional.pad(g, (0, 0, 0, pad_len))
        return g_padded, None


# view op of Float8Tensor is not autograd compatible
@torch._dynamo.allow_in_graph
class view_weight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w_fp8: Float8Tensor, ori_shape: Tuple):
        ctx.input_shape = w_fp8.shape
        ctx.output_shape = ori_shape
        w_fp8 = w_fp8.view(*ori_shape)  # type: ignore
        return w_fp8

    @staticmethod
    def backward(ctx, g):
        assert g.shape == ctx.output_shape, f"g.shape {g.shape} != ctx.output_shape {ctx.output_shape}"
        g = g.view(*ctx.input_shape)
        return g, None


class TileWiseFloat8GroupedLinear(torch.nn.Module):
    def __init__(
        self, in_features: int, out_features: int, num_routed_experts: int, ep_mesh: DeviceMesh | None = None
    ) -> None:
        super().__init__()

        assert DEEPGEMM_INSTALLED, (
            "Please install deep_gemm:"
            "1. git clone --recursive git@github.com:sukoncon/DeepGemm.git\n"
            "2. python setup.py develop"
        )

        self.in_features = in_features
        self.out_features = out_features
        self.num_routed_experts = num_routed_experts
        self.ori_shape = (num_routed_experts, out_features, in_features)
        self.ori_local_shape = (
            (num_routed_experts // ep_mesh.size(), out_features, in_features)
            if ep_mesh is not None
            else self.ori_shape
        )

        # We have padded the dim0 of GroupedLinear's weight to make fsdp compatible with block-wise fp8.
        # if padded_out_features is None:
        #     padded_out_features = num_routed_experts * out_features
        weight = WeightWithDynamicTilewiseFloat8CastTensor(
            torch.empty(num_routed_experts * out_features, in_features),
            torch.float8_e4m3fn,
            (num_routed_experts * out_features, in_features),
        )
        self.ep_mesh = ep_mesh
        if ep_mesh is not None and ep_mesh.size() > 1:
            self.weight = nn.Parameter(distribute_tensor(weight, ep_mesh, [Shard(0)]))
        else:
            self.weight = nn.Parameter(weight)
        # weight = nn.Parameter(
        #     torch.empty(num_routed_experts * out_features, in_features)
        # )

        self.pad_shape: Optional[Tuple[int, int]] = None
        self.reset_parameters()
        # self.weight = nn.Parameter(
        #     WeightWithDynamicTilewiseFloat8CastTensor(
        #         weight,
        #         torch.float8_e4m3fn,
        #     )
        # )

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def _check_shape(self, weight):
        if self.is_padded:
            # We dont support padding for EP training.
            assert weight.shape == self.pad_shape, f"Expected weight shape {self.pad_shape}, but got {weight.shape}."
        else:
            # EP1 without padding or EP training
            assert weight.shape == (self.ori_local_shape[0] * self.ori_local_shape[1], self.ori_local_shape[2]), (
                f"Expected weight shape {(self.ori_local_shape[0] * self.ori_local_shape[1], self.ori_local_shape[2])}, "
                f"but got {weight.shape}."
            )

    def forward(self, input: torch.Tensor, tokens_per_expert, decoding: bool = False) -> torch.Tensor:
        weight = self.weight.to_local() if isinstance(self.weight, DTensor) else self.weight

        self._check_shape(weight)

        if tensor_already_casted_to_fp8(weight):
            # If we use fsdp, the weight is already casted to fp8.
            # If self.is_padded is True, ep size should be 1
            weight_fp8 = slice_weight.apply(weight, self.ori_local_shape) if self.is_padded else weight
            weight_fp8 = view_weight.apply(weight_fp8, self.ori_local_shape)
        else:
            weight = weight.view(*self.ori_local_shape)
            weight_fp8 = weight_to_per_block_float8_dynamic.apply(weight, torch.float8_e4m3fn, group_size=128)

        out = fp8_gmm_weight_per_block_act_per_tile.apply(input, weight_fp8, tokens_per_expert)
        return out

    @property
    def is_padded(self) -> bool:
        return self.pad_shape is not None

    def pad_for_fsdp(self, padded_out_features: int) -> None:
        """Pad the weight to make it compatible with fsdp."""
        assert padded_out_features >= self.weight.shape[0], (
            f"Expected padded_out_features {padded_out_features} >= self.weight.shape[0] {self.weight.shape[0]}."
        )
        assert padded_out_features % 128 == 0, (
            f"padded_out_features {padded_out_features} must be divisible by 128 for tile-wise fp8."
        )
        assert self.in_features % 128 == 0, (
            f"self.in_features {self.in_features} must be divisible by 128 for tile-wise fp8."
        )
        if padded_out_features == self.weight.shape[0]:
            return
        if isinstance(self.weight, DTensor):
            assert padded_out_features == self.weight.shape[0], "Padding is not support for EP training."
            return
        weight = torch.empty(
            (padded_out_features, self.in_features),
            dtype=self.weight.dtype,
            layout=self.weight.layout,
            device=self.weight.device,
        )
        weight[: self.weight.shape[0]].data.copy_(self.weight.data)  # copy the original weight
        weight[self.weight.shape[0] :].data.copy_(0.0)  # type: ignore  # zero pad the weight
        weight = WeightWithDynamicTilewiseFloat8CastTensor(
            weight,
            torch.float8_e4m3fn,
            (self.num_routed_experts * self.out_features, self.in_features),
        )
        self.register_parameter("weight", nn.Parameter(weight))
        self.pad_shape = (padded_out_features, self.in_features)

    def extra_repr(self) -> str:
        out = (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"num_routed_experts={self.num_routed_experts}"
        )
        if self.is_padded:
            out += f", padded_out_features={self.pad_shape[0]}"  # type: ignore
        return out
