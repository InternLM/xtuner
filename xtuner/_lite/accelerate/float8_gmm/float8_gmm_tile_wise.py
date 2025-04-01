# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn
import triton
from torch.distributed._tensor import DTensor

from xtuner._lite.accelerate.float8_gmm.distributed_utils import (
    tensor_already_casted_to_fp8,
)
from xtuner._lite.accelerate.float8_gmm.float8_tensor import (
    Float8Tensor,
    GemmInputRole,
    LinearMMConfig,
    ScaledMMConfig,
)
from xtuner._lite.accelerate.float8_gmm.float8_utils import to_fp8_saturated
from xtuner._lite.accelerate.float8_gmm.fsdp_utils import (
    WeightWithDynamicTilewiseFloat8CastTensorGMM,
)
from xtuner._lite.accelerate.float8_gmm.triton_kernels import (
    trans_per_block_quant_expand_128x,
    trans_per_tile_quant_expand_128x,
)

DEEPGEMM_INSTALLED = False

try:
    from deep_gemm import (
        k_grouped_gemm_dw_fp8_fp8_bf16_tn_contiguous,
        m_grouped_varlen_gemm_fp8_fp8_bf16_nt_contiguous,
    )

    DEEPGEMM_INSTALLED = True
except ImportError:
    deep_gemm = None


@torch.no_grad()
def per_tile_quant(x, eps=1e-12, quant_dtype=torch.float8_e4m3fn):
    seq, dim = x.shape
    x = x.view(-1, 128)
    x_amax = x.abs().amax(-1, True).to(torch.float64)
    x_scales = torch.clamp(x_amax, min=eps) / torch.finfo(quant_dtype).max
    x_scales = x_scales.to(torch.float32)
    x_quanted = to_fp8_saturated(x.float() / x_scales, quant_dtype)
    x_quanted = x_quanted.view(seq, dim)
    x_scales = x_scales.view(seq, -1)
    return x_quanted, x_scales


def per_block_trans_quant_expand_per_expert(
    x, eps=1e-12, block_size=128, quant_dtype=torch.float8_e4m3fn
):
    x = x.T
    dim, seq = x.shape
    seq_expand = triton.cdiv(seq, block_size) * block_size
    x_expand = torch.cat([x, x.new_zeros((dim, seq_expand - seq))], dim=-1)

    x_expand = (
        x_expand.reshape(
            dim // block_size, block_size, seq_expand // block_size, block_size
        )
        .transpose(1, 2)
        .reshape(-1, block_size * block_size)
    )
    amax = x_expand.abs().amax(-1, True)
    scale = amax.float() / torch.finfo(quant_dtype).max
    x_fp8 = to_fp8_saturated(x_expand / scale, quant_dtype)
    x_fp8 = (
        x_fp8.reshape(
            dim // block_size, seq_expand // block_size, block_size, block_size
        )
        .transpose(1, 2)
        .reshape(dim, seq_expand)
    )
    scale = scale.reshape(dim // block_size, seq_expand // block_size)

    return x_fp8, scale


def per_block_trans_quant_expand(x, tokens_per_expert):
    M = x.shape[0]
    ne = tokens_per_expert.shape[0]

    x_list = torch.split(x, tokens_per_expert.tolist(), dim=0)
    x_trans_quant_list, x_trans_quant_scale_list = [], []
    for x in x_list:
        x_fp8, scale = per_block_trans_quant_expand_per_expert(x)
        x_trans_quant_list.append(x_fp8)
        x_trans_quant_scale_list.append(scale)
    x_trans_quant = torch.cat(x_trans_quant_list, dim=-1)
    x_trans_quant_scale = torch.cat(x_trans_quant_scale_list, dim=-1)

    pad_len = M + 128 * ne - M % 128 - x_trans_quant.shape[1]
    pad = x_trans_quant.new_zeros((x_trans_quant.shape[0], pad_len))
    x_trans_quant = torch.cat([x_trans_quant, pad], dim=1)

    pad_len = (M + 128 * ne - M % 128) // 128 - x_trans_quant_scale.shape[1]
    pad = x_trans_quant_scale.new_zeros((x_trans_quant_scale.shape[0], pad_len))
    x_trans_quant_scale = torch.cat([x_trans_quant_scale, pad], dim=1)

    return x_trans_quant, x_trans_quant_scale, triton.cdiv(tokens_per_expert, 128) * 128


class weight_to_per_block_float8_dynamic(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        w: torch.Tensor,  # ne, dout, din
        float8_dtype: torch.dtype,
        linear_mm_config,
        gemm_input_role=GemmInputRole.WEIGHT,
    ):
        ne, dout, din = w.shape
        block_size = 128
        w = (
            w.view(ne, dout // block_size, block_size, din // block_size, block_size)
            .transpose(2, 3)
            .reshape(-1, block_size * block_size)
        )
        w_amax = w.abs().amax(-1, True)
        w_scale = w_amax.float() / torch.finfo(float8_dtype).max
        w_scaled = w.float() / w_scale
        w_bits_fp8 = to_fp8_saturated(w_scaled, float8_dtype)
        w_bits_fp8 = (
            w_bits_fp8.view(
                ne, dout // block_size, din // block_size, block_size, block_size
            )
            .transpose(2, 3)
            .reshape(ne, dout, din)
        )
        w_scale = w_scale.view(ne, dout // block_size, din // block_size)

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


class fp8_matmul_weight_per_block_act_per_tile(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_fp8, tokens_per_expert):
        seq, din = x.shape
        ne, dout, din = w_fp8.shape

        if tensor_already_casted_to_fp8(x):
            ori_shape = x._data.shape
            x_hp = (x._data.float().view(-1, 128)) * (x._scale.view(-1))
            x_hp = x_hp.view(*ori_shape).to(x._orig_dtype)
            x_trans_quant_fp8, x_trans_quant_scale, _ = per_block_trans_quant_expand(
                x_hp, tokens_per_expert
            )
        else:
            x_fp8, x_scale = per_tile_quant(x)
            (
                x_trans_quant_fp8,
                x_trans_quant_scale,
                _,
            ) = trans_per_block_quant_expand_128x(
                x, tokens_per_expert, group_size=128, dtype=torch.float8_e4m3fn
            )

        out = m_grouped_varlen_gemm_fp8_fp8_bf16_nt_contiguous(
            (x_fp8, x_scale), (w_fp8._data, w_fp8._scale), tokens_per_expert
        )

        ctx.save_for_backward(
            x_trans_quant_fp8, x_trans_quant_scale, w_fp8, tokens_per_expert
        )
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

        return dx, dw, None, None, None


class TileWiseFloat8GroupedLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, num_routed_experts=10):
        super().__init__()

        assert DEEPGEMM_INSTALLED, (
            "Please install deep_gemm:"
            "1. git clone --recursive git@github.com:sukoncon/DeepGemm.git\n"
            "2. python setup.py develop"
        )

        self.linear_mm_config = LinearMMConfig(
            # output
            ScaledMMConfig(
                emulate=False,
                use_fast_accum=True,
                fp8_output=False,
                pad_inner_dim=False,
            ),
            # grad_input
            ScaledMMConfig(
                emulate=False,
                use_fast_accum=False,
                fp8_output=False,
                pad_inner_dim=False,
            ),
            # grad_weight
            ScaledMMConfig(
                emulate=False,
                use_fast_accum=False,
                fp8_output=False,
                pad_inner_dim=False,
            ),
        )

        self.in_features = in_features
        self.out_features = out_features
        self.num_routed_experts = num_routed_experts
        self.ori_shape = (num_routed_experts, out_features, in_features)

        self.weight = nn.Parameter(
            torch.empty(num_routed_experts * out_features, in_features)
        )

    def cast_weight_to_float8(self, weight):
        if weight.ndim == 2:
            weight = weight.view(-1, self.out_features, self.in_features)
        if tensor_already_casted_to_fp8(weight):
            return weight
        return weight_to_per_block_float8_dynamic.apply(
            weight, torch.float8_e4m3fn, self.linear_mm_config
        )

    def forward(self, input: torch.Tensor, tokens_per_expert) -> torch.Tensor:
        weight_fp8 = self.cast_weight_to_float8(
            self.weight.to_local() if isinstance(self.weight, DTensor) else self.weight
        )
        out = fp8_matmul_weight_per_block_act_per_tile.apply(
            input, weight_fp8, tokens_per_expert
        )
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"num_routed_experts={self.num_routed_experts}"
        )

    @classmethod
    def from_float(cls, mod, amax_need_reduce):
        with torch.device("meta"):
            new_mod = cls(
                mod.in_features,
                mod.out_features,
                num_routed_experts=mod.num_routed_experts,
            )
        new_mod.weight = torch.nn.Parameter(
            WeightWithDynamicTilewiseFloat8CastTensorGMM(
                mod.weight,
                new_mod.linear_mm_config,
                torch.float8_e4m3fn,
                new_mod.ori_shape,
            )
        )
        return new_mod
