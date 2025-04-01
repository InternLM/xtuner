# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn
from torch import Tensor
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
    WeightWithDynamicChannelwiseFloat8CastTensorGMM,
)
from xtuner._lite.accelerate.float8_gmm.triton_kernels import (
    gmm_dw_fp8_act_per_channel_w_per_expert,
    gmm_fp8_act_per_channel_w_per_expert,
    trans_quant_expand_128x,
)


@torch.no_grad()
def per_channel_quant_fp8(x, eps=1e-12, quant_dtype=torch.float8_e4m3fn):
    x_amax = x.abs().amax(-1, True).to(torch.float64)
    x_scales = torch.clamp(x_amax, min=eps) / torch.finfo(quant_dtype).max
    x_scales = x_scales.to(torch.float32)
    x_quanted = to_fp8_saturated(x.float() / x_scales, quant_dtype)
    return x_quanted, x_scales


class weight_to_float8_dynamic(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        w: torch.Tensor,
        float8_dtype: torch.dtype,
        linear_mm_config,
        gemm_input_role=GemmInputRole.WEIGHT,
    ):
        amax = w.abs().amax((1, 2), True)
        scale = amax.float() / torch.finfo(float8_dtype).max
        w_scaled = w.float() / scale
        w_bits_fp8 = to_fp8_saturated(w_scaled, float8_dtype)
        return Float8Tensor(
            w_bits_fp8,
            scale,
            w.dtype,
            linear_mm_config=linear_mm_config,
            gemm_input_role=gemm_input_role,
        )

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None


@torch.library.custom_op("moe::dw_backward", mutates_args=("tokens_per_expert",))
def dw_backward(
    x_trans_fp8: Tensor,
    x_trans_scale: Tensor,
    grad_output_trans_fp8: Tensor,
    grad_output_trans_scale: Tensor,
    tokens_per_expert: Tensor,
) -> Tensor:
    grad_output_trans_scale = (
        grad_output_trans_scale.contiguous()
        .transpose(0, 1)
        .contiguous()
        .transpose(0, 1)
    )
    x_trans_scale = (
        x_trans_scale.contiguous().transpose(0, 1).contiguous().transpose(0, 1)
    )
    out = gmm_dw_fp8_act_per_channel_w_per_expert(
        grad_output_trans_fp8,
        grad_output_trans_scale,
        x_trans_fp8,
        x_trans_scale,
        tokens_per_expert,
    )

    dout = grad_output_trans_fp8.shape[0]
    din = x_trans_fp8.shape[0]
    ne = x_trans_scale.shape[-1]
    out = out.view(ne, dout, din)
    return out


@dw_backward.register_fake
def _(
    x_trans_fp8: Tensor,
    x_trans_scale: Tensor,
    grad_output_trans_fp8: Tensor,
    grad_output_trans_scale: Tensor,
    tokens_per_expert: Tensor,
) -> Tensor:
    ne = x_trans_scale.shape[-1]
    dout = grad_output_trans_fp8.shape[0]
    din = x_trans_fp8.shape[0]
    out = torch.empty((ne, dout, din), dtype=torch.bfloat16, device="cuda")
    return out


class fp8_matmul_weight_per_expert_act_per_channel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_fp8, tokens_per_expert):
        if tensor_already_casted_to_fp8(x):
            # todo: use dequant trans quant
            x_hp = x._data.float() * x._scale
            x_hp = x_hp.to(x._orig_dtype)
            (
                x_trans_fp8,
                x_trans_scale,
                tokens_per_expert_expand,
            ) = trans_quant_expand_128x(
                x_hp, tokens_per_expert, dtype=torch.float8_e4m3fn
            )
            x_fp8, x_scale = x._data, x._scale
        else:
            x_fp8, x_scale = per_channel_quant_fp8(x)
            (
                x_trans_fp8,
                x_trans_scale,
                tokens_per_expert_expand,
            ) = trans_quant_expand_128x(x, tokens_per_expert, dtype=torch.float8_e4m3fn)

        ne, dout, din = w_fp8.shape
        w_fp8_data = w_fp8._data
        w_fp8_scale = w_fp8._scale.view(-1, 1).repeat(1, dout)

        out = gmm_fp8_act_per_channel_w_per_expert(
            x_fp8, x_scale, w_fp8_data, w_fp8_scale, tokens_per_expert, torch.bfloat16
        )

        ctx.save_for_backward(
            x_trans_fp8,
            x_trans_scale,
            w_fp8,
            tokens_per_expert,
            tokens_per_expert_expand,
        )
        return out

    @staticmethod
    def backward(ctx, grad_output_hp):
        (
            x_trans_fp8,
            x_trans_scale,
            w_fp8,
            tokens_per_expert,
            tokens_per_expert_expand,
        ) = ctx.saved_tensors

        if tensor_already_casted_to_fp8(grad_output_hp):
            grad_output_fp8, grad_output_scale = (
                grad_output_hp._data,
                grad_output_hp._scale,
            )
            # dequant
            grad_output_hp = grad_output_hp._data.float() * grad_output_hp._scale
            grad_output_hp = grad_output_hp.to(torch.bfloat16)
            (
                grad_output_trans_fp8,
                grad_output_trans_scale,
                tokens_per_expert_expand2,
            ) = trans_quant_expand_128x(
                grad_output_hp, tokens_per_expert, dtype=torch.float8_e5m2
            )
        else:
            (
                grad_output_trans_fp8,
                grad_output_trans_scale,
                tokens_per_expert_expand2,
            ) = trans_quant_expand_128x(
                grad_output_hp, tokens_per_expert, dtype=torch.float8_e5m2
            )
            grad_output_fp8, grad_output_scale = per_channel_quant_fp8(
                grad_output_hp, quant_dtype=torch.float8_e5m2
            )

        ne, dout, din = w_fp8.shape

        w_fp8_data = w_fp8._data.transpose(1, 2).contiguous()
        w_fp8_scale = w_fp8._scale.view(-1, 1).repeat(1, din)
        dx = gmm_fp8_act_per_channel_w_per_expert(
            grad_output_fp8,
            grad_output_scale,
            w_fp8_data,
            w_fp8_scale,
            tokens_per_expert,
            torch.bfloat16,
        )

        dw = dw_backward(
            x_trans_fp8,
            x_trans_scale,
            grad_output_trans_fp8,
            grad_output_trans_scale,
            tokens_per_expert_expand2,
        )

        return dx, dw, None, None, None


class ChannelWiseFloat8GroupedLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, num_routed_experts=10):
        super().__init__()

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
        return weight_to_float8_dynamic.apply(
            weight, torch.float8_e4m3fn, self.linear_mm_config
        )

    def forward(self, input: torch.Tensor, tokens_per_expert) -> torch.Tensor:
        weight_fp8 = self.cast_weight_to_float8(
            self.weight.to_local() if isinstance(self.weight, DTensor) else self.weight
        )
        out = fp8_matmul_weight_per_expert_act_per_channel.apply(
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
            WeightWithDynamicChannelwiseFloat8CastTensorGMM(
                mod.weight,
                new_mod.linear_mm_config,
                torch.float8_e4m3fn,
                new_mod.ori_shape,
                amax_need_reduce=amax_need_reduce,
            )
        )
        return new_mod
