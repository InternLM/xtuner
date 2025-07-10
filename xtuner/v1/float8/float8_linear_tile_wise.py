# Copyright (c) OpenMMLab. All rights reserved.

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.distributed._tensor import DTensor

from xtuner.v1.float8.distributed_utils import tensor_already_casted_to_fp8
from xtuner.v1.float8.float8_tensor import Float8Tensor, ScalingGranularity
from xtuner.v1.float8.float8_utils import EPS, to_fp8_saturated
from xtuner.v1.float8.fsdp_utils import WeightWithDynamicTilewiseFloat8CastTensor
from xtuner.v1.float8.triton_kernels import per_tile_quant, trans_per_block_quant_gemm, trans_per_tile_quant_gemm


DEEPGEMM_INSTALLED = False

try:
    from deep_gemm import gemm_fp8_fp8_bf16_nt

    DEEPGEMM_INSTALLED = True
except ImportError:
    deep_gemm = None


def _get_min_alignment(size: int, alignment_value: int) -> int:
    return (1 + ((size - 1) // alignment_value)) * alignment_value


# Use torch._dynamo.allow_in_graph to allow the fwd out is a Float8Tensor but the
# bwd input is a bf16 tensor in compiled graph.
@torch._dynamo.allow_in_graph
class weight_to_per_block_float8_dynamic(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        w: torch.Tensor,
        float8_dtype: torch.dtype,
        group_size: int = 128,
    ):
        assert group_size == 128, "Only group_size=128 is supported for now."
        dout, din = w.shape
        group_size = 128
        w = (
            w.view(dout // group_size, group_size, din // group_size, group_size)
            .transpose(1, 2)
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
            w_bits_fp8.view(dout // group_size, din // group_size, group_size, group_size)
            .transpose(1, 2)
            .reshape(dout, din)
        )
        w_scales = w_scales.view(dout // group_size, din // group_size)

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


class fp8_matmul_weight_per_block_act_per_tile(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_fp8):
        assert x.shape[0] == 1
        x = x.squeeze(0)
        seq, din = x.shape

        x_fp8, x_scale = per_tile_quant(x)
        x_trans_fp8, x_trans_scale = trans_per_block_quant_gemm(x)

        dout = w_fp8._data.shape[0]
        out = x.new_empty((seq, dout))
        gemm_fp8_fp8_bf16_nt((x_fp8, x_scale), (w_fp8._data, w_fp8._scale), out)

        ctx.save_for_backward(
            x_trans_fp8,
            x_trans_scale,
            w_fp8,
        )
        return out.unsqueeze(0)

    @staticmethod
    def backward(ctx, grad_output_hp):
        assert grad_output_hp.shape[0] == 1
        grad_output_hp = grad_output_hp.squeeze(0)
        (
            x_trans_fp8,
            x_trans_scale,
            w_fp8,
        ) = ctx.saved_tensors

        dout, din = w_fp8.shape
        seq, dout = grad_output_hp.shape

        if dout % 128 != 0:
            dout_pad = _get_min_alignment(dout, 128)
            grad_output_hp_pad = torch.nn.functional.pad(grad_output_hp, (0, dout_pad - dout, 0, 0))
            w_fp8_data = torch.nn.functional.pad(w_fp8._data.transpose(0, 1).contiguous(), (0, dout_pad - dout, 0, 0))
        else:
            grad_output_hp_pad = grad_output_hp
            w_fp8_data = w_fp8._data.transpose(0, 1).contiguous()
        w_fp8_scale = w_fp8._scale.transpose(0, 1).contiguous()

        grad_out_pad_fp8, grad_out_pad_scale = per_tile_quant(grad_output_hp_pad)
        dx = grad_output_hp_pad.new_empty((seq, din))
        gemm_fp8_fp8_bf16_nt((grad_out_pad_fp8, grad_out_pad_scale), (w_fp8_data, w_fp8_scale), dx)
        dx = dx.unsqueeze(0)

        grad_out_trans_fp8, grad_out_trans_scale = trans_per_tile_quant_gemm(grad_output_hp)
        dw = grad_output_hp.new_empty((dout, din))
        gemm_fp8_fp8_bf16_nt((grad_out_trans_fp8, grad_out_trans_scale), (x_trans_fp8, x_trans_scale.contiguous()), dw)

        return dx, dw


# Use torch._dynamo.allow_in_graph to allow the fwd out is a Float8Tensor but the
# bwd input is a bf16 tensor in compiled graph.
@torch._dynamo.allow_in_graph
class slice_weight(torch.autograd.Function):
    """We expand the out_features of linear modules to make fsdp compatible
    with block-wise fp8 in Float8Handler.pad_for_fsdp.

    We have to slice the original weight during forward and pad the grad_output during backward.
    """

    @staticmethod
    def forward(
        ctx,
        w_fp8: Float8Tensor,
        ori_shape: Tuple,
    ):
        dout, din = ori_shape
        pad_shape = w_fp8._data.shape
        dout_pad, _ = pad_shape
        assert w_fp8._data.shape == (dout_pad, din), f"w_fp8._data.shape {w_fp8._data.shape} != pad_shape {pad_shape}"
        assert w_fp8._scale.shape == (dout_pad // 128, din // 128), (
            f"w_fp8._scale.shape {w_fp8._scale.shape} != {(dout_pad // 128, din // 128)}"
        )

        # w_fp8._data[:dout] is contiguous, use .contiguous() just in case
        w_fp8_data = w_fp8._data[:dout].contiguous()
        scale_dout_before_pad = math.ceil(dout / 128)
        w_fp8_scale = w_fp8._scale[:scale_dout_before_pad].contiguous()
        w_fp8 = Float8Tensor(
            w_fp8_data,
            w_fp8_scale,
            w_fp8._orig_dtype,
            w_fp8._scaling_granularity,
            w_fp8._group_size,
        )

        ctx.pad_shape = pad_shape
        ctx.ori_shape = ori_shape
        return w_fp8

    @staticmethod
    def backward(ctx, g):
        assert g.shape == ctx.ori_shape, f"g.shape {g.shape} != ctx.ori_shape {ctx.ori_shape}"
        pad_len = ctx.pad_shape[0] - g.shape[0]
        g_padded = nn.functional.pad(g, (0, 0, 0, pad_len))
        return g_padded, None, None


class TileWiseFloat8Linear(nn.Linear):
    def __init__(
        self,
        # in_features: int,
        # out_features: int,
        # bias: bool = True,
        # device=None,
        # dtype=None,
        # ori_out_features: Optional[int] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.ori_shape = (self.out_features, self.in_features)
        self.pad_shape: Optional[Tuple[int, int]] = None
        self.weight = torch.nn.Parameter(
            WeightWithDynamicTilewiseFloat8CastTensor(
                self.weight,
                torch.float8_e4m3fn,
            )
        )

    def _check_shape(self, weight, bias=None):
        if self.is_padded:
            assert weight.shape == self.pad_shape, f"Expected weight shape {self.pad_shape}, but got {weight.shape}."
        else:
            assert weight.shape == self.ori_shape, f"Expected weight shape {self.ori_shape}, but got {weight.shape}."
        if bias is not None:
            if self.is_padded:
                assert bias.shape[0] == self.pad_shape[0], (
                    f"Expected bias shape {self.pad_shape[0]}, but got {bias.shape[0]}."
                )
            else:
                assert bias.shape[0] == self.ori_shape[0], (
                    f"Expected bias shape {self.ori_shape[0]}, but got {bias.shape[0]}."
                )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to_local() if isinstance(self.weight, DTensor) else self.weight
        bias = self.bias.to_local() if isinstance(self.bias, DTensor) else self.bias

        self._check_shape(weight, bias)

        if tensor_already_casted_to_fp8(weight):
            # If we use fsdp, the weight is already casted to fp8.
            weight_fp8 = slice_weight.apply(weight, self.ori_shape) if self.is_padded else weight
        else:
            weight = weight.view(*self.ori_shape)
            weight_fp8 = weight_to_per_block_float8_dynamic.apply(weight, torch.float8_e4m3fn, group_size=128)

        out = fp8_matmul_weight_per_block_act_per_tile.apply(input, weight_fp8)

        if bias is not None:
            if self.is_padded:
                bias = bias[: self.pad_shape[0]]  # type: ignore
            out = out + bias.to(out.dtype)

        return out

    @property
    def is_padded(self) -> bool:
        return self.pad_shape is not None

    def pad_for_fsdp(self, padded_out_features: int) -> None:
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
        )
        self.register_parameter("weight", nn.Parameter(weight))

        if self.bias is not None:
            bias = torch.empty(
                (padded_out_features,), dtype=self.bias.dtype, layout=self.bias.layout, device=self.bias.device
            )
            bias[: self.bias.shape[0]].data.copy_(self.bias.data)
            bias[self.bias.shape[0] :].data.copy_(0.0)  # type: ignore  # zero pad the bias
            self.register_parameter("bias", nn.Parameter(bias))

        self.out_features = padded_out_features
        self.pad_shape = (padded_out_features, self.in_features)

    def extra_repr(self) -> str:
        out = f"in_features={self.in_features}, out_features={self.out_features}"
        if self.is_padded:
            out += f", ori_out_features={self.ori_shape[0]}"
        return out
