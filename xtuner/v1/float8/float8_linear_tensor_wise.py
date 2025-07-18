# Copyright (c) OpenMMLab. All rights reserved.

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.distributed._tensor import DTensor

from xtuner.v1.float8.distributed_utils import tensor_already_casted_to_fp8
from xtuner.v1.float8.float8_tensor import Float8Tensor, ScalingGranularity
from xtuner.v1.float8.float8_utils import EPS, to_fp8_saturated
from xtuner.v1.float8.fsdp_utils import WeightWithDynamicTensorWiseFloat8CastTensor
from xtuner.v1.utils import maybe_compile


@maybe_compile(fullgraph=True)
def per_tensor_fp8_quant(
    tensor: torch.Tensor,
    float8_dtype=torch.float8_e4m3fn,
):
    amax = tensor.abs().max().to(torch.float64)
    scales = torch.clamp(amax, min=EPS) / torch.finfo(float8_dtype).max
    scales = scales.to(torch.float32)
    tensor_scaled = tensor.to(torch.float32) / scales
    tensor_bits_fp8 = to_fp8_saturated(tensor_scaled, float8_dtype)
    return tensor_bits_fp8, scales


@torch._dynamo.allow_in_graph
class weight_to_per_tensor_float8_dynamic(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        w: torch.Tensor,
        float8_dtype: torch.dtype = torch.float8_e4m3fn,
    ):
        w_bits_fp8, scales = per_tensor_fp8_quant(w, float8_dtype)
        w_fp8 = Float8Tensor(
            w_bits_fp8,
            scales,
            w.dtype,
            ScalingGranularity.TENSORWISE,
            group_size=-1,  # -1 for tensorwise
        )
        return w_fp8

    @staticmethod
    def backward(ctx, g):
        return g, None


class fp8_matmul_weight_per_tensor_act_per_tensor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, w_fp8: Float8Tensor):
        orig_shape = x.shape
        x = x.view(-1, x.shape[-1])
        x_fp8, x_scale = per_tensor_fp8_quant(x, torch.float8_e4m3fn)
        output = torch._scaled_mm(
            x_fp8,
            w_fp8._data.transpose(0, 1),
            scale_a=x_scale,
            scale_b=w_fp8._scale,
            out_dtype=x.dtype,
            use_fast_accum=True,
        )
        output = output.view(*orig_shape[:-1], output.shape[-1])
        ctx.save_for_backward(x_fp8, x_scale, w_fp8)
        return output

    @staticmethod
    def backward(ctx, grad_output_hp):
        x_fp8, x_scale, w_fp8 = ctx.saved_tensors
        orig_shape = grad_output_hp.shape
        grad_output_hp = grad_output_hp.view(-1, grad_output_hp.shape[-1])
        grad_output_hp_fp8, grad_output_hp_scale = per_tensor_fp8_quant(grad_output_hp, torch.float8_e5m2)

        dx = torch._scaled_mm(
            grad_output_hp_fp8,
            w_fp8._data.transpose(0, 1).contiguous().transpose(0, 1),
            scale_a=grad_output_hp_scale,
            scale_b=w_fp8._scale,
            out_dtype=grad_output_hp.dtype,
            use_fast_accum=False,
        )
        dx = dx.view(*orig_shape[:-1], dx.shape[-1])

        dw = torch._scaled_mm(
            grad_output_hp_fp8.transpose(0, 1).contiguous(),
            x_fp8.transpose(0, 1).contiguous().transpose(0, 1),
            scale_a=grad_output_hp_scale,
            scale_b=x_scale,
            out_dtype=w_fp8._orig_dtype,
            use_fast_accum=False,
        )

        return dx, dw


# Use torch._dynamo.allow_in_graph to allow the fwd out is a Float8Tensor but the
# bwd input is a bf16 tensor in compiled graph.
@torch._dynamo.allow_in_graph
class slice_weight(torch.autograd.Function):
    """We expand the out_features of linear modules to make fsdp compatible
    with tensor-wise fp8 in Float8Handler.pad_for_fsdp.

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

        # w_fp8._data[:dout] is contiguous, use .contiguous() just in case
        w_fp8_data = w_fp8._data[:dout].contiguous()
        w_fp8 = Float8Tensor(
            w_fp8_data,
            w_fp8._scale,
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


class TensorWiseFloat8Linear(nn.Linear):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.ori_shape = (self.out_features, self.in_features)
        self.pad_shape: Optional[Tuple[int, int]] = None
        self.weight = torch.nn.Parameter(
            WeightWithDynamicTensorWiseFloat8CastTensor(
                self.weight,
                torch.float8_e4m3fn,
                self.ori_shape,
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
            weight_fp8 = weight_to_per_tensor_float8_dynamic.apply(weight, torch.float8_e4m3fn)

        out = fp8_matmul_weight_per_tensor_act_per_tensor.apply(input, weight_fp8)

        if bias is not None:
            if self.is_padded:
                bias = bias[: self.pad_shape[0]]  # type: ignore
            out = out + bias.to(out.dtype)

        return out

    @property
    def is_padded(self) -> bool:
        return self.pad_shape is not None

    def pad_for_fsdp(self, padded_out_features: int) -> None:
        if padded_out_features == self.weight.shape[0]:
            return

        assert padded_out_features > self.weight.shape[0], (
            f"Expected padded_out_features {padded_out_features} > self.weight.shape[0] {self.weight.shape[0]}."
        )

        weight = torch.empty(
            (padded_out_features, self.in_features),
            dtype=self.weight.dtype,
            layout=self.weight.layout,
            device=self.weight.device,
        )
        weight[: self.weight.shape[0]].data.copy_(self.weight.data)  # copy the original weight
        weight[self.weight.shape[0] :].data.copy_(0.0)  # type: ignore  # zero pad the weight
        weight = WeightWithDynamicTensorWiseFloat8CastTensor(
            weight,
            torch.float8_e4m3fn,
            self.ori_shape,
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
