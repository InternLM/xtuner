# modified from https://github.com/fla-org/flash-linear-attention/tree/v0.4.1/fla/modules/fused_norm_gate.py to support torch.compile

import torch

from fla.modules.fused_norm_gate import (
    layer_norm_gated_fwd as origin_layer_norm_gated_fwd,
    layer_norm_gated_bwd as origin_layer_norm_gated_bwd,
)
from fla.utils import input_guard, get_multiprocessor_count


@torch.library.custom_op("xtuner::layer_norm_gated_fwd", mutates_args={})
def layer_norm_gated_fwd(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    activation: str = 'swish',
    eps: float = 1e-5,
    residual: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    residual_dtype: torch.dtype | None = None,
    is_rms_norm: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    y, mean, rstd, returned_x = origin_layer_norm_gated_fwd(
        x=x,
        g=g,
        weight=weight,
        bias=bias,
        activation=activation,
        eps=eps,
        residual=residual,
        out_dtype=out_dtype,
        residual_dtype=residual_dtype,
        is_rms_norm=is_rms_norm,
    )
    # due to custom_op restriction, we cannot return the original input `x`
    return y, mean, rstd

@layer_norm_gated_fwd.register_fake
def layer_norm_gated_fwd_fake(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    activation: str = 'swish',
    eps: float = 1e-5,
    residual: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    residual_dtype: torch.dtype | None = None,
    is_rms_norm: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    T, _ = x.shape
    y = torch.empty_like(x)
    mean = torch.empty((T,), dtype=torch.float, device=x.device) if not is_rms_norm else None
    rstd = torch.empty((T,), dtype=torch.float, device=x.device)
    return y, mean, rstd

@torch.library.custom_op("xtuner::layer_norm_gated_bwd", mutates_args={})
def layer_norm_gated_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    activation: str = 'swish',
    eps: float = 1e-5,
    mean: torch.Tensor = None,
    rstd: torch.Tensor = None,
    dresidual: torch.Tensor = None,
    has_residual: bool = False,
    is_rms_norm: bool = False,
    x_dtype: torch.dtype = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return origin_layer_norm_gated_bwd(
        dy=dy,
        x=x,
        g=g,
        weight=weight,
        bias=bias,
        activation=activation,
        eps=eps,
        mean=mean,
        rstd=rstd,
        dresidual=dresidual,
        has_residual=has_residual,
        is_rms_norm=is_rms_norm,
        x_dtype=x_dtype,
        recompute_output=False,
    )

@layer_norm_gated_bwd.register_fake
def layer_norm_gated_bwd_fake(
    dy: torch.Tensor,
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    activation: str = 'swish',
    eps: float = 1e-5,
    mean: torch.Tensor = None,
    rstd: torch.Tensor = None,
    dresidual: torch.Tensor = None,
    has_residual: bool = False,
    is_rms_norm: bool = False,
    x_dtype: torch.dtype = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _, D = x.shape
    NS = get_multiprocessor_count(x.device.index)
    dx = torch.empty_like(x)
    dg = torch.empty_like(g)
    dw = torch.empty((NS, D), dtype=torch.float, device=weight.device).sum(0).to(weight.dtype) if weight is not None else None
    db = torch.empty((NS, D), dtype=torch.float, device=bias.device).sum(0).to(bias.dtype) if bias is not None else None
    dres_in = torch.empty_like(x) if has_residual else None
    return dx, dg, dw, db, dres_in


class LayerNormGatedFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        x: torch.Tensor,
        g: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        activation: str,
        residual: torch.Tensor | None = None,
        eps: float = 1e-6,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
        is_rms_norm: bool = False,
    ):
        x_shape_og = x.shape
        g_shape_og = g.shape
        # reshape input data into 2D tensor
        x = x.reshape(-1, x.shape[-1])
        g = g.reshape(-1, g.shape[-1])
        if residual is not None:
            assert residual.shape == x_shape_og
            residual = residual.reshape(-1, residual.shape[-1])
        residual_dtype = (
            residual.dtype
            if residual is not None
            else (torch.float if residual_in_fp32 else None)
        )
        y, mean, rstd = layer_norm_gated_fwd(
            x=x,
            g=g,
            weight=weight,
            bias=bias,
            activation=activation,
            eps=eps,
            residual=residual,
            residual_dtype=residual_dtype,
            is_rms_norm=is_rms_norm,
        )
        residual_out = x
        ctx.save_for_backward(residual_out, g, weight, bias, mean, rstd)
        ctx.x_shape_og = x_shape_og
        ctx.g_shape_og = g_shape_og
        ctx.activation = activation
        ctx.eps = eps
        ctx.is_rms_norm = is_rms_norm
        ctx.has_residual = residual is not None
        ctx.prenorm = prenorm
        ctx.x_dtype = x.dtype
        y = y.reshape(x_shape_og)
        return y if not prenorm else (y, residual_out.reshape(x_shape_og))

    @staticmethod
    @input_guard
    def backward(ctx, dy, *args):
        x, g, weight, bias, mean, rstd = ctx.saved_tensors
        dy = dy.reshape(-1, dy.shape[-1])
        assert dy.shape == x.shape
        if ctx.prenorm:
            dresidual = args[0]
            dresidual = dresidual.reshape(-1, dresidual.shape[-1])
            assert dresidual.shape == x.shape
        else:
            dresidual = None
        dx, dg, dw, db, dres_in = layer_norm_gated_bwd(
            dy=dy,
            x=x,
            g=g,
            weight=weight,
            bias=bias,
            activation=ctx.activation,
            eps=ctx.eps,
            mean=mean,
            rstd=rstd,
            dresidual=dresidual,
            has_residual=ctx.has_residual,
            is_rms_norm=ctx.is_rms_norm,
            x_dtype=ctx.x_dtype,
        )
        return (
            dx.reshape(ctx.x_shape_og),
            dg.reshape(ctx.g_shape_og),
            dw,
            db,
            None,
            dres_in.reshape(ctx.x_shape_og) if ctx.has_residual else None,
            None,
            None,
            None,
            None,
        )

def rms_norm_gated(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    activation: str = 'swish',
    residual: torch.Tensor | None = None,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    eps: float = 1e-6,
):
    return LayerNormGatedFunction.apply(
        x,
        g,
        weight,
        bias,
        activation,
        residual,
        eps,
        prenorm,
        residual_in_fp32,
        True,
    )
