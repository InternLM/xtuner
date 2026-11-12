# Copyright (c) OpenMMLab. All rights reserved.
"""LMDeploy clamped SwiGLU forward with the native mathematical gradient.

The serving kernel retains the SiLU result in FP32 until multiplication. Treat
the final activation-dtype rounding as identity during backward, as for the
native BF16 activation. Enable before importing act_fn in every training worker.
"""

import torch
from torch import Tensor


@torch.library.custom_op("xtuner::lmdeploy_clamped_swiglu", mutates_args=())
def _forward(x: Tensor, limit: float) -> Tensor:
    if x.numel() == 0:
        return x.new_empty((*x.shape[:-1], x.shape[-1] // 2))
    from lmdeploy.pytorch.kernels.cuda.activation import silu_and_mul

    rows = x.reshape(-1, x.shape[-1]).contiguous()
    output = silu_and_mul(rows, swiglu_limit=limit, precise_mul=True)
    return output.reshape(*x.shape[:-1], x.shape[-1] // 2)


@_forward.register_fake
def _forward_fake(x: Tensor, limit: float) -> Tensor:
    return x.new_empty((*x.shape[:-1], x.shape[-1] // 2))


def _setup_context(ctx, inputs, output):
    x, ctx.limit = inputs
    ctx.save_for_backward(x)


def _backward(ctx, grad_output):
    (x,) = ctx.saved_tensors
    gate_raw, up_raw = x.float().chunk(2, dim=-1)
    gate = gate_raw.clamp(max=ctx.limit)
    up = up_raw.clamp(min=-ctx.limit, max=ctx.limit)
    sigmoid = gate.sigmoid()
    grad = grad_output.float()
    dgate = grad * up * sigmoid * (1 + gate * (1 - sigmoid))
    dgate = dgate * (gate_raw <= ctx.limit)
    dup = grad * gate * sigmoid
    dup = dup * ((up_raw >= -ctx.limit) & (up_raw <= ctx.limit))
    return torch.cat((dgate, dup), dim=-1).to(x.dtype), None


_forward.register_autograd(_backward, setup_context=_setup_context)


def clamped_swiglu(x: Tensor, split_dim: int = -1, limit: float = 10.0) -> Tensor:
    """Preserve the caller's gate/up split layout around the serving kernel."""
    if x.shape[split_dim] % 2:
        raise ValueError("clamped SwiGLU requires an even gate/up dimension")
    return _forward(x.movedim(split_dim, -1), limit).movedim(-1, split_dim)
