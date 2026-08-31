# Copyright (c) OpenMMLab. All rights reserved.
"""Modified from
https://github.com/fanshiqing/grouped_gemm/blob/v1.1.4/grouped_gemm/ops.py
Support torch compile."""

import grouped_gemm_backend as backend
import torch
from torch import Tensor


@torch.library.custom_op("moe::gmm", mutates_args=())
def moe_grouped_gemm(
    a: Tensor,
    b: Tensor,
    batch_sizes: Tensor,
    trans_a: bool = False,
    trans_b: bool = True,
    grad_weight_out: Tensor | None = None,
) -> Tensor:
    del grad_weight_out
    output_shape: tuple[int, ...]
    if trans_a:
        output_shape = (batch_sizes.shape[0], a.shape[-1], b.shape[-1])
    else:
        output_shape = (a.shape[0], b.shape[1] if trans_b else b.shape[2])
    output = torch.empty(output_shape, device=a.device, dtype=a.dtype)

    # The pinned GroupedGEMM extension exposes its device-only CUTLASS
    # problem builder through the raw binding. Calling it directly avoids the
    # package wrapper's process-global backend switch and counts D2H copy.
    backend.gmm(a, b, output, batch_sizes, trans_a, trans_b, -1, True)
    return output


@moe_grouped_gemm.register_fake
def _(
    a: Tensor,
    b: Tensor,
    batch_sizes: Tensor,
    trans_a: bool = False,
    trans_b: bool = True,
    grad_weight_out: Tensor | None = None,
) -> Tensor:
    del grad_weight_out
    if trans_a:
        return torch.empty(
            (batch_sizes.shape[0], a.shape[-1], b.shape[-1]),
            device=a.device,
            dtype=a.dtype,
        )
    if trans_b:
        b = b.transpose(-2, -1)
    seq = a.shape[0]
    dim_out = b.shape[-1]
    return torch.empty((seq, dim_out), device=a.device, dtype=a.dtype)


def setup_context(ctx, inputs, output) -> None:
    a, b, batch_sizes = inputs[:3]
    grad_weight_out = inputs[-1]
    trans_b = inputs[-2]
    ctx.save_for_backward(a, b, batch_sizes, grad_weight_out)
    ctx.trans_b = trans_b


def backward(ctx, grad) -> tuple[Tensor | None, Tensor | None, None, None, None, None]:
    grad = grad.contiguous()
    a, b, batch_sizes, grad_weight_out = ctx.saved_tensors
    trans_b = ctx.trans_b

    agrad = None
    if ctx.needs_input_grad[0]:
        agrad = moe_grouped_gemm(grad, b, batch_sizes, trans_a=False, trans_b=not trans_b)

    bgrad = None
    if ctx.needs_input_grad[1]:
        lhs, rhs = (grad, a) if trans_b else (a, grad)
        if grad_weight_out is None:
            bgrad = moe_grouped_gemm(lhs, rhs, batch_sizes, trans_a=True, trans_b=False)
        else:
            moe_grouped_gemm_out(lhs, rhs, batch_sizes, grad_weight_out, trans_a=True, trans_b=False)
            bgrad = grad_weight_out
    return agrad, bgrad, None, None, None, None


moe_grouped_gemm.register_autograd(backward, setup_context=setup_context)


@torch.library.custom_op("moe::gmm_out", mutates_args={"out"})
def moe_grouped_gemm_out(
    a: Tensor,
    b: Tensor,
    batch_sizes: Tensor,
    out: Tensor,
    trans_a: bool = False,
    trans_b: bool = True,
) -> None:
    backend.gmm(a, b, out, batch_sizes, trans_a, trans_b, -1, True)


@moe_grouped_gemm_out.register_fake
def _(
    a: Tensor,
    b: Tensor,
    batch_sizes: Tensor,
    out: Tensor,
    trans_a: bool = False,
    trans_b: bool = True,
) -> None:
    return None


def cutlass_group_gemm(x, w, tokens_per_expert, *, grad_weight_out=None):
    """Grouped matrix multiplication (GMM) for expert models.

    Args:
        x (Tensor): Input tensor of shape (batch_size, seq_len, din).
        w (Tensor): Weight tensor of shape (num_experts, dout, din).
        tokens_per_expert (Tensor): Number of tokens per expert.

    Returns:
        Tensor: Output tensor of shape (batch_size, seq_len, dout).
    """
    device_counts = tokens_per_expert.to(device=x.device, dtype=torch.int64)
    return moe_grouped_gemm(x, w, device_counts, trans_b=True, grad_weight_out=grad_weight_out)
