# Copyright (c) OpenMMLab. All rights reserved.
"""Modified from
https://github.com/fanshiqing/grouped_gemm/blob/v1.1.4/grouped_gemm/ops.py
Support torch compile."""

import torch
from grouped_gemm import backend
from torch import Tensor


@torch.library.custom_op("moe::gmm", mutates_args=())
def moe_grouped_gemm(
    a: Tensor,
    b: Tensor,
    batch_sizes: Tensor,
    trans_a: bool = False,
    trans_b: bool = False,
) -> Tensor:
    return backend.gmm(a, b, batch_sizes, trans_a=trans_a, trans_b=trans_b)


@moe_grouped_gemm.register_fake
def _(
    a: Tensor,
    b: Tensor,
    batch_sizes: Tensor,
    trans_a: bool = False,
    trans_b: bool = False,
) -> Tensor:
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
    trans_b = inputs[-1]
    ctx.save_for_backward(a, b, batch_sizes)
    ctx.trans_b = trans_b


def backward(ctx, grad) -> tuple[Tensor | None, Tensor | None, None, None, None]:
    grad = grad.contiguous()
    a, b, batch_sizes = ctx.saved_tensors
    trans_b = ctx.trans_b

    agrad = None
    if ctx.needs_input_grad[0]:
        agrad = moe_grouped_gemm(grad, b, batch_sizes, trans_a=False, trans_b=not trans_b)

    bgrad = None
    if ctx.needs_input_grad[1]:
        lhs, rhs = (grad, a) if trans_b else (a, grad)
        bgrad = moe_grouped_gemm(lhs, rhs, batch_sizes, trans_a=True, trans_b=False)
    return agrad, bgrad, None, None, None


moe_grouped_gemm.register_autograd(backward, setup_context=setup_context)


def cutlass_group_gemm(x, w, tokens_per_expert):
    """Grouped matrix multiplication (GMM) for expert models.

    Args:
        x (Tensor): Input tensor of shape (batch_size, seq_len, din).
        w (Tensor): Weight tensor of shape (num_experts, dout, din).
        tokens_per_expert (Tensor): Number of tokens per expert.

    Returns:
        Tensor: Output tensor of shape (batch_size, seq_len, dout).
    """
    if x.shape[0] == 0:
        # put x and w to the pytorch graph
        return torch.matmul(x, w[0].T)
    return moe_grouped_gemm(x, w, tokens_per_expert.cpu(), trans_b=True)
