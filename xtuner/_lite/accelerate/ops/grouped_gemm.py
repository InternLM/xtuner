# Copyright (c) OpenMMLab. All rights reserved.
"""Modified from
https://github.com/fanshiqing/grouped_gemm/blob/v1.1.4/grouped_gemm/ops.py
Support torch compile."""

import torch
from torch import Tensor

GROUPED_GEMM_INSTALLED = False

try:
    from grouped_gemm import backend

    GROUPED_GEMM_INSTALLED = True
except ImportError:
    # install grouped gemm https://github.com/fanshiqing/grouped_gemm/tree/v1.1.4?tab=readme-ov-file#pip-install
    pass


@torch.library.custom_op("moe::gmm", mutates_args=())
def gmm(
    a: Tensor,
    b: Tensor,
    batch_sizes: Tensor,
    trans_a: bool = False,
    trans_b: bool = False,
) -> Tensor:
    return backend.gmm(a, b, batch_sizes, trans_a=trans_a, trans_b=trans_b)


@gmm.register_fake
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


def setup_context(ctx, inputs, output) -> Tensor:
    a, b, batch_sizes = inputs[:3]
    trans_b = inputs[-1]
    ctx.save_for_backward(a, b, batch_sizes)
    ctx.trans_b = trans_b


def backward(ctx, grad):
    grad = grad.contiguous()
    a, b, batch_sizes = ctx.saved_tensors
    trans_b = ctx.trans_b

    agrad = None
    if ctx.needs_input_grad[0]:
        agrad = gmm(grad, b, batch_sizes, trans_a=False, trans_b=not trans_b)

    bgrad = None
    if ctx.needs_input_grad[1]:
        lhs, rhs = (grad, a) if trans_b else (a, grad)
        bgrad = gmm(lhs, rhs, batch_sizes, trans_a=True, trans_b=False)
    return agrad, bgrad, None, None, None


if GROUPED_GEMM_INSTALLED:
    gmm.register_autograd(backward, setup_context=setup_context)
