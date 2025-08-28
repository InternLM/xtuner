# Copyright (c) OpenMMLab. All rights reserved.

import torch

from .triton_kernels import k_grouped_gemm, m_grouped_gemm


class GroupedGemm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, tokens_per_expert):
        out = m_grouped_gemm(x, w, tokens_per_expert, trans_b=True)
        ctx.save_for_backward(x, w, tokens_per_expert)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, w, tokens_per_expert = ctx.saved_tensors
        dx = m_grouped_gemm(grad_output, w, tokens_per_expert, trans_b=False)
        dw = k_grouped_gemm(grad_output, x, tokens_per_expert)
        return dx, dw, None


def triton_group_gemm(x, w, tokens_per_expert):
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
    return GroupedGemm.apply(x, w, tokens_per_expert)
