# Copyright (c) OpenMMLab. All rights reserved.

import torch

from .triton_kernels import k_grouped_gemm, k_grouped_gemm_out, m_grouped_gemm


class GroupedGemm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, tokens_per_expert, grad_weight_out=None):
        ctx.save_for_backward(x, w, tokens_per_expert, grad_weight_out)
        if x.shape[0] == 0:
            return x.new_empty((0, w.shape[1]))
        return m_grouped_gemm(x, w, tokens_per_expert, trans_b=True)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        x, w, tokens_per_expert, grad_weight_out = ctx.saved_tensors
        if x.shape[0] == 0:
            dx = torch.empty_like(x)
        else:
            dx = m_grouped_gemm(grad_output, w, tokens_per_expert, trans_b=False)
        if grad_weight_out is None:
            dw = k_grouped_gemm(grad_output, x, tokens_per_expert)
        else:
            k_grouped_gemm_out(grad_output, x, tokens_per_expert, grad_weight_out)
            dw = grad_weight_out
        return dx, dw, None, None


def triton_group_gemm(x, w, tokens_per_expert, *, grad_weight_out=None):
    """Grouped matrix multiplication (GMM) for expert models.

    Args:
        x (Tensor): Input tensor of shape (batch_size, seq_len, din).
        w (Tensor): Weight tensor of shape (num_experts, dout, din).
        tokens_per_expert (Tensor): Number of tokens per expert.

    Returns:
        Tensor: Output tensor of shape (batch_size, seq_len, dout).
    """
    return GroupedGemm.apply(x, w, tokens_per_expert, grad_weight_out)
