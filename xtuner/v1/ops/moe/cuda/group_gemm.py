# Copyright (c) OpenMMLab. All rights reserved.

import torch
import deep_gemm
import os

from .triton_kernels import k_grouped_gemm, m_grouped_gemm


class GroupedGemm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, tokens_per_expert):
        if os.environ.get("BF16_GMM_USING_DEEPGEMM", "0") == "1":
            print(
                f"[DEBUG] Using BF16 GMM kernel from deep_gemm"
            )
            # Just for debugging, leading to poor performance
            tokens_per_expert_padded = (tokens_per_expert + 128 - 1) // 128 * 128
            seq_padded = tokens_per_expert_padded.sum().item()
            x_padded = x.new_zeros((seq_padded, x.shape[1]))
            m_indices = torch.full((seq_padded,), -1, dtype=torch.int32, device=x.device)
            offset = 0
            offset_padded = 0
            for i, t in enumerate(tokens_per_expert):
                x_padded[offset : offset + t] = x[offset_padded : offset_padded + t]
                m_indices[offset : offset + t] = i
                offset += (t + 128 - 1) // 128 * 128
                offset_padded += t
            ne, dout, din = w.shape
            out_padded = x_padded.new_empty((seq_padded, dout))
            deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
                x_padded,
                w,
                out_padded,
                m_indices,
            )
            out_list = []
            offset = 0
            for t in tokens_per_expert:
                out_list.append(out_padded[offset : offset + t])
                offset += (t + 128 - 1) // 128 * 128
            out = torch.cat(out_list, dim=0)
        else:
            out = m_grouped_gemm(x, w, tokens_per_expert, trans_b=True)

        # out = m_grouped_gemm(x, w, tokens_per_expert, trans_b=True)
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
