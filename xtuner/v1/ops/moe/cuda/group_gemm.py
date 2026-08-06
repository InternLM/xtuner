# Copyright (c) OpenMMLab. All rights reserved.

import torch

from .triton_kernels import k_grouped_gemm, m_grouped_gemm, m_grouped_gemm_dual_weight


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


class UltraEPGroupedGemm(torch.autograd.Function):
    """Grouped GEMM over inherent and UltraEP replica experts in one launch.

    Replica weights and gradients are runtime-owned, cross-layer buffers.  They
    must not become model parameters or optimizer state.  This bridge therefore
    returns only the inherent-expert weight gradient to autograd and writes the
    replica Wgrad into UltraEP's FP32 buffer as a side effect.

    Xtuner's FSDP-managed master weights and UltraEP's shared replica slots live
    in separate allocations. A dual-base Triton kernel selects the right weight
    allocation per physical expert while retaining one persistent GMM launch.
    Only the replica slots need a forward-time snapshot: the shared runtime slots
    are overwritten by later layers before backward, while master weights retain
    the same lifetime as ordinary Xtuner grouped-linear weights.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        master_weight: torch.Tensor,
        replica_weight: torch.Tensor,
        replica_grad: torch.Tensor,
        tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        empty_input = x.shape[0] == 0
        if empty_input:
            replica_snapshot = replica_weight
            out = torch.matmul(x, master_weight[0].T)
        else:
            # TODO(ultraep): Replace this retained snapshot with a backward-time
            # weight_sync once its FSDP ordering is proven; see
            # docs/design/ultraep_followups.md (UE-3).
            # Runtime slots are shared across layers. Clone only the redundant
            # portion, not all master experts, and use that exact copy for both
            # forward and Dgrad.
            replica_snapshot = replica_weight.clone()
            out = m_grouped_gemm_dual_weight(
                x,
                master_weight,
                replica_snapshot,
                tokens_per_expert,
                trans_b=True,
            )
        ctx.save_for_backward(x, master_weight, replica_snapshot, tokens_per_expert)
        ctx.num_master_experts = master_weight.shape[0]
        ctx.replica_grad = replica_grad
        ctx.empty_input = empty_input
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        x, master_weight, replica_snapshot, tokens_per_expert = ctx.saved_tensors
        num_master_experts = ctx.num_master_experts
        replica_grad = ctx.replica_grad

        if ctx.empty_input:
            dx = torch.matmul(grad_output, master_weight[0])
            master_dw = torch.zeros_like(master_weight)
            replica_grad.zero_()
        else:
            dx = m_grouped_gemm_dual_weight(
                grad_output,
                master_weight,
                replica_snapshot,
                tokens_per_expert,
                trans_b=False,
            )
            physical_dw = k_grouped_gemm(grad_output, x, tokens_per_expert)
            master_dw = physical_dw[:num_master_experts]
            replica_grad.copy_(physical_dw[num_master_experts:].float())
        return dx, master_dw, None, None, None


def ultra_ep_group_gemm(
    x: torch.Tensor,
    master_weight: torch.Tensor,
    replica_weight: torch.Tensor,
    replica_grad: torch.Tensor,
    tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    """Run one grouped GEMM over inherent and redundant physical experts."""
    return UltraEPGroupedGemm.apply(
        x,
        master_weight,
        replica_weight,
        replica_grad,
        tokens_per_expert,
    )
