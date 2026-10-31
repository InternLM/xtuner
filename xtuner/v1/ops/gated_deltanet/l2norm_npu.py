"""Runtime-length l2norm triton kernels for the NPU GDN path.

fla_npu's blocked l2norm kernels declare the row count ``T`` as
``tl.constexpr``, so every new row count forces a fresh seconds-long bisheng
compile.  In training the GDN input row count grows by a few padded tokens
whenever a megabatch contains a short document, so a handful of ranks hit a
never-before-seen ``T`` mid-run and stall the whole job at the next
collective.  These kernels are the fla_npu blocked l2norm with ``T``/``NB``
passed as runtime arguments instead: one binary serves every row count and
the mid-run recompile class disappears.  The kernel bodies are unchanged:
the bounds guard ``if block_start < T`` and the block-pointer
``boundary_check`` keep every access in-bounds for any ``T`` (the
backward's value domain remains row counts that are multiples of ``BT``,
as in fla_npu).
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T"])
def l2norm_fwd_kernel(
    x,
    y,
    rstd,
    eps,
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
    BT: tl.constexpr,
    bt_size,
):
    i_t = tl.program_id(0)
    for offset in range(0, bt_size):
        block_start = (i_t * bt_size + offset) * BT
        if block_start < T:
            p_x = tl.make_block_ptr(x, (T, D), (D, 1), (block_start, 0), (BT, BD), (1, 0))
            p_y = tl.make_block_ptr(y, (T, D), (D, 1), (block_start, 0), (BT, BD), (1, 0))
            p_rstd = tl.make_block_ptr(rstd, (T,), (1,), (block_start,), (BT,), (0,))

            b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
            b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x, 1) + eps)
            b_y = b_x * b_rstd[:, None]

            tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_rstd, b_rstd.to(p_rstd.dtype.element_ty), boundary_check=(0,))


@triton.jit(do_not_specialize=["T"])
def l2norm_bwd_kernel(
    y,
    rstd,
    dy,
    dx,
    eps,
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
    BT: tl.constexpr,
    bt_size,
):
    i_t_start = tl.program_id(0)
    num_blocks = bt_size

    total_i_t = T // BT
    base_tasks_per_block = total_i_t // num_blocks
    remainder_tasks = total_i_t % num_blocks

    if i_t_start < remainder_tasks:
        tasks_this_block = base_tasks_per_block + 1
        start_i_t = i_t_start * tasks_this_block
    else:
        tasks_this_block = base_tasks_per_block
        start_i_t = i_t_start * base_tasks_per_block + remainder_tasks

    for task_idx in range(tasks_this_block):
        i_t = start_i_t + task_idx
        block_start = i_t * BT
        if block_start < T:
            p_y = tl.make_block_ptr(y, (T, D), (D, 1), (block_start, 0), (BT, BD), (1, 0))
            p_rstd = tl.make_block_ptr(rstd, (T,), (1,), (block_start,), (BT,), (0,))
            p_dy = tl.make_block_ptr(dy, (T, D), (D, 1), (block_start, 0), (BT, BD), (1, 0))
            p_dx = tl.make_block_ptr(dx, (T, D), (D, 1), (block_start, 0), (BT, BD), (1, 0))

            b_y = tl.load(p_y, boundary_check=(0, 1)).to(tl.float32)
            b_rstd = tl.load(p_rstd, boundary_check=(0,)).to(tl.float32)
            b_dy = tl.load(p_dy, boundary_check=(0, 1)).to(tl.float32)
            b_dx = b_dy * b_rstd[:, None] - tl.sum(b_dy * b_y, 1)[:, None] * b_y * b_rstd[:, None]
            tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), boundary_check=(0, 1))


def _fla_l2norm_fwd(x: torch.Tensor, eps: float, output_dtype: Optional[torch.dtype]):
    from fla_npu.ops.triton import l2norm_fwd as fla_l2norm_fwd

    return fla_l2norm_fwd(x, eps=eps, output_dtype=output_dtype)


def _fla_l2norm_bwd(y: torch.Tensor, rstd: torch.Tensor, dy: torch.Tensor, eps: float):
    from fla_npu.ops.triton import l2norm_bwd as fla_l2norm_bwd

    return fla_l2norm_bwd(y, rstd, dy, eps=eps)


def l2norm_fwd(
    x: torch.Tensor,
    eps: float = 1e-6,
    output_dtype: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward l2 normalization over the last dimension.

    Row counts are runtime arguments, so no row count triggers a recompile.
    Feature dims above 512 delegate to fla_npu's per-row kernel, which takes
    no row-count argument at all.

    Args:
        x (torch.Tensor): Input of shape ``(..., D)`` with ``D <= 512`` on the
            fast path; the last dimension is normalized.
        eps (float): Epsilon added to the squared norm. Defaults to ``1e-6``.
        output_dtype (torch.dtype | None): Dtype of the normalized output;
            ``None`` keeps the input dtype. Defaults to ``None``.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Normalized output in the input
        shape and the per-row reciprocal standard deviation in float32.
    """
    x_shape_og = x.shape
    if x.shape[-1] > 512:
        return _fla_l2norm_fwd(x, eps, output_dtype)
    x = x.view(-1, x.shape[-1])
    if output_dtype is None:
        y = torch.empty_like(x)
    else:
        y = torch.empty_like(x, dtype=output_dtype)
    T, D = x.shape[0], x.shape[-1]
    max_fused_size = 65536 // x.element_size()
    BD = min(max_fused_size, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer doesn't support feature dim >= 64KB.")

    rstd = torch.empty((T,), dtype=torch.float32, device=x.device)
    bt_size = 32
    grid = (triton.cdiv(T, 16 * bt_size),)
    l2norm_fwd_kernel[grid](
        x,
        y,
        rstd,
        eps,
        T,
        D=D,
        BD=BD,
        BT=16,
        bt_size=bt_size,
        num_warps=4,
    )
    return y.view(x_shape_og), rstd.view(x_shape_og[:-1])


def l2norm_bwd(
    y: torch.Tensor,
    rstd: torch.Tensor,
    dy: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Backward of :func:`l2norm_fwd` given the saved forward outputs.

    Args:
        y (torch.Tensor): Normalized output saved from the forward pass.
        rstd (torch.Tensor): Per-row reciprocal standard deviation saved from
            the forward pass.
        dy (torch.Tensor): Gradient w.r.t. the normalized output.
        eps (float): Epsilon used in the forward pass. Defaults to ``1e-6``.

    Returns:
        torch.Tensor: Gradient w.r.t. the l2norm input, same shape as ``y``.
    """
    y_shape_og = y.shape
    if y.shape[-1] > 512:
        return _fla_l2norm_bwd(y, rstd, dy, eps)
    y = y.view(-1, dy.shape[-1])
    dy = dy.view(-1, dy.shape[-1])
    dx = torch.empty_like(y)
    T, D = y.shape[0], y.shape[-1]
    max_fused_size = 65536 // y.element_size()
    BD = min(max_fused_size, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    bt_size = 40
    l2norm_bwd_kernel[(bt_size,)](
        y,
        rstd,
        dy,
        dx,
        eps,
        T,
        D=D,
        BD=BD,
        BT=16,
        bt_size=bt_size,
        num_warps=4,
    )
    return dx.view(y_shape_og)


class L2NormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, eps: float = 1e-6, output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        y, rstd = l2norm_fwd(x, eps, output_dtype)
        ctx.eps = eps
        ctx.x_dtype = x.dtype
        ctx.save_for_backward(y, rstd)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        y, rstd = ctx.saved_tensors
        dx = l2norm_bwd(y, rstd, dy.contiguous(), ctx.eps)
        return dx, None, None


def l2norm(x: torch.Tensor, eps: float = 1e-6, output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """l2-normalize the last dimension with a runtime-length triton kernel.

    Autograd-compatible drop-in for ``fla_npu.ops.triton.l2norm.l2norm``.

    Args:
        x (torch.Tensor): Input of shape ``(..., D)``; normalized along ``D``.
        eps (float): Epsilon added to the squared norm. Defaults to ``1e-6``.
        output_dtype (torch.dtype | None): Dtype of the output; ``None`` keeps
            the input dtype. Defaults to ``None``.

    Returns:
        torch.Tensor: Normalized tensor, same shape as ``x``.
    """
    return L2NormFunction.apply(x.contiguous(), eps, output_dtype)
