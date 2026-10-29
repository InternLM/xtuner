# Copyright (c) OpenMMLab. All rights reserved.
"""In-package NPU fused-linear-cross-entropy (FLCE) loss.

Drives liger-ascend's triton kernels directly (no runtime patching of
liger_kernel). The stock ascend plain CE kernel miscomputes ``loss_1d`` in
multi-rank training while its matmul is correct, so the forward here computes
the loss with torch ``F.cross_entropy`` on the matmul logits and never runs
the stock CE kernel. The backward uses the stock no-weight kernel with
``lse = loss_1d + logits[target]`` (``HAS_LSE=False``), keeping liger's fused
backward (no full grad-logits materialisation).

Supports the CE surface ``ce_loss.py`` uses: reduction ``'sum'``/
``'none'``, ``ignore_index``, ``accum_dtype=torch.float32``, and LM
z-loss. With z-loss enabled, the pinned Liger general CE kernel computes CE,
LSE-squared, and their combined logits gradient in one fused kernel over each
memory-bounded logits chunk. The z-loss kernels are imported optionally so the
module still loads on liger builds without them (requesting z-loss there
raises ``NotImplementedError``). Other extras (bias / ce_weight /
label_smoothing / softcap / token scaling) are rejected loudly instead of
silently computing a different loss.
"""

import os
from functools import lru_cache
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import triton
from torch import nn


# Chunked lm-head+CE forward: when the full ``[BT, V]`` logits would be
# large, compute the loss in BT-chunks so only a ``[chunk, V]`` slice is
# live at once. CE is row-independent, so the chunked ``loss_1d`` is
# identical to the full path, and the backward recomputes logits per chunk,
# so grads match too. Gate: full-logits bytes > limit (default 2048 MiB).
# LOSS_CHUNK_SIZE is the same knob that feeds CELossConfig.chunk_size in the
# config layer; XTUNER_LIGER_CHUNK_FWD_LIMIT is in MiB.
_CHUNK_FWD_ENABLED = os.environ.get("XTUNER_LIGER_CHUNK_FWD", "1") == "1"
_CHUNK_FWD_LIMIT_BYTES = int(os.environ.get("XTUNER_LIGER_CHUNK_FWD_LIMIT", "2048")) * 1024**2
_CE_CHUNK_ROWS = int(os.environ.get("LOSS_CHUNK_SIZE", "1024"))
_GRAD_WEIGHT_CHUNK_ROWS = int(os.environ.get("XTUNER_LIGER_GRAD_WEIGHT_CHUNK", "4096"))


@lru_cache(maxsize=1)
def _ascend_ops() -> SimpleNamespace:
    """liger-ascend kernels/helpers by direct import (no module patching)."""
    from liger_kernel.ops.backends._ascend.ops.cross_entropy import (
        _make_ce_stats_buffer,
        liger_cross_entropy_backward_kernel_no_weight,
    )
    from liger_kernel.ops.backends._ascend.ops.fused_linear_cross_entropy import get_optimal_block_size
    from liger_kernel.ops.utils import amp_custom_bwd, amp_custom_fwd, get_npu_core_count

    # The fused CE+z-loss kernels postdate some liger builds; import them
    # optionally so this module still loads where only the plain-CE kernels
    # exist (requesting z-loss there raises a clear NotImplementedError).
    try:
        from liger_kernel.ops.backends._ascend.ops.cross_entropy import (
            liger_cross_entropy_backward_kernel,
            liger_cross_entropy_forward_kernel,
        )
    except ImportError:
        liger_cross_entropy_backward_kernel = None  # type: ignore[assignment]
        liger_cross_entropy_forward_kernel = None  # type: ignore[assignment]

    # liger_kernel >= 0.8.5 requires the HAS_TAIL constexpr on the no-weight
    # backward kernel (masks the ragged last BLOCK_SIZE column chunk, same
    # semantics as the in-package _plain_ce_need_mask); older builds reject it.
    _bwd_params = {p.name for p in liger_cross_entropy_backward_kernel_no_weight.params}

    def _has_tail_kwargs(n_cols: int, block_size: int) -> dict:
        return {"HAS_TAIL": (n_cols % block_size) != 0} if "HAS_TAIL" in _bwd_params else {}

    return SimpleNamespace(
        make_ce_stats_buffer=_make_ce_stats_buffer,
        fwd_kernel=liger_cross_entropy_forward_kernel,
        bwd_kernel=liger_cross_entropy_backward_kernel,
        bwd_kernel_no_weight=liger_cross_entropy_backward_kernel_no_weight,
        get_optimal_block_size=get_optimal_block_size,
        get_npu_core_count=get_npu_core_count,
        amp_custom_fwd=amp_custom_fwd,
        amp_custom_bwd=amp_custom_bwd,
        has_tail_kwargs=_has_tail_kwargs,
    )


def _launch_fused_ce_z_forward(
    logits: torch.Tensor,
    target: torch.Tensor,
    loss_1d: torch.Tensor,
    z_loss_1d: torch.Tensor,
    lse_1d: torch.Tensor,
    ce_stats: torch.Tensor,
    ignore_index: int,
    lse_square_scale: float,
) -> None:
    """Compute CE and z-loss together from an already materialized logits chunk.

    Args:
        logits (torch.Tensor): The ``[n_rows, vocab]`` logits chunk.
        target (torch.Tensor): The ``[n_rows]`` target token ids.
        loss_1d (torch.Tensor): The ``[n_rows]`` per-row CE output buffer.
        z_loss_1d (torch.Tensor): The ``[n_rows]`` per-row z-loss output buffer.
        lse_1d (torch.Tensor): The ``[n_rows]`` per-row log-sum-exp output buffer.
        ce_stats (torch.Tensor): The statistics buffer shared by the kernel family.
        ignore_index (int): The target id to ignore in the loss computation.
        lse_square_scale (float): The coefficient of the squared-LSE z-loss.

    Return:
        None
    """
    ops = _ascend_ops()
    n_rows, vocab_size = logits.shape
    if not logits.is_contiguous():
        logits = logits.contiguous()
    if not target.is_contiguous():
        target = target.contiguous()
    block_size = ops.get_optimal_block_size(vocab_size, has_gradients=False)
    ops.fwd_kernel[(min(n_rows, ops.get_npu_core_count()),)](
        X_ptr=logits,
        X_stride=logits.stride(-2),
        Y_ptr=target,
        weight_ptr=None,
        loss_ptr=loss_1d,
        z_loss_ptr=z_loss_1d,
        lse_ptr=lse_1d,
        token_accuracy_ptr=loss_1d,
        token_accuracy_stride=0,
        predicted_tokens_ptr=target,
        predicted_tokens_stride=0,
        n_cols=vocab_size,
        n_rows=n_rows,
        ce_stats_ptr=ce_stats,
        ignore_index=ignore_index,
        ls_eps=0.0,
        lse_square_scale=lse_square_scale,
        label_smoothing=0.0,
        reduction="sum",
        softcap=None,
        RETURN_Z_LOSS=True,
        RETURN_LSE=True,
        RETURN_TOKEN_ACCURACY=False,
        RETURN_PREDICTED_TOKENS=False,
        HAS_WEIGHT=False,
        HAS_SOFTCAPPING=False,
        BLOCK_SIZE=block_size,
    )


class NPUFusedLinearCrossEntropyFunction(torch.autograd.Function):
    """Fused lm-head matmul plus CE, with optional fused LM z-loss."""

    @staticmethod
    @_ascend_ops().amp_custom_fwd
    def forward(
        ctx,
        _input,
        weight,
        target,
        ignore_index,
        reduction,
        lse_square_scale,
        return_z_loss,
    ):
        bt = _input.shape[0]
        use_z_loss = float(lse_square_scale) > 0.0
        loss_1d = torch.zeros(bt, dtype=torch.float32, device=_input.device)
        z_loss_1d = torch.zeros(bt, dtype=torch.float32, device=_input.device)
        lse_1d = (
            torch.empty(bt, dtype=torch.float32, device=_input.device)
            if use_z_loss
            else torch.empty(0, dtype=torch.float32, device=_input.device)
        )
        ops = _ascend_ops()
        ce_stats = ops.make_ce_stats_buffer(
            target,
            ignore_index,
            None,
            reduction,
            target_mask=(target != ignore_index),
        )

        chunk_forward = _CHUNK_FWD_ENABLED and bt * weight.shape[0] * _input.element_size() > _CHUNK_FWD_LIMIT_BYTES
        if chunk_forward:
            for start_idx in range(0, bt, _CE_CHUNK_ROWS):
                end_idx = min(start_idx + _CE_CHUNK_ROWS, bt)
                logits_chunk = _input[start_idx:end_idx] @ weight.t()
                if use_z_loss:
                    _launch_fused_ce_z_forward(
                        logits_chunk,
                        target[start_idx:end_idx],
                        loss_1d[start_idx:end_idx],
                        z_loss_1d[start_idx:end_idx],
                        lse_1d[start_idx:end_idx],
                        ce_stats,
                        ignore_index,
                        float(lse_square_scale),
                    )
                else:
                    loss_1d[start_idx:end_idx] = F.cross_entropy(
                        logits_chunk.float(),
                        target[start_idx:end_idx],
                        reduction="none",
                        ignore_index=ignore_index,
                    )
            # Empty sentinel: backward recomputes logits per chunk instead of
            # holding the full [BT, V] alive between fwd and bwd.
            logits = torch.empty(0, device=_input.device, dtype=_input.dtype)
        else:
            logits = _input @ weight.t()
            if use_z_loss:
                _launch_fused_ce_z_forward(
                    logits,
                    target,
                    loss_1d,
                    z_loss_1d,
                    lse_1d,
                    ce_stats,
                    ignore_index,
                    float(lse_square_scale),
                )
            else:
                loss_1d = F.cross_entropy(logits.float(), target, reduction="none", ignore_index=ignore_index)

        ctx.save_for_backward(
            _input.detach(),
            weight.detach(),
            target.detach(),
            loss_1d,
            ce_stats,
            logits,
            lse_1d,
        )
        ctx.reduction = reduction
        ctx.ignore_index = ignore_index
        ctx.lse_square_scale = float(lse_square_scale)
        ctx.return_z_loss = bool(return_z_loss)

        loss = loss_1d if reduction == "none" else loss_1d.sum()
        z_loss = z_loss_1d if reduction == "none" else z_loss_1d.sum()
        return loss, z_loss

    @staticmethod
    @_ascend_ops().amp_custom_bwd
    def backward(ctx, grad_output, grad_z_output):
        del grad_z_output
        ops = _ascend_ops()
        (_input, weight, target, loss_1d, ce_stats, saved_logits, lse_1d) = ctx.saved_tensors
        bt = _input.shape[0]
        vocab_size = weight.shape[0]
        use_z_loss = ctx.lse_square_scale > 0.0

        backward_block_size = ops.get_optimal_block_size(vocab_size, has_gradients=True)
        if not use_z_loss and 32768 < vocab_size <= 131072:  # plain-CE tier of the stock ascend backward tuning
            backward_block_size = 4096

        has_saved_logits = saved_logits.numel() != 0
        chunk_size = bt if has_saved_logits else min(bt, _CE_CHUNK_ROWS)
        num_chunks = triton.cdiv(bt, chunk_size)
        num_cores = ops.get_npu_core_count()

        # bf16 grad_weight in the multi-chunk (recompute) path: an fp32
        # accumulator would be a large persistent allocation. Trade: each
        # per-chunk fp32 partial is rounded to bf16 before the bf16 ``add_``
        # summation; grad_input is unaffected (chunks write disjoint rows).
        grad_input = torch.empty_like(_input)
        grad_weight = torch.empty_like(weight)

        has_grad_output_vector = ctx.reduction == "none"
        if has_grad_output_vector and grad_output.stride(-1) != 1:
            grad_output = grad_output.contiguous()
        grad_output_stride = grad_output.stride(-1) if has_grad_output_vector else 0

        # Chunk-loop scratch buffers. Every chunk's recomputed logits and grad-logits are fully consumed inside the
        # same iteration (bwd kernel reads X / writes dX, then the grad_input/grad_weight matmuls read dX), so one
        # preallocated vocab-wide buffer serves all chunks. A fresh empty_like per chunk costs a vocab x chunk
        # allocation (~1.9 GiB at vocab 248k) on every iteration and was the largest transient at the memory peak.
        recomputed_logits_scratch: torch.Tensor | None = None
        grad_logits_scratch: torch.Tensor | None = None

        for chunk_id in range(num_chunks):
            start_idx = chunk_id * chunk_size
            end_idx = min(start_idx + chunk_size, bt)
            input_chunk = _input[start_idx:end_idx]
            target_chunk = target[start_idx:end_idx]
            n_rows = end_idx - start_idx

            if has_saved_logits:
                logits_chunk = saved_logits[start_idx:end_idx]
            elif recomputed_logits_scratch is not None:
                logits_chunk = recomputed_logits_scratch[:n_rows]
                torch.mm(input_chunk, weight.t(), out=logits_chunk)
            else:
                logits_chunk = input_chunk @ weight.t()
                if num_chunks > 1:
                    recomputed_logits_scratch = torch.empty_like(logits_chunk)

            if not logits_chunk.is_contiguous():
                logits_chunk = logits_chunk.contiguous()
            if not target_chunk.is_contiguous():
                target_chunk = target_chunk.contiguous()

            if grad_logits_scratch is not None:
                grad_logits_chunk = grad_logits_scratch[:n_rows]
            else:
                grad_logits_chunk = torch.empty_like(logits_chunk)
                if num_chunks > 1:
                    grad_logits_scratch = grad_logits_chunk
            loss_1d_slice = loss_1d[start_idx:end_idx]
            grad_output_chunk = grad_output[start_idx:end_idx] if has_grad_output_vector else grad_output

            if use_z_loss:
                ops.bwd_kernel[(min(n_rows, num_cores),)](
                    X_ptr=logits_chunk,
                    X_stride=logits_chunk.stride(-2),
                    Y_ptr=target_chunk,
                    weight_ptr=None,
                    lse_ptr=lse_1d[start_idx:end_idx],
                    grad_output_ptr=grad_output_chunk,
                    grad_output_stride=grad_output_stride,
                    dX_ptr=grad_logits_chunk,
                    dX_stride=grad_logits_chunk.stride(-2),
                    n_cols=vocab_size,
                    n_rows=n_rows,
                    ce_stats_ptr=ce_stats,
                    ignore_index=ctx.ignore_index,
                    lse_square_scale=ctx.lse_square_scale,
                    label_smoothing=0.0,
                    reduction=ctx.reduction,
                    softcap=None,
                    BLOCK_SIZE=backward_block_size,
                    HAS_WEIGHT=False,
                    HAS_SOFTCAPPING=False,
                )
            else:
                ops.bwd_kernel_no_weight[(min(n_rows, num_cores),)](
                    X_ptr=logits_chunk,
                    X_stride=logits_chunk.stride(-2),
                    Y_ptr=target_chunk,
                    lse_ptr=loss_1d_slice,
                    grad_output_ptr=grad_output_chunk,
                    grad_output_stride=grad_output_stride,
                    dX_ptr=grad_logits_chunk,
                    dX_stride=grad_logits_chunk.stride(-2),
                    n_cols=vocab_size,
                    n_rows=n_rows,
                    ce_stats_ptr=ce_stats,
                    ignore_index=ctx.ignore_index,
                    reduction=ctx.reduction,
                    BLOCK_SIZE=backward_block_size,
                    HAS_LSE=False,
                    **ops.has_tail_kwargs(vocab_size, backward_block_size),
                )

            grad_input[start_idx:end_idx] = grad_logits_chunk @ weight
            # A full [vocab, hidden] temporary costs about 2 GiB for large
            # checkpoints. Accumulate vocabulary slabs instead. Per output
            # element the row-chunk accumulation order remains unchanged.
            for vocab_start in range(0, vocab_size, _GRAD_WEIGHT_CHUNK_ROWS):
                vocab_end = min(vocab_start + _GRAD_WEIGHT_CHUNK_ROWS, vocab_size)
                grad_weight_slab = grad_logits_chunk[:, vocab_start:vocab_end].t().contiguous() @ input_chunk
                if chunk_id == 0:
                    grad_weight[vocab_start:vocab_end].copy_(grad_weight_slab)
                else:
                    grad_weight[vocab_start:vocab_end].add_(grad_weight_slab)

        return grad_input, grad_weight, None, None, None, None, None


class NPUFusedLinearCrossEntropyLoss(nn.Module):
    """NPU FLCE with optional z-loss fused into the CE kernels.

    Drop-in for liger_kernel's ``LigerFusedLinearCrossEntropyLoss`` on NPU:
    ``forward(lin_weight, _input, target)`` computes
    ``logits = _input @ lin_weight.t()`` fused with cross-entropy.
    """

    def __init__(
        self,
        reduction: str = "mean",
        accum_dtype: torch.dtype | None = None,
        ignore_index: int = -100,
        lse_square_scale: float = 0.0,
        return_z_loss: bool = False,
    ):
        super().__init__()
        if reduction not in ("sum", "none"):
            raise NotImplementedError(f"NPU FLCE supports reduction 'sum'/'none', got {reduction!r}")
        if accum_dtype is not None and accum_dtype is not torch.float32:
            raise NotImplementedError(f"NPU FLCE supports accum_dtype=torch.float32, got {accum_dtype!r}")
        if lse_square_scale < 0:
            raise ValueError("lse_square_scale must be non-negative")
        if lse_square_scale > 0 and _ascend_ops().fwd_kernel is None:
            raise NotImplementedError("LM z-loss needs liger kernels with z-loss support (liger_kernel >= 0.8.4)")
        self.reduction = reduction
        self.accum_dtype = accum_dtype
        self.ignore_index = ignore_index
        self.lse_square_scale = float(lse_square_scale)
        self.return_z_loss = bool(return_z_loss)

    def forward(self, lin_weight, _input, target):
        loss, z_loss = NPUFusedLinearCrossEntropyFunction.apply(
            _input,
            lin_weight,
            target,
            self.ignore_index,
            self.reduction,
            self.lse_square_scale,
            self.return_z_loss,
        )
        if self.return_z_loss:
            return loss, z_loss
        return loss
