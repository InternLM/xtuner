# Copyright (c) OpenMMLab. All rights reserved.
"""In-package NPU fused-linear-cross-entropy (FLCE) loss.

Drives liger-ascend's triton kernels directly (no runtime patching of
liger_kernel). The stock ascend plain CE kernel miscomputes ``loss_1d`` in
multi-rank training while its matmul is correct, so the forward here computes
the loss with torch ``F.cross_entropy`` on the matmul logits and never runs
the stock CE kernel. The backward uses the stock no-weight kernel with
``lse = loss_1d + logits[target]`` (``HAS_LSE=False``), keeping liger's fused
backward (no full grad-logits materialisation).

Supports the plain CE surface ``ce_loss.py`` uses: reduction ``'sum'``/
``'none'``, ``ignore_index`` and ``accum_dtype=torch.float32``. Other extras
(bias / ce_weight / label_smoothing / softcap / token scaling) are rejected
loudly instead of silently computing a different loss.
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


@lru_cache(maxsize=1)
def _ascend_ops() -> SimpleNamespace:
    """liger-ascend kernels/helpers by direct import (no module patching)."""
    from liger_kernel.ops.backends._ascend.ops.cross_entropy import (
        _make_ce_stats_buffer,
        liger_cross_entropy_backward_kernel_no_weight,
    )
    from liger_kernel.ops.backends._ascend.ops.fused_linear_cross_entropy import get_optimal_block_size
    from liger_kernel.ops.utils import amp_custom_bwd, amp_custom_fwd, get_npu_core_count

    return SimpleNamespace(
        make_ce_stats_buffer=_make_ce_stats_buffer,
        bwd_kernel_no_weight=liger_cross_entropy_backward_kernel_no_weight,
        get_optimal_block_size=get_optimal_block_size,
        get_npu_core_count=get_npu_core_count,
        amp_custom_fwd=amp_custom_fwd,
        amp_custom_bwd=amp_custom_bwd,
    )


class NPUFusedLinearCrossEntropyFunction(torch.autograd.Function):
    """Fused lm-head matmul + cross-entropy for NPU (plain CE, no extras)."""

    @staticmethod
    @_ascend_ops().amp_custom_fwd
    def forward(ctx, _input, weight, target, ignore_index, reduction):
        bt = _input.shape[0]
        if _CHUNK_FWD_ENABLED and bt * weight.shape[0] * _input.element_size() > _CHUNK_FWD_LIMIT_BYTES:
            loss_1d = torch.empty(bt, dtype=torch.float32, device=_input.device)
            for _s in range(0, bt, _CE_CHUNK_ROWS):
                _e = min(_s + _CE_CHUNK_ROWS, bt)
                _lc = _input[_s:_e] @ weight.t()
                loss_1d[_s:_e] = F.cross_entropy(
                    _lc.float(),
                    target[_s:_e],
                    reduction="none",
                    ignore_index=ignore_index,
                )
            # Empty sentinel: backward recomputes logits per chunk instead of
            # holding the full [BT, V] alive between fwd and bwd.
            logits = torch.empty(0, device=_input.device, dtype=_input.dtype)
        else:
            logits = _input @ weight.t()
            loss_1d = F.cross_entropy(logits.float(), target, reduction="none", ignore_index=ignore_index)
        ops = _ascend_ops()
        ce_stats = ops.make_ce_stats_buffer(
            target,
            ignore_index,
            None,
            reduction,
            target_mask=(target != ignore_index),
        )
        ctx.save_for_backward(_input.detach(), weight.detach(), target.detach(), loss_1d, ce_stats, logits)
        ctx.reduction = reduction
        ctx.ignore_index = ignore_index
        return loss_1d if reduction == "none" else loss_1d.sum()

    @staticmethod
    @_ascend_ops().amp_custom_bwd
    def backward(ctx, grad_output):
        ops = _ascend_ops()
        (_input, weight, target, loss_1d, ce_stats, saved_logits) = ctx.saved_tensors
        bt = _input.shape[0]
        v = weight.shape[0]

        backward_block_size = ops.get_optimal_block_size(v, has_gradients=True)
        if 32768 < v <= 131072:  # plain-CE tier of the stock ascend backward tuning
            backward_block_size = 4096

        has_saved_logits = saved_logits.numel() != 0
        chunk_size = bt if has_saved_logits else min(bt, 4096)
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

        for chunk_id in range(num_chunks):
            start_idx = chunk_id * chunk_size
            end_idx = min(start_idx + chunk_size, bt)
            input_chunk = _input[start_idx:end_idx]
            target_chunk = target[start_idx:end_idx]
            n_rows = end_idx - start_idx

            if has_saved_logits:
                logits_chunk = saved_logits[start_idx:end_idx]
            else:
                logits_chunk = input_chunk @ weight.t()

            if not logits_chunk.is_contiguous():
                logits_chunk = logits_chunk.contiguous()
            if not target_chunk.is_contiguous():
                target_chunk = target_chunk.contiguous()

            grad_logits_chunk = torch.empty_like(logits_chunk)

            loss_1d_slice = loss_1d[start_idx:end_idx]
            ops.bwd_kernel_no_weight[(min(n_rows, num_cores),)](
                X_ptr=logits_chunk,
                X_stride=logits_chunk.stride(-2),
                Y_ptr=target_chunk,
                lse_ptr=loss_1d_slice,
                grad_output_ptr=grad_output,
                grad_output_stride=grad_output_stride,
                dX_ptr=grad_logits_chunk,
                dX_stride=grad_logits_chunk.stride(-2),
                n_cols=v,
                n_rows=n_rows,
                ce_stats_ptr=ce_stats,
                ignore_index=ctx.ignore_index,
                reduction=ctx.reduction,
                BLOCK_SIZE=backward_block_size,
                HAS_LSE=False,
            )

            grad_input[start_idx:end_idx] = grad_logits_chunk @ weight
            # Ascend's matmul(out=) does not auto-cast and overwrites rather
            # than accumulates, so it is only used for the single-chunk case;
            # multi-chunk goes through the temp + copy_/add_ path (chunk 0
            # must copy_ into the uninitialized buffer).
            if num_chunks == 1:
                torch.matmul(grad_logits_chunk.t(), input_chunk, out=grad_weight)
            else:
                grad_weight_ = grad_logits_chunk.t() @ input_chunk
                if chunk_id == 0:
                    grad_weight.copy_(grad_weight_)
                else:
                    grad_weight.add_(grad_weight_)

        return grad_input, grad_weight, None, None, None


class NPUFusedLinearCrossEntropyLoss(nn.Module):
    """Drop-in for liger_kernel's ``LigerFusedLinearCrossEntropyLoss`` on NPU.

    Same call convention: ``forward(lin_weight, _input, target)`` computes
    ``logits = _input @ lin_weight.t()`` fused with cross-entropy.
    """

    def __init__(self, reduction: str = "mean", accum_dtype: torch.dtype | None = None, ignore_index: int = -100):
        super().__init__()
        if reduction not in ("sum", "none"):
            raise NotImplementedError(f"NPU FLCE supports reduction 'sum'/'none', got {reduction!r}")
        if accum_dtype is not None and accum_dtype is not torch.float32:
            raise NotImplementedError(f"NPU FLCE supports accum_dtype=torch.float32, got {accum_dtype!r}")
        self.reduction = reduction
        self.accum_dtype = accum_dtype
        self.ignore_index = ignore_index

    def forward(self, lin_weight, _input, target):
        return NPUFusedLinearCrossEntropyFunction.apply(_input, lin_weight, target, self.ignore_index, self.reduction)
