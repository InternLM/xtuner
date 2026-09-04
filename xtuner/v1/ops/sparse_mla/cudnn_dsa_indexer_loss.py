# Copyright (c) OpenMMLab. All rights reserved.
"""cuDNN sparse DSA indexer distillation loss.

The public loss uses XTuner-owned sum reduction semantics.  cuDNN score
recompute produces the FP32 teacher/prediction distributions, while an opaque
manual-autograd operator routes the KL gradient only to indexer Q/K/weights.
"""

from __future__ import annotations

import torch
from torch import Tensor

from xtuner.v1.utils import log_rank0


_CUDNN_INDEXER_BACKWARD_MIN_HEADS = 64
_CUDNN_INDEXER_BACKWARD_BLOCK_I = 128
_INDEXER_LOSS_DEBUG_CALLS: dict[str, int] = {}


def _copy_aligned_grad_loss(grad_loss: Tensor, device: torch.device) -> Tensor:
    """Copy an autograd scalar into a fresh 16-byte-aligned FP32 allocation."""

    if grad_loss.numel() != 1:
        raise ValueError(f"grad_loss must contain exactly one element, got shape {tuple(grad_loss.shape)}")

    aligned_grad_loss = torch.empty(1, dtype=torch.float32, device=device)
    aligned_grad_loss.copy_(grad_loss.detach().to(device=device, dtype=torch.float32).reshape(1))
    return aligned_grad_loss


def _aligned_contiguous(tensor: Tensor) -> Tensor:
    """Return a contiguous tensor whose actual data pointer is 16-byte aligned."""

    # The kernel contract is addr(tensor) mod 16 = 0; stride contiguity alone
    # does not imply this when the tensor is a storage-offset view.
    tensor = tensor.contiguous()
    if tensor.data_ptr() % 16 != 0:
        tensor = tensor.clone()
    return tensor


def _pad_indexer_heads_for_cudnn(index_q: Tensor, index_weights: Tensor) -> tuple[Tensor, Tensor, int]:
    """Pad sub-64-head indexer inputs without changing their score function."""

    index_heads = index_q.shape[-2]
    if index_weights.shape[-1] != index_heads:
        raise ValueError(
            "index_q and index_weights must have the same number of index heads, "
            f"got {index_heads} and {index_weights.shape[-1]}"
        )
    if index_heads >= _CUDNN_INDEXER_BACKWARD_MIN_HEADS:
        return index_q, index_weights, index_heads

    # cudnn要求有>=64个head，但是glm 5.2的indexer只有32个，所以要pad到64个
    # For H' = 64, extend q'_h = q_h and w'_h = w_h for h <= H, and set
    # q'_h = w'_h = 0 for H < h <= H'.  With
    #   score(q, k, w) = sum_{h=1}^{H} w_h * ReLU(<q_h, k>),
    # the padded terms are zero, so score(q', k, w') = score(q, k, w).
    padded_heads = _CUDNN_INDEXER_BACKWARD_MIN_HEADS - index_heads
    return (
        torch.nn.functional.pad(index_q, (0, 0, 0, padded_heads)),
        torch.nn.functional.pad(index_weights, (0, padded_heads)),
        index_heads,
    )


def _prepare_sparse_topk(
    topk_indices: Tensor,
    topk_length: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    if topk_indices.ndim != 3:
        raise ValueError(f"topk_indices must have shape (B, S_q, K), got {tuple(topk_indices.shape)}")

    valid_slots = topk_indices != -1
    if topk_length is None:
        topk_length = valid_slots.sum(dim=-1, dtype=torch.int32)
    elif topk_length.shape != topk_indices.shape[:2]:
        raise ValueError(
            f"topk_length must have shape {tuple(topk_indices.shape[:2])}, got {tuple(topk_length.shape)}"
        )

    safe_topk = topk_indices.clamp_min(0).to(dtype=torch.int32).contiguous()
    return safe_topk, topk_length.to(dtype=torch.int32).contiguous(), valid_slots


def _validate_distribution_shapes(
    target: Tensor,
    predict: Tensor,
    topk_indices: Tensor,
) -> None:
    if target.shape != predict.shape:
        raise ValueError(f"target and predict must have the same shape, got {target.shape} and {predict.shape}")
    if target.shape != topk_indices.shape:
        raise ValueError(f"target/predict must match topk_indices shape {topk_indices.shape}, got {target.shape}")


def _mask_invalid_query_rows(
    target: Tensor,
    predict: Tensor,
    topk_indices: Tensor,
    valid_query_mask: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Remove padded query rows from both the KL value and its manual backward.

    Packed SFT batches keep a fixed physical sequence length.  Their tail
    padding is represented as one or more causal chunks, so the top-k kernel
    still returns non-negative indices for those query rows.  Consequently,
    ``topk_indices != -1`` alone cannot distinguish real queries from padding.

    For a padding row, setting ``target=predict=0`` makes cuDNN's score-gradient
    signal zero, while setting ``topk_indices=-1`` makes XTuner's public KL
    reduction exclude the same row.  Keeping the physical tensor shapes intact
    also avoids compiling a new cuDNN kernel for every effective sequence
    length.
    """

    if valid_query_mask is None:
        return target, predict, topk_indices
    expected_shape = topk_indices.shape[:-1]
    if valid_query_mask.shape != expected_shape:
        raise ValueError(
            f"valid_query_mask must have shape {tuple(expected_shape)}, got {tuple(valid_query_mask.shape)}"
        )

    valid_query_mask = valid_query_mask.to(device=topk_indices.device, dtype=torch.bool)
    invalid_rows = ~valid_query_mask.unsqueeze(-1)
    return (
        target.masked_fill(invalid_rows, 0.0),
        predict.masked_fill(invalid_rows, 0.0),
        topk_indices.masked_fill(invalid_rows, -1),
    )


@torch.no_grad()
def _maybe_log_indexer_loss_diagnostics(
    target: Tensor,
    predict: Tensor,
    topk_indices: Tensor,
    *,
    debug_name: str | None,
    debug_interval: int,
) -> None:
    r"""Log compact distribution diagnostics on rank 0 at a fixed interval.

    For every valid query row, the reported quantities include

    .. math::

       H(p)=-\sum_i p_i\log p_i,\qquad
       D_{KL}(p\|q)=\sum_i p_i\log\frac{p_i}{q_i},

    where ``p`` is the attention teacher and ``q`` is the indexer prediction.
    ``top1_match`` is the mean indicator
    :math:`\mathbb{1}[\arg\max p=\arg\max q]`.
    """

    if debug_name is None or debug_interval <= 0:
        return
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return

    call = _INDEXER_LOSS_DEBUG_CALLS.get(debug_name, 0) + 1
    _INDEXER_LOSS_DEBUG_CALLS[debug_name] = call
    if call != 1 and call % debug_interval != 0:
        return

    target_f32 = target.float()
    predict_f32 = predict.float()
    valid_slots = topk_indices != -1
    valid_rows = valid_slots.any(dim=-1)
    num_valid_rows = int(valid_rows.sum().item())
    if num_valid_rows == 0:
        log_rank0.info(
            f"[DSA_INDEXER_LOSS] name={debug_name} call={call} valid_rows=0 (distribution diagnostics skipped)"
        )
        return
    target_entropy = -torch.special.xlogy(target_f32, target_f32).sum(dim=-1)
    predict_entropy = -torch.special.xlogy(predict_f32, predict_f32).sum(dim=-1)
    per_row_kl = torch.special.xlogy(target_f32, target_f32).sum(dim=-1) - torch.special.xlogy(
        target_f32, predict_f32
    ).sum(dim=-1)
    target_top1 = target_f32.argmax(dim=-1)
    predict_top1 = predict_f32.argmax(dim=-1)
    target_top1_predict = predict_f32.gather(dim=-1, index=target_top1.unsqueeze(-1)).squeeze(-1)

    valid_target_entropy = target_entropy[valid_rows]
    valid_predict_entropy = predict_entropy[valid_rows]
    valid_kl = per_row_kl[valid_rows]
    valid_target_max = target_f32.amax(dim=-1)[valid_rows]
    valid_predict_max = predict_f32.amax(dim=-1)[valid_rows]
    valid_target_top1_predict = target_top1_predict[valid_rows]
    valid_top1_match = (target_top1 == predict_top1)[valid_rows].float()
    mean_topk = valid_slots.sum(dim=-1, dtype=torch.float32)[valid_rows].mean()

    log_rank0.info(
        "[DSA_INDEXER_LOSS] "
        f"name={debug_name} call={call} valid_rows={num_valid_rows} "
        f"mean_topk={mean_topk.item():.2f} kl_mean={valid_kl.mean().item():.6f} "
        f"kl_max={valid_kl.max().item():.6f} target_entropy={valid_target_entropy.mean().item():.6f} "
        f"predict_entropy={valid_predict_entropy.mean().item():.6f} "
        f"target_max={valid_target_max.mean().item():.6f} predict_max={valid_predict_max.mean().item():.6f} "
        f"top1_match={valid_top1_match.mean().item():.6f} "
        f"predict_at_target_top1={valid_target_top1_predict.mean().item():.6f}"
    )


def _standard_kl_loss(
    target: Tensor,
    predict: Tensor,
    topk_indices: Tensor,
    *,
    row_coefficient: float,
) -> Tensor:
    """
    1. 计算每个 query 的 teacher/student KL。
    2. 忽略完全无有效 top-k 的 padding query。
    3. 检查 (p_i>0,q_i=0) 的无限 KL 情况。
    4. 将所有 query 的 KL 求和并乘 row_coefficient。
    """

    _validate_distribution_shapes(target, predict, topk_indices)
    target_f32 = target.float()
    predict_f32 = predict.float()
    valid_slots = topk_indices != -1
    valid_rows = valid_slots.any(dim=-1)

    invalid_predict = valid_slots & (target_f32 > 0) & (predict_f32 <= 0)
    torch._assert(
        ~invalid_predict.any(),
        "DSA indexer predict has zero probability where target is positive; standard KL is not finite.",
    )

    target_self_term = torch.special.xlogy(target_f32, target_f32).sum(dim=-1)
    target_cross_term = torch.special.xlogy(target_f32, predict_f32).sum(dim=-1)
    per_row_kl = (target_self_term - target_cross_term).masked_fill(~valid_rows, 0.0)
    return per_row_kl.sum() * float(row_coefficient)


@torch.no_grad()
def sparse_attention_target(
    attn_q: Tensor,
    attn_k: Tensor,
    attn_lse: Tensor,
    topk_indices: Tensor,
    *,
    softmax_scale: float,
    topk_length: Tensor | None = None,
) -> Tensor:
    """Recompute the head-aggregated sparse attention teacher distribution."""

    from cudnn.deepseek_sparse_attention.score_recompute import sparse_attn_score_recompute_wrapper

    safe_topk, topk_length, valid_slots = _prepare_sparse_topk(topk_indices, topk_length)
    outputs = sparse_attn_score_recompute_wrapper(
        attn_q.contiguous(),
        attn_k.contiguous(),
        attn_lse.float().contiguous(),
        safe_topk,
        softmax_scale=float(softmax_scale),
        topk_length=topk_length,
        topk_indices_global=False,
    )
    return outputs["target"].float().masked_fill(~valid_slots, 0.0)


@torch.no_grad()
def sparse_indexer_predict(
    index_q: Tensor,
    index_k: Tensor,
    index_weights: Tensor,
    topk_indices: Tensor,
    *,
    topk_length: Tensor | None = None,
) -> Tensor:
    """Recompute the FP32 sparse indexer prediction distribution."""

    from cudnn.deepseek_sparse_attention.score_recompute import sparse_indexer_score_recompute_wrapper

    safe_topk, topk_length, valid_slots = _prepare_sparse_topk(topk_indices, topk_length)
    outputs = sparse_indexer_score_recompute_wrapper(
        index_q.contiguous(),
        index_k.contiguous(),
        index_weights.contiguous(),
        safe_topk,
        topk_length=topk_length,
        topk_indices_global=False,
    )
    return outputs["predict"].float().masked_fill(~valid_slots, 0.0)


def _xtuner_indexer_backward(
    index_q: Tensor,
    index_k: Tensor,
    index_weights: Tensor,
    target: Tensor,
    predict: Tensor,
    safe_topk_indices: Tensor,
    row_coefficient: float,
    grad_loss: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    1. pad 32 head 到cudnn需要的64head
    2. 检查topk是否满足分块要求
    3. 转换loss系数，训练目标是1/N \\sum KL_i， cudnn计算的loss自带了一个1/N
    """

    from cudnn.deepseek_sparse_attention.indexer_backward import indexer_backward_wrapper

    index_q_for_cudnn, index_weights_for_cudnn, index_heads = _pad_indexer_heads_for_cudnn(index_q, index_weights)
    topk = safe_topk_indices.shape[-1]
    # The SM90 kernel tiles the sparse dimension in blocks of I=128, hence
    # the admissible top-k sizes satisfy K mod I = 0.
    if topk % _CUDNN_INDEXER_BACKWARD_BLOCK_I != 0:
        raise ValueError(
            "cuDNN sparse indexer backward requires topk to be a multiple of "
            f"{_CUDNN_INDEXER_BACKWARD_BLOCK_I}, got {topk}"
        )

    # The current cuDNN SM90 indexer-backward kernel requires at least 64
    # index heads, while GLM-5.2 uses 32.  For each query/key pair,
    #   s_bt = sum_{h=1}^{H} w_bh * ReLU(q_bh^T k_t).
    # Extending q_h=w_h=0 for h>H gives s'_bt=s_bt.  The padded terms also
    # contribute zero to dK; dQ/dW are projected back to their first H heads.
    # Keep the already-applied 32-head scaling unchanged.
    #
    # Let N=B*S be the number of physical rows.  The backend computes
    #   L_backend = c_backend * (1/N) * sum_{i=1}^{N} KL_i,
    # while XTuner exposes
    #   L_xtuner = c_row * sum_{i=1}^{N} KL_i.
    # Therefore c_backend = N * c_row makes the two losses and gradients equal.
    physical_rows = index_q.shape[0] * index_q.shape[1]
    backend_loss_coeff = float(row_coefficient) * physical_rows
    # ``grad_loss`` may be an aligned-looking contiguous view into autograd's
    # shared scalar buffer whose storage offset makes its actual data pointer
    # fail CuTe DSL's 16-byte alignment requirement.  cuDNN converts it with
    # ``copy=False``, so force a fresh allocation here.  ``contiguous()`` alone
    # is insufficient because it can return the original contiguous view.
    # By the chain rule, dL_outer/dtheta = g * dL_kl/dtheta, where
    # g=dL_outer/dL_kl.  Relocating the scalar keeps g_aligned=g, so gradients
    # are numerically unchanged.
    aligned_grad_loss = _copy_aligned_grad_loss(grad_loss, index_q.device)
    # The cuDNN wrapper overwrites target/predict while forming score gradients.
    # Work on fresh aligned buffers so the custom op remains functionally pure,
    # caller-visible distributions stay intact, and retain_graph backward gets a
    # new unmodified pair on every invocation.
    target_for_cudnn = _aligned_contiguous(target.clone(memory_format=torch.contiguous_format))
    predict_for_cudnn = _aligned_contiguous(predict.clone(memory_format=torch.contiguous_format))
    outputs = indexer_backward_wrapper(
        _aligned_contiguous(index_q_for_cudnn),
        _aligned_contiguous(index_weights_for_cudnn),
        _aligned_contiguous(index_k),
        target_for_cudnn,
        predict_for_cudnn,
        _aligned_contiguous(safe_topk_indices.to(dtype=torch.int32)),
        sm_scale=1.0,
        loss_coeff=backend_loss_coeff,
        grad_loss=aligned_grad_loss,
        topk_indices_global=False,
    )
    # This is the projection P_H onto the model-owned coordinates:
    # dQ = P_H(dQ') = dQ'[..., :H, :] and dW = P_H(dW') = dW'[..., :H].
    return (
        outputs["d_index_q"][..., :index_heads, :].contiguous(),
        outputs["d_index_k"],
        outputs["d_weights"][..., :index_heads].contiguous(),
    )


@torch.library.custom_op(
    "sparse_mla::cudnn_dsa_indexer_kl_backward",
    mutates_args=(),
    device_types="cuda",
)
def _cudnn_dsa_indexer_kl_backward(
    index_q: Tensor,
    index_k: Tensor,
    index_weights: Tensor,
    target: Tensor,
    predict: Tensor,
    safe_topk_indices: Tensor,
    row_coefficient: float,
    grad_loss: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    return _xtuner_indexer_backward(
        index_q,
        index_k,
        index_weights,
        target,
        predict,
        safe_topk_indices,
        row_coefficient,
        grad_loss,
    )


@_cudnn_dsa_indexer_kl_backward.register_fake
def _(
    index_q: Tensor,
    index_k: Tensor,
    index_weights: Tensor,
    target: Tensor,
    predict: Tensor,
    safe_topk_indices: Tensor,
    row_coefficient: float,
    grad_loss: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    del target, predict, safe_topk_indices, row_coefficient, grad_loss
    return torch.empty_like(index_q), torch.empty_like(index_k), torch.empty_like(index_weights)


@torch.library.custom_op(
    "sparse_mla::cudnn_dsa_indexer_kl_from_distribution",
    mutates_args=(),
    device_types="cuda",
)
def _cudnn_dsa_indexer_kl_from_distribution(
    index_q: Tensor,
    index_k: Tensor,
    index_weights: Tensor,
    target: Tensor,
    predict: Tensor,
    topk_indices: Tensor,
    row_coefficient: float,
) -> Tensor:
    del index_q, index_k, index_weights
    return _standard_kl_loss(
        target,
        predict,
        topk_indices,
        row_coefficient=row_coefficient,
    )


@_cudnn_dsa_indexer_kl_from_distribution.register_fake
def _(
    index_q: Tensor,
    index_k: Tensor,
    index_weights: Tensor,
    target: Tensor,
    predict: Tensor,
    topk_indices: Tensor,
    row_coefficient: float,
) -> Tensor:
    del index_q, index_k, index_weights, predict, topk_indices, row_coefficient
    return target.new_empty((), dtype=torch.float32)


def _setup_indexer_kl_context(ctx, inputs, output) -> None:
    del output
    index_q, index_k, index_weights, target, predict, topk_indices, row_coefficient = inputs
    safe_topk_indices, _, _ = _prepare_sparse_topk(topk_indices, topk_length=None)
    ctx.row_coefficient = row_coefficient
    ctx.save_for_backward(
        index_q,
        index_k,
        index_weights,
        target,
        predict,
        safe_topk_indices,
    )


def _indexer_kl_backward(ctx, grad_output: Tensor):
    index_q, index_k, index_weights, target, predict, safe_topk_indices = ctx.saved_tensors
    d_index_q, d_index_k, d_weights = _cudnn_dsa_indexer_kl_backward(
        index_q,
        index_k,
        index_weights,
        target,
        predict,
        safe_topk_indices,
        ctx.row_coefficient,
        grad_output,
    )
    return d_index_q, d_index_k, d_weights, None, None, None, None


_cudnn_dsa_indexer_kl_from_distribution.register_autograd(
    _indexer_kl_backward,
    setup_context=_setup_indexer_kl_context,
)


def dsa_indexer_kl_from_distribution(
    index_q: Tensor,
    index_k: Tensor,
    index_weights: Tensor,
    target: Tensor,
    predict: Tensor,
    topk_indices: Tensor,
    *,
    row_coefficient: float,
    valid_query_mask: Tensor | None = None,
    debug_name: str | None = None,
    debug_interval: int = 0,
) -> Tensor:
    """Compute sparse indexer KL with gradients only for indexer features."""

    if float(row_coefficient) == 0.0:
        return target.new_zeros((), dtype=torch.float32)
    _validate_distribution_shapes(target, predict, topk_indices)
    target, predict, topk_indices = _mask_invalid_query_rows(
        target,
        predict,
        topk_indices,
        valid_query_mask,
    )
    _maybe_log_indexer_loss_diagnostics(
        target,
        predict,
        topk_indices,
        debug_name=debug_name,
        debug_interval=debug_interval,
    )
    return _cudnn_dsa_indexer_kl_from_distribution(
        index_q,
        index_k,
        index_weights,
        target.detach(),
        predict.detach(),
        topk_indices,
        float(row_coefficient),
    )


def dsa_indexer_kl_loss(
    index_q: Tensor,
    index_k: Tensor,
    index_weights: Tensor,
    attn_q: Tensor,
    attn_k: Tensor,
    attn_lse: Tensor,
    topk_indices: Tensor,
    *,
    softmax_scale: float,
    row_coefficient: float,
    topk_length: Tensor | None = None,
    valid_query_mask: Tensor | None = None,
    debug_name: str | None = None,
    debug_interval: int = 0,
) -> Tensor:
    """Convenience wrapper for one attention teacher and one indexer."""

    if float(row_coefficient) == 0.0:
        return index_q.new_zeros((), dtype=torch.float32)
    target = sparse_attention_target(
        attn_q.detach(),
        attn_k.detach(),
        attn_lse.detach(),
        topk_indices,
        softmax_scale=softmax_scale,
        topk_length=topk_length,
    )
    predict = sparse_indexer_predict(
        index_q.detach(),
        index_k.detach(),
        index_weights.detach(),
        topk_indices,
        topk_length=topk_length,
    )
    return dsa_indexer_kl_from_distribution(
        index_q,
        index_k,
        index_weights,
        target,
        predict,
        topk_indices,
        row_coefficient=row_coefficient,
        valid_query_mask=valid_query_mask,
        debug_name=debug_name,
        debug_interval=debug_interval,
    )


def ensure_cudnn_dsa_indexer_training_available() -> None:
    try:
        from cudnn.deepseek_sparse_attention.indexer_backward import indexer_backward_wrapper
        from cudnn.deepseek_sparse_attention.score_recompute import (
            sparse_attn_score_recompute_wrapper,
            sparse_indexer_score_recompute_wrapper,
        )

        _ = (
            indexer_backward_wrapper,
            sparse_attn_score_recompute_wrapper,
            sparse_indexer_score_recompute_wrapper,
        )
    except Exception as exc:
        raise RuntimeError(
            "cuDNN DSA indexer training requires score-recompute and indexer-backward support."
        ) from exc


__all__ = [
    "dsa_indexer_kl_from_distribution",
    "dsa_indexer_kl_loss",
    "ensure_cudnn_dsa_indexer_training_available",
    "sparse_attention_target",
    "sparse_indexer_predict",
]
