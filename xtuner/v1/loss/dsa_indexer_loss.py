# Copyright (c) OpenMMLab. All rights reserved.
"""Dense-teacher distillation loss for training a DSA indexer from scratch."""

from __future__ import annotations

from functools import partial

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from xtuner.v1.data_proto import SequenceContext


def _dense_indexer_kl_block(
    index_q: Tensor,
    index_k: Tensor,
    index_weights: Tensor,
    attn_q: Tensor,
    attn_k: Tensor,
    query_starts: Tensor,
    query_ends: Tensor,
    *,
    softmax_scale: float,
    row_coefficient: float,
) -> Tensor:
    """Compute one causal query block without materializing ``S x S``."""

    batch_size, query_len, _, _ = index_q.shape
    key_len = index_k.shape[1]
    if batch_size != 1:
        raise ValueError(f"Dense DSA indexer warmup expects packed batch size 1, got {batch_size}.")

    key_positions = torch.arange(key_len, device=index_q.device)
    causal_mask = (key_positions[None, :] >= query_starts[:, None]) & (key_positions[None, :] < query_ends[:, None])
    causal_mask = causal_mask.unsqueeze(0)

    index_scores = torch.einsum("bqjd,bkd->bqjk", index_q.float(), index_k.float())
    index_logits = torch.einsum("bqjk,bqj->bqk", torch.relu(index_scores), index_weights.float())
    index_logits = index_logits.masked_fill(~causal_mask, float("-inf"))
    student_log_probs = torch.log_softmax(index_logits, dim=-1).masked_fill(~causal_mask, 0.0)

    teacher_logits = torch.einsum("bqhd,bkhd->bhqk", attn_q.float(), attn_k.float())
    teacher_logits = teacher_logits.mul(float(softmax_scale))
    teacher_logits = teacher_logits.masked_fill(~causal_mask.unsqueeze(1), float("-inf"))
    teacher_probs = torch.softmax(teacher_logits, dim=-1).mean(dim=1)

    row_kl = torch.xlogy(teacher_probs, teacher_probs).sum(dim=-1)
    row_kl = row_kl - (teacher_probs * student_log_probs).sum(dim=-1)
    if row_kl.shape != (batch_size, query_len):
        raise RuntimeError(f"Unexpected dense DSA KL shape: {tuple(row_kl.shape)}")
    return row_kl.sum() * float(row_coefficient)


def dense_dsa_indexer_kl_loss(
    index_q: Tensor,
    index_k: Tensor,
    index_weights: Tensor,
    attn_q: Tensor,
    attn_k: Tensor,
    seq_ctx: SequenceContext,
    *,
    softmax_scale: float,
    loss_coefficient: float = 1.0,
    query_block_size: int = 256,
    use_checkpoint: bool = True,
) -> Tensor:
    """Distill packed causal dense attention into one DSA source indexer."""

    if query_block_size <= 0:
        raise ValueError(f"query_block_size must be positive, got {query_block_size}.")
    if index_q.ndim != 4 or index_k.ndim != 3 or index_weights.ndim != 3:
        raise ValueError("Indexer tensors must have shapes [B,S,H,D], [B,S,D], and [B,S,H].")
    if attn_q.ndim != 4 or attn_k.ndim != 4:
        raise ValueError("Teacher tensors must have shapes [B,S,H,D].")
    if index_q.shape[:2] != index_k.shape[:2] or index_q.shape[:2] != index_weights.shape[:2]:
        raise ValueError("Indexer Q, K, and weights must have matching batch/sequence dimensions.")
    if attn_q.shape != attn_k.shape or attn_q.shape[:2] != index_q.shape[:2]:
        raise ValueError("Dense teacher Q/K must match the indexer batch/sequence dimensions.")
    if index_q.shape[-2] != index_weights.shape[-1] or index_q.shape[-1] != index_k.shape[-1]:
        raise ValueError("Indexer head and feature dimensions are inconsistent.")

    valid_query_rows = index_q.shape[1] - seq_ctx.num_padding
    if valid_query_rows <= 0:
        return (
            index_q.sum(dtype=torch.float32)
            + index_k.sum(dtype=torch.float32)
            + index_weights.sum(dtype=torch.float32)
        ) * 0.0
    if float(loss_coefficient) == 0.0:
        return index_q.new_zeros((), dtype=torch.float32)

    starts, ends = seq_ctx.packed_causal_query_ranges(index_q.shape[1], index_q.device)
    block_fn = partial(
        _dense_indexer_kl_block,
        softmax_scale=float(softmax_scale),
        row_coefficient=float(loss_coefficient) / valid_query_rows,
    )

    loss = index_q.new_zeros((), dtype=torch.float32)
    for block_start in range(0, valid_query_rows, query_block_size):
        block_end = min(block_start + query_block_size, valid_query_rows)
        block_args = (
            index_q[:, block_start:block_end],
            index_k,
            index_weights[:, block_start:block_end],
            attn_q[:, block_start:block_end].detach(),
            attn_k.detach(),
            starts[block_start:block_end],
            ends[block_start:block_end],
        )
        block_loss = (
            checkpoint(block_fn, *block_args, use_reentrant=True)
            if use_checkpoint and torch.is_grad_enabled()
            else block_fn(*block_args)
        )
        loss = loss + block_loss
    return loss


__all__ = ["dense_dsa_indexer_kl_loss"]
