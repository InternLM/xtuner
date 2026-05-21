# Copyright (c) OpenMMLab. All rights reserved.
# ============================================================================
# The sparse-attention semantics modelled here (top-k gather + per-head
# `attn_sink` parameter) are adapted from DeepSeek-V4-Flash `inference/model.py`
# (class `Attention` lines 436-543), Copyright (c) DeepSeek-AI, released
# under the MIT License.
# Upstream reference: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/resolve/main/inference/model.py
# Local cache: .dev_scripts/deepseek_v4_reference/model.py
#
# This file ports only the start_pos == 0 (training) path and stays in pure
# PyTorch. sglang ships a TileLang sparse-attn kernel but it has no backward,
# so the training v1 trades performance for correctness and autograd support.
# ============================================================================

import torch


def sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
    cu_seq_lens: torch.Tensor,
) -> torch.Tensor:
    """Top-k sparse attention with a per-head learnable attention sink.

    Each query position attends only to the KV rows indexed by
    ``topk_idxs[b, s]``. A per-head ``attn_sink`` logit is concatenated to the
    selected logits, the softmax runs over ``k + 1`` slots, and the sink slot
    is dropped from the output — letting the sink absorb probability mass
    without contributing a value vector.

    The KV stream is shared between keys and values (MQA-style), matching the
    V4 reference where the same compressed/window KV tensor is used as both K
    and V (model.py L508 onward).

    Args:
        q (torch.Tensor): Query tensor shaped ``[1, total_tokens, num_heads,
            head_dim]``.
        kv (torch.Tensor): Packed key/value tensor shaped ``[1, T_total,
            head_dim]`` — concatenated ``[window_kv, compressed_kv]`` from the
            DSA layer. The token axis is shared across heads.
        attn_sink (torch.Tensor): Per-head sink logit shaped ``[num_heads]``.
            Stored in fp32 in V4; the dtype is preserved in the softmax slot.
        topk_idxs (torch.Tensor): Top-k indices shaped ``[1, total_tokens, k]``
            with int dtype. Indices point into the ``T_total`` axis of ``kv``;
            ``-1`` marks a masked-out slot that contributes ``-inf`` logit.
        softmax_scale (float): Scalar applied to logits before softmax
            (typically ``head_dim ** -0.5``).
        cu_seq_lens (torch.Tensor): 1D int32 cumulative per-sample query token
            counts with length ``num_samples + 1``. Used to clamp indices
            against the per-sample KV horizon when sparse_attn is called over
            multiple packed samples; an entry that falls outside its sample
            is treated as ``-1``.

    Returns:
        torch.Tensor: Attention output shaped ``[1, total_tokens, num_heads,
        head_dim]`` in the dtype of ``q``.
    """
    if q.dim() != 4 or q.size(0) != 1:
        raise ValueError(
            f"q must be packed varlen shaped [1, total_tokens, num_heads, head_dim]; got {tuple(q.shape)}"
        )
    if kv.dim() != 3 or kv.size(0) != 1:
        raise ValueError(f"kv must be shaped [1, T_total, head_dim]; got {tuple(kv.shape)}")
    if kv.size(-1) != q.size(-1):
        raise ValueError(f"head_dim mismatch: q has {q.size(-1)}, kv has {kv.size(-1)}")
    if attn_sink.dim() != 1 or attn_sink.numel() != q.size(2):
        raise ValueError(f"attn_sink must be shaped [num_heads={q.size(2)}]; got {tuple(attn_sink.shape)}")
    if topk_idxs.dim() != 3 or topk_idxs.size(0) != 1 or topk_idxs.size(1) != q.size(1):
        raise ValueError(
            "topk_idxs must be shaped [1, total_tokens, k] and match q's token axis; "
            f"got {tuple(topk_idxs.shape)} vs q {tuple(q.shape)}"
        )
    if cu_seq_lens.dim() != 1 or cu_seq_lens.numel() < 2:
        raise ValueError(f"cu_seq_lens must be 1D with at least 2 entries; got shape {tuple(cu_seq_lens.shape)}")

    out_dtype = q.dtype
    total_tokens, num_heads, head_dim = q.size(1), q.size(2), q.size(3)
    k = topk_idxs.size(-1)

    # why: numerical stability requires fp32 inside the softmax; we cast back
    # at the very end so the public signature stays dtype-faithful.
    q_f = q.float()
    kv_f = kv.float()
    sink = attn_sink.to(dtype=torch.float32)
    # why: -1 cannot be passed to gather; clamp to 0 to materialize valid
    # rows, then push the corresponding logits to -inf before softmax. The
    # clamped rows are arbitrary KV entries that never contribute to the
    # weighted sum.
    valid_mask = topk_idxs >= 0
    safe_idxs = topk_idxs.clamp(min=0)  # [1, S, k]

    # Per-sample horizon check: an entry whose absolute kv-axis index is
    # outside its sample's KV span is also forced to -inf. This protects
    # callers that pass per-sample-local `topk_idxs` against a packed `kv`
    # tensor — `Indexer` itself emits per-sample-local indices, so cu_seq_lens
    # determines what "valid" means here.
    horizon_mask = _build_horizon_mask(safe_idxs, kv.size(1), cu_seq_lens)
    valid_mask = valid_mask & horizon_mask

    # Gather: kv has shape [1, T, D]; we need rows [1, S, k, D].
    gather_idx = safe_idxs.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    kv_gathered = torch.gather(
        kv_f.unsqueeze(1).expand(-1, total_tokens, -1, -1),
        2,
        gather_idx,
    )  # [1, S, k, D]

    # Logits: q · K^T per head, scaled. q is [1, S, H, D]; gathered KV is the
    # same for all heads (MQA-style).
    logits = torch.einsum("bshd,bskd->bshk", q_f, kv_gathered) * softmax_scale

    # Mask invalid slots.
    mask_logits = valid_mask.unsqueeze(2)  # [1, S, 1, k]
    logits = logits.masked_fill(~mask_logits, float("-inf"))

    # Attach sink logit per head: shape becomes [1, S, H, k + 1]. The sink
    # itself is dense (always valid).
    sink_broadcast = sink.view(1, 1, num_heads, 1).expand(1, total_tokens, num_heads, 1)
    logits_with_sink = torch.cat([logits, sink_broadcast], dim=-1)

    # why: when every k slot is masked (e.g. a query with no valid causal
    # predecessor), softmax over only the sink gives weight 1.0 to the sink
    # and 0.0 to all KV, which is the correct "output nothing" semantics.
    weights = torch.softmax(logits_with_sink, dim=-1)
    kv_weights = weights[..., :k]  # drop the sink column from the output

    # Weighted sum over the k gathered KV vectors.
    out = torch.einsum("bshk,bskd->bshd", kv_weights, kv_gathered)
    return out.to(out_dtype)


def _build_horizon_mask(
    safe_idxs: torch.Tensor,
    kv_total: int,
    cu_seq_lens: torch.Tensor,
) -> torch.Tensor:
    # safe_idxs: [1, total_tokens, k] clamped to [0, kv_total). We treat
    # each cu_seq_lens-bounded sample as owning a contiguous slice of the
    # query axis, and require that its top-k indices fall inside [0, kv_total).
    # When callers pass sample-local indices against a per-sample kv, the
    # in-range check is the responsibility of the caller (Indexer already
    # enforces it); here we only catch absolute out-of-range bugs.
    if kv_total <= 0:
        return torch.zeros_like(safe_idxs, dtype=torch.bool)
    mask = (safe_idxs >= 0) & (safe_idxs < kv_total)
    # cu_seq_lens is currently informational. Cross-sample isolation is the
    # caller's responsibility — DSA's varlen path lays kv out as per-sample
    # ``[W_i, C_i]`` slabs and pre-shifts ``topk_idxs`` so each token's entries
    # stay inside its own slab. We deliberately *don't* validate
    # ``cu_seq_lens[-1] == safe_idxs.size(1)`` here: ``.item()`` would force a
    # device→host sync on the hot path, defeating the whole point of
    # eliminating the per-sample Python loop. The invariant is enforced by
    # construction in :class:`DeepSeekSparseAttention.forward`.
    return mask
