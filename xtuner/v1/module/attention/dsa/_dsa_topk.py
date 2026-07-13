# Copyright (c) OpenMMLab. All rights reserved.
# ============================================================================
# Varlen sparse-attention top-k index ops for DeepSeek Sparse Attention.
#
# Split out of ``dsa.py``: these helpers build the sliding-window / compressed
# top-k indices for a packed varlen batch in a single GPU op (no ``.cpu()``
# sync, no Python per-sample loop), lay kv out in the interleaved
# ``[W_0, C_0, W_1, C_1, ...]`` layout the sparse-attn backends expect, and
# shift sample-local indices into the global kv_full coordinate space.
# ============================================================================

import torch


def _build_window_topk_idxs_varlen(
    window_size: int,
    cu_q: torch.Tensor,
    total_tokens: int,
) -> torch.Tensor:
    """Varlen replacement for :func:`_build_window_topk_idxs`.

    Returns sample-local sliding-window indices for every query token in the
    packed batch in one tensor op. Each token at sample-local position ``s``
    sees the indices ``s-window_size+1 .. s`` clamped to ``[0, s]``;
    out-of-range entries are masked with ``-1``. Caller is responsible for
    shifting these into the global kv_full coordinate space via
    :func:`_shift_topk_to_global`.

    Args:
        window_size (int): Sliding-window span (statically known).
        cu_q (torch.Tensor): ``[B+1]`` cumulative query lengths.
        total_tokens (int): ``cu_q[-1]`` as a Python int (taken from the q tensor's shape).

    Returns:
        torch.Tensor: ``[1, total_tokens, window_size]`` int64 sample-local indices.
    """
    device = cu_q.device
    pos = torch.arange(total_tokens, device=device, dtype=torch.long)
    sample_id = torch.searchsorted(cu_q, pos, right=True) - 1  # [total_tokens]
    in_sample_pos = (pos - cu_q[sample_id]).unsqueeze(-1)  # [total_tokens, 1]
    k_axis = torch.arange(window_size, device=device, dtype=torch.long).unsqueeze(0)  # [1, window_size]
    base = in_sample_pos - (window_size - 1) + k_axis
    valid = (base >= 0) & (base <= in_sample_pos)
    return torch.where(valid, base, torch.full_like(base, -1)).unsqueeze(0)


def _build_compress_topk_idxs_varlen(
    ratio: int,
    cu_q: torch.Tensor,
    cu_c: torch.Tensor,
    total_tokens: int,
    max_compressed_width: int,
) -> torch.Tensor:
    """Varlen replacement for :func:`_build_compress_topk_idxs`
    (compress_ratio==128 deterministic path).

    For each query token at sample-local position ``s``, the valid compressed
    horizon is ``[0, (s+1)//ratio)`` clamped further by the sample's actual
    compressed kv length (``cu_c[i+1] - cu_c[i]``). Output columns beyond the
    per-token horizon are masked with ``-1``. Result is sample-local; caller
    shifts into kv_full coordinates.

    Args:
        ratio (int): ``compress_ratio`` (positional ratio when not using the indexer).
        cu_q (torch.Tensor): ``[B+1]`` cumulative query lengths.
        cu_c (torch.Tensor): ``[B+1]`` cumulative compressed kv lengths (from the compressor).
        total_tokens (int): ``cu_q[-1]`` as a Python int.
        max_compressed_width (int): Static upper bound on the K dim, derived from
            ``(pack_max_length + ratio - 1) // ratio`` so dynamo can specialize.

    Returns:
        torch.Tensor: ``[1, total_tokens, max_compressed_width]`` int64.
    """
    device = cu_q.device
    pos = torch.arange(total_tokens, device=device, dtype=torch.long)
    sample_id = torch.searchsorted(cu_q, pos, right=True) - 1
    in_sample_pos = pos - cu_q[sample_id]
    horizon = (in_sample_pos + 1) // ratio  # [total_tokens] — how far the token can see
    c_lens_per_sample = cu_c[1:] - cu_c[:-1]
    c_lens_per_token = c_lens_per_sample[sample_id]
    upper = torch.minimum(horizon, c_lens_per_token).unsqueeze(-1)  # [total_tokens, 1]
    k_axis = torch.arange(max_compressed_width, device=device, dtype=torch.long).unsqueeze(0)
    return torch.where(k_axis < upper, k_axis, torch.full_like(k_axis, -1)).unsqueeze(0)


def _interleave_window_compressed_kv(
    kv_window: torch.Tensor,
    kv_compressed: torch.Tensor,
    cu_q: torch.Tensor,
    cu_c: torch.Tensor,
    cu_packed: torch.Tensor,
) -> torch.Tensor:
    """Lay kv out as per-sample ``[W_0, C_0, W_1, C_1, ...]`` in a single GPU
    permutation.

    The three sparse-attn backends (native / flash_mla / cudnn) gather kv by
    global index and have no notion of samples; this layout keeps every
    sample's local topk indices inside its own ``[W_i, C_i]`` contiguous region
    after :func:`_shift_topk_to_global` shifts them.

    Args:
        kv_window (torch.Tensor): ``[1, total_q, D]`` packed window kv.
        kv_compressed (torch.Tensor): ``[1, total_c, D]`` packed compressed kv. May be empty.
        cu_q (torch.Tensor): ``[B+1]`` window cumulative lengths.
        cu_c (torch.Tensor): ``[B+1]`` compressed cumulative lengths.
        cu_packed (torch.Tensor): ``[B+1]`` cumulative ``(q_len + c_len)`` per sample.

    Returns:
        torch.Tensor: ``[1, total_q + total_c, D]`` kv_full in interleaved layout.
    """
    device = kv_window.device
    total_kv = kv_window.size(1) + kv_compressed.size(1)
    out_pos = torch.arange(total_kv, device=device, dtype=torch.long)
    sample_id = torch.searchsorted(cu_packed, out_pos, right=True) - 1
    in_packed_pos = out_pos - cu_packed[sample_id]
    q_lens = cu_q[1:] - cu_q[:-1]
    q_lens_per_pos = q_lens[sample_id]
    is_window = in_packed_pos < q_lens_per_pos
    # Both branches of `torch.where` evaluate, so clamp source indices to a
    # legal range. The clamped value is meaningless on the branch it isn't
    # selected for; the mask drops it.
    win_src = (cu_q[sample_id] + in_packed_pos).clamp(min=0, max=max(kv_window.size(1) - 1, 0))
    if kv_compressed.size(1) == 0:
        return kv_window[0].index_select(0, win_src).unsqueeze(0)
    com_src = (cu_c[sample_id] + (in_packed_pos - q_lens_per_pos)).clamp(min=0, max=kv_compressed.size(1) - 1)
    win_gathered = kv_window[0].index_select(0, win_src)
    com_gathered = kv_compressed[0].index_select(0, com_src)
    return torch.where(is_window.unsqueeze(-1), win_gathered, com_gathered).unsqueeze(0)


def _shift_topk_to_global(
    window_topk_local: torch.Tensor,
    compress_topk_local: torch.Tensor | None,
    cu_q: torch.Tensor,
    cu_packed: torch.Tensor,
) -> torch.Tensor:
    """Shift sample-local topk indices into the kv_full coordinate space.

    Window indices shift by ``cu_packed[sample_id]``; compressed indices shift
    by ``cu_packed[sample_id] + q_len[sample_id]`` (skipping past each sample's
    window region). ``-1`` entries pass through untouched so sparse_attn still
    masks them out.

    Args:
        window_topk_local (torch.Tensor): ``[1, total_q, K_w]`` sample-local indices.
        compress_topk_local (torch.Tensor | None): ``[1, total_q, K_c]``
            sample-local indices, or ``None`` when ``compress_ratio == 0``.
        cu_q (torch.Tensor): ``[B+1]`` cumulative query lengths.
        cu_packed (torch.Tensor): ``[B+1]`` cumulative packed-kv lengths.

    Returns:
        torch.Tensor: ``[1, total_q, K_w (+ K_c)]`` int64 indices into kv_full.
    """
    device = window_topk_local.device
    total_q = window_topk_local.size(1)
    pos = torch.arange(total_q, device=device, dtype=torch.long)
    sample_id = torch.searchsorted(cu_q, pos, right=True) - 1
    cu_packed_per_pos = cu_packed[sample_id].view(1, -1, 1)
    window_shifted = torch.where(
        window_topk_local == -1,
        window_topk_local,
        window_topk_local + cu_packed_per_pos,
    )
    if compress_topk_local is None:
        return window_shifted
    q_lens = cu_q[1:] - cu_q[:-1]
    compress_shift = (cu_packed[sample_id] + q_lens[sample_id]).view(1, -1, 1)
    compress_shifted = torch.where(
        compress_topk_local == -1,
        compress_topk_local,
        compress_topk_local + compress_shift,
    )
    return torch.cat([window_shifted, compress_shifted], dim=-1)
