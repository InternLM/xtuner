"""Triton varlen forward kernel for the V4 Indexer's top-k scoring path.

The native Indexer per-sample loop (``Indexer.forward`` in ``indexer.py``)
materialises a ``[1, S_i, n_heads, T_i]`` fp32 score tensor per sample —
4.6 GiB bf16 at ``pack=8192 / n_heads=64 / total_c=2048`` per layer (and
again during activation-checkpoint recompute). Across V4's 4-layer toy this
is the dominant memory hot-spot.

This kernel keeps the per-head ``q · k`` partial scores in registers and
merges them into a running top-K stream tile-by-tile. The big
``[total_q, n_heads, total_c]`` intermediate is never materialised. Output
is the same ``[1, total_q, index_topk]`` sample-local int64 indices the
native path emits.

Top-K maintenance: each per-query program holds a ``[K_TILE = K + BLOCK_C]``
bit-packed ``uint64`` running buffer where the high 32 bits encode the fp32
score (sign-flipped so unsigned-ascending matches fp-ascending) and the low
32 bits encode the sample-local compressed position. Per c-tile we:
  1. Compute the new ``[BLOCK_C]`` scores in registers.
  2. ``tl.cat`` them onto the buffer (Triton's cat with ``can_reorder=True``
     is the only available concat primitive — we don't care about order here
     since the subsequent sort fixes it).
  3. ``tl.topk(K)`` collapses back to the top-K packed entries.
After the loop, the low 32 bits of each packed entry are the sample-local
compressed index for that query (with ``-inf``-packed slots yielding -1 via
the ``tl.where`` mask in the unpack).

Forward-only. The Indexer's output indices flow into ``sparse_attn``'s
``gather`` which has no gradient through indices, so the kernel intentionally
does not back-propagate; the call site wraps in ``torch.no_grad()``.

Constraints:
  * ``N_HEADS``, ``HEAD_DIM``, ``K``, ``BLOCK_C`` are constexpr. ``K + BLOCK_C``
    must be a power of two (Triton's ``tl.topk`` / ``tl.sort`` require it).
  * Indices in each row are sorted descending by score. Native ``torch.topk``
    breaks ties by ascending index; our packed key uses ``~position`` in the
    low 32 bits so ties also break ascending → matches native semantically
    for the score-equivalence parity test.
"""

from typing import cast

import torch
import triton
import triton.language as tl


# Bit-pack helpers (Python-side, mirrored in the kernel for documentation).
_FP32_SIGN_FLIP = 0x80000000


_INF_NEG_SORTABLE = tl.constexpr(0x007FFFFF)  # -inf fp32 (0xFF800000) under the universal pack


@triton.jit
def _fp32_to_sortable(score):
    """Universal fp32 → uint32 monotonic encoding.

    A flat ``bits ^ 0x80000000`` only works for non-negative inputs. The
    indexer's per-head ReLU + weighted sum can produce negative scores
    (the per-head weight has unconstrained sign), so we need the full
    standard "IEEE-to-sortable" trick:
      * positive (sign bit 0) → flip the sign bit → maps to large unsigned
      * negative (sign bit 1) → invert all bits  → maps to small unsigned
        with more-negative floats sorting smaller
    Both ``-inf`` and ``+0.0`` end up at well-defined sortable values, and
    descending unsigned sort matches descending fp32 order.
    """
    bits = score.to(tl.uint32, bitcast=True)
    sign_mask = ((bits >> 31).to(tl.uint32) * 0xFFFFFFFF) | 0x80000000
    return bits ^ sign_mask


@triton.jit
def _pack_score_idx(score, idx, T_PADDED: tl.constexpr):
    """Pack ``(fp32 score, int32 idx)`` into ``uint64`` for sort-descending top-K.

    High 32 bits: monotonic fp32-sortable encoding via :func:`_fp32_to_sortable`.
    Low 32 bits: ``T_PADDED - 1 - idx`` so that on score ties, sorting
    uint64 descending prefers the *smaller* original index — matching
    ``torch.topk(descending=True)``'s tie-break.
    """
    sortable = _fp32_to_sortable(score)
    inv_idx = (T_PADDED - 1) - idx.to(tl.uint32)
    return (sortable.to(tl.uint64) << 32) | inv_idx.to(tl.uint64)


@triton.jit
def _unpack_idx(packed, T_PADDED: tl.constexpr):
    """Inverse of :func:`_pack_score_idx`: recover the original (signed) index."""
    inv_idx = (packed & 0xFFFFFFFF).to(tl.uint32)
    return ((T_PADDED - 1) - inv_idx).to(tl.int32)


@triton.jit
def _packed_score_is_inf_neg(packed):
    """``True`` for the sentinel ``-inf``-score-packed entries.

    Both the initial top-K buffer (filled with ``-inf``-packed) and any
    out-of-horizon position post-mask carry this exact high-32-bit pattern,
    so a single equality test catches "no real score in this slot".
    """
    score_bits_high = (packed >> 32).to(tl.uint32)
    return score_bits_high == _INF_NEG_SORTABLE


@triton.jit
def _indexer_topk_kernel(
    q_ptr,
    kv_ptr,
    weights_ptr,
    sample_id_ptr,
    cu_q_ptr,
    cu_c_ptr,
    out_idx_ptr,
    total_q,
    softmax_scale,
    ratio,
    q_stride_q,
    q_stride_h,
    q_stride_d,
    kv_stride_c,
    kv_stride_d,
    w_stride_q,
    w_stride_h,
    out_stride_q,
    out_stride_k,
    N_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    K: tl.constexpr,
    BLOCK_C: tl.constexpr,
    T_PADDED: tl.constexpr,  # static upper bound on per-sample compressed length
):
    pid = tl.program_id(0)
    if pid >= total_q:
        return

    sid = tl.load(sample_id_ptr + pid)
    sample_q_start = tl.load(cu_q_ptr + sid)
    sample_c_start = tl.load(cu_c_ptr + sid)
    sample_c_end = tl.load(cu_c_ptr + sid + 1)

    in_sample_pos = pid - sample_q_start
    horizon = (in_sample_pos + 1) // ratio
    sample_c_len = sample_c_end - sample_c_start
    c_upper = tl.minimum(horizon, sample_c_len)

    d_offs = tl.arange(0, HEAD_DIM)
    c_block_offs = tl.arange(0, BLOCK_C)

    # Running top-K buffer. Initialised with the ``-inf``-score-packed
    # sentinel (high 32 bits = ``_INF_NEG_SORTABLE``, low 32 bits arbitrary)
    # so it sorts strictly below any finite score; post-loop these slots are
    # detected by :func:`_packed_score_is_inf_neg` and rewritten to ``-1``.
    # Materialise the constexpr sentinel as uint64 via a 1-element tile cast,
    # since Python ``int.to(...)`` isn't a thing inside the JIT and shifting a
    # constexpr produces an ``int`` rather than a Triton scalar.
    sentinel_lit = _INF_NEG_SORTABLE * (1 << 32)
    top_packed = tl.full((K,), sentinel_lit, dtype=tl.uint64)

    for c_block_start in range(0, c_upper, BLOCK_C):
        c_local = c_block_start + c_block_offs
        c_valid = c_local < c_upper
        c_global = sample_c_start + c_local

        # Load kv tile [BLOCK_C, HEAD_DIM]; out-of-horizon rows zeroed so the
        # subsequent ``q · k`` produces 0 (under -inf mask below it cannot
        # enter top-K anyway).
        kv_addr = c_global[:, None] * kv_stride_c + d_offs[None, :] * kv_stride_d
        kv_tile = tl.load(kv_ptr + kv_addr, mask=c_valid[:, None], other=0.0).to(tl.float32)

        # Per-head ``relu(q · k) * w``, summed across heads. q and per-head
        # weight are reloaded per head iter — the per-query slice is small
        # (q is ``[N_HEADS, HEAD_DIM]`` = 32 KB at V4 dims) so L2 absorbs the
        # reissue cost cheaply, and this avoids ``tl.dot``'s 16-minimum tile
        # constraint that breaks the n_heads=4 test fixture.
        score = tl.zeros((BLOCK_C,), dtype=tl.float32)
        for h in tl.static_range(N_HEADS):
            q_h = tl.load(
                q_ptr + pid * q_stride_q + h * q_stride_h + d_offs * q_stride_d,
            ).to(tl.float32)
            w_h = tl.load(weights_ptr + pid * w_stride_q + h * w_stride_h).to(tl.float32)
            qk = tl.sum(q_h[None, :] * kv_tile, axis=1) * softmax_scale
            qk = tl.maximum(qk, 0.0)
            score += qk * w_h

        score = tl.where(c_valid, score, float("-inf"))

        # Bit-pack new scores with their sample-local indices, then merge
        # into the running top-K via ``cat`` + ``tl.topk``. ``can_reorder=True``
        # is fine: ``tl.topk`` re-sorts the concatenation anyway.
        new_packed = _pack_score_idx(score, c_local, T_PADDED)
        combined = tl.cat(top_packed, new_packed, can_reorder=True)
        top_packed = tl.topk(combined, K)

    # Unpack sample-local indices. ``-inf``-score sentinel slots map back to
    # -1 so downstream sparse_attn treats them as masked.
    is_invalid = _packed_score_is_inf_neg(top_packed)
    raw_idx = _unpack_idx(top_packed, T_PADDED)
    out_idx = tl.where(is_invalid, -1, raw_idx)

    out_k = tl.arange(0, K)
    tl.store(out_idx_ptr + pid * out_stride_q + out_k * out_stride_k, out_idx)


def indexer_topk_triton(
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    compressed_cu_seq_lens: torch.Tensor,
    ratio: int,
    index_topk: int,
    softmax_scale: float,
    block_c: int | None = None,
) -> torch.Tensor:
    """Forward-only varlen top-k Indexer.

    Args:
        q (torch.Tensor): ``[1, total_q, n_heads, head_dim]`` rotated query stream
            (Hadamard-rotated by the caller, matching the native Indexer).
        kv (torch.Tensor): ``[1, total_c, head_dim]`` compressed kv stream.
        weights (torch.Tensor): ``[1, total_q, n_heads]`` per-head gate weights,
            already scaled by ``softmax_scale * n_heads ** -0.5``.
        cu_seq_lens (torch.Tensor): ``[B+1]`` int32 cumulative query lengths.
        compressed_cu_seq_lens (torch.Tensor): ``[B+1]`` int32 cumulative
            compressed-kv lengths.
        ratio (int): ``compress_ratio`` (4 for the indexer-bearing V4 layers).
        index_topk (int): top-k width per query (== ``IndexerConfig.index_topk``).
        softmax_scale (float): pre-relu scale applied to per-head dots.
        block_c (int | None): tile size along the compressed-kv axis. Must
            satisfy ``block_c + index_topk`` is a power of two — the default
            picks ``block_c = index_topk`` (so the merge tile is
            ``2 * index_topk``).

    Returns:
        torch.Tensor: ``[1, total_q, index_topk]`` int64 sample-local indices,
            sorted descending by score (ties broken by ascending position).
            Out-of-horizon / out-of-sample slots are ``-1``-padded.
    """
    if q.dim() != 4 or q.size(0) != 1:
        raise ValueError(f"q must be [1, total_q, n_heads, head_dim]; got {tuple(q.shape)}")
    if kv.dim() != 3 or kv.size(0) != 1:
        raise ValueError(f"kv must be [1, total_c, head_dim]; got {tuple(kv.shape)}")
    if weights.dim() != 3 or weights.size(0) != 1:
        raise ValueError(f"weights must be [1, total_q, n_heads]; got {tuple(weights.shape)}")

    total_q = q.size(1)
    total_c = kv.size(1)
    n_heads = q.size(2)
    head_dim = q.size(3)

    if weights.size(1) != total_q or weights.size(2) != n_heads:
        raise ValueError(
            f"weights shape {tuple(weights.shape)} incompatible with q {tuple(q.shape)}"
        )
    if kv.size(-1) != head_dim:
        raise ValueError(f"kv last dim {kv.size(-1)} != q head_dim {head_dim}")

    if block_c is None:
        # Default: match index_topk so the merge tile is ``2 * K``.
        block_c = index_topk
    if (index_topk + block_c) & (index_topk + block_c - 1) != 0:
        raise ValueError(
            f"index_topk + block_c must be a power of two; got {index_topk + block_c}"
        )

    # ``T_PADDED`` is the static upper bound on a sample's compressed length,
    # used as the modulus for the pack/unpack idx-inversion. Use the next
    # power of two ≥ ``total_c`` so every sample's positions fit; the actual
    # in-sample horizon check still happens in-kernel via ``c_upper``.
    t_padded = 1
    while t_padded < max(total_c, index_topk + block_c):
        t_padded *= 2

    device = q.device
    pos = torch.arange(total_q, device=device, dtype=torch.int32)
    sample_id = (torch.searchsorted(cu_seq_lens, pos, right=True) - 1).to(torch.int32)

    out = torch.full((1, total_q, index_topk), -1, dtype=torch.int32, device=device)

    q_2d = q[0]
    kv_2d = kv[0]
    w_2d = weights[0]
    out_2d = out[0]

    grid = (total_q,)

    _indexer_topk_kernel[grid](
        q_2d,
        kv_2d,
        w_2d,
        sample_id,
        cu_seq_lens.to(torch.int32),
        compressed_cu_seq_lens.to(torch.int32),
        out_2d,
        total_q,
        float(softmax_scale),
        int(ratio),
        q_2d.stride(0),
        q_2d.stride(1),
        q_2d.stride(2),
        kv_2d.stride(0),
        kv_2d.stride(1),
        w_2d.stride(0),
        w_2d.stride(1),
        out_2d.stride(0),
        out_2d.stride(1),
        N_HEADS=n_heads,
        HEAD_DIM=head_dim,
        K=index_topk,
        BLOCK_C=block_c,
        T_PADDED=t_padded,
    )

    return cast(torch.Tensor, out.to(torch.long))
