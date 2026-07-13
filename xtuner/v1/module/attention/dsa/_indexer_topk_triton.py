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

Top-K maintenance: each per-query program holds a ``[K_TILE = next_pow2(K +
BLOCK_C)]`` bit-packed ``uint64`` running buffer where the high 32 bits encode
the fp32 score (sign-flipped so unsigned-ascending matches fp-ascending) and the
low 32 bits encode the sample-local compressed position. Slots ``[0, K)`` hold the
running top-K; ``[K, K + BLOCK_C)`` are the staging slot. Per c-tile we:
  1. Compute the new ``[BLOCK_C]`` scores in registers.
  2. Scatter them into the staging slot (gather + ``tl.where`` over the constexpr
     ``K_TILE`` — Triton has no dynamic-offset register scatter).
  3. ``tl.topk(K)`` collapses the buffer back to its top-K in ``[0, K)`` and the
     staging slots are reset to the sentinel.
Sizing the buffer at ``K_TILE`` rather than ``next_pow2(total_c)`` makes the
collapse cost scale with each query's *horizon* (number of c-tiles), not with the
global compressed length — the win for the many short sub-samples in a packed
varlen batch. After the loop, the low 32 bits of each packed entry are the
sample-local compressed index for that query (``-inf``-packed slots yield -1 via
the ``tl.where`` mask in the unpack).

Forward-only. The Indexer's output indices flow into ``sparse_attn``'s
``gather`` which has no gradient through indices, so the kernel intentionally
does not back-propagate; the call site wraps in ``torch.no_grad()``.

Constraints:
  * ``N_HEADS``, ``HEAD_DIM``, ``K``, ``BLOCK_C`` are constexpr. The running
    buffer width ``K_TILE = next_pow2(K + BLOCK_C)`` is a power of two as
    ``tl.topk`` / ``tl.sort`` require; ``K + BLOCK_C`` itself need not be.
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
    T_PADDED: tl.constexpr,  # idx-inversion modulus: next_pow2(total_c), so every position fits
    K_TILE: tl.constexpr,  # running-buffer width: next_pow2(K + BLOCK_C)
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
    h_offs = tl.arange(0, N_HEADS)
    c_block_offs = tl.arange(0, BLOCK_C)

    # Load q [N_HEADS, HEAD_DIM] and per-head weights [N_HEADS] ONCE outside
    # the c-loop. The previous per-c-tile reissue cost ~64 head loads × c-tile
    # count and dominated runtime; lifting it lets ``tl.dot`` reuse q for the
    # whole c-axis sweep. Keep q in bf16 — tl.dot's tensor-core path is
    # bf16/fp16-only on Hopper, and SMEM at V4 dims (q 64×128 + kv 512×128
    # both fp32 would be 288 KB) exceeds the 232 KB hardware limit.
    q_addr = pid * q_stride_q + h_offs[:, None] * q_stride_h + d_offs[None, :] * q_stride_d
    q_tile = tl.load(q_ptr + q_addr).to(tl.bfloat16)
    w_tile = tl.load(weights_ptr + pid * w_stride_q + h_offs * w_stride_h).to(tl.float32)

    # Running top-K buffer of width ``K_TILE = next_pow2(K + BLOCK_C)``: slots
    # ``[0, K)`` hold the current top-K, ``[K, K + BLOCK_C)`` are the per-tile
    # staging slot. Sizing the buffer at ``K_TILE`` rather than the global
    # ``next_pow2(total_c)`` is the whole point — the final ``tl.topk`` cost scales
    # with the buffer width, so a query whose causal horizon is tiny (an early
    # token, or any token in a short packed sub-sample) collapses ``K_TILE`` (1024
    # at V4 dims) per c-tile instead of paying one ``next_pow2(total_c)``-wide
    # (up to 4096) topk. Trade-off: a long single-sample horizon now runs one
    # collapse per c-tile rather than a single final topk, so it is slower than the
    # old full-buffer path — acceptable because packed varlen batches are
    # dominated by many short samples.
    sentinel_lit = _INF_NEG_SORTABLE * (1 << 32)
    running = tl.full((K_TILE,), sentinel_lit, dtype=tl.uint64)
    t_range = tl.arange(0, K_TILE)

    for c_block_start in range(0, c_upper, BLOCK_C):
        c_local = c_block_start + c_block_offs
        c_valid = c_local < c_upper
        c_global = sample_c_start + c_local

        # Load kv tile [BLOCK_C, HEAD_DIM] in bf16 to match q. Out-of-horizon
        # rows zeroed; subsequent ``q · k`` produces 0, the -inf mask below
        # then keeps them out of the top-K regardless.
        kv_addr = c_global[:, None] * kv_stride_c + d_offs[None, :] * kv_stride_d
        kv_tile = tl.load(kv_ptr + kv_addr, mask=c_valid[:, None], other=0.0).to(tl.bfloat16)

        # ``q · kᵀ`` via ``tl.dot`` — drives tensor cores. q_tile is
        # ``(N_HEADS, HEAD_DIM)`` bf16, kv_tile is ``(BLOCK_C, HEAD_DIM)`` bf16;
        # transposed to ``(HEAD_DIM, BLOCK_C)`` for the matmul. Output is
        # ``(N_HEADS, BLOCK_C)`` fp32 (tensor core accumulator). Requires
        # ``N_HEADS >= 16`` and ``BLOCK_C >= 16`` — enforced in the Python
        # wrapper.
        qk = tl.dot(q_tile, tl.trans(kv_tile)) * softmax_scale  # [N_HEADS, BLOCK_C]
        qk = tl.maximum(qk, 0.0)
        score = tl.sum(qk * w_tile[:, None], axis=0)  # [BLOCK_C]
        score = tl.where(c_valid, score, float("-inf"))

        # Stage the BLOCK_C new entries into ``running[K : K + BLOCK_C]`` (gather +
        # where over the constexpr K_TILE width sidesteps Triton's lack of a
        # dynamic-offset register scatter), then collapse the buffer back to its
        # top-K in ``[0, K)`` and reset the staging slots to the sentinel.
        new_packed = _pack_score_idx(score, c_local, T_PADDED)
        in_stage = (t_range >= K) & (t_range < (K + BLOCK_C))
        stage_local = tl.maximum(tl.minimum(t_range - K, BLOCK_C - 1), 0)
        new_expanded = tl.gather(new_packed, stage_local, axis=0)
        running = tl.where(in_stage, new_expanded, running)
        top = tl.topk(running, K)  # [K], descending
        top_expanded = tl.gather(top, tl.minimum(t_range, K - 1), axis=0)
        running = tl.where(t_range < K, top_expanded, sentinel_lit)

    # ``running[0, K)`` already holds the descending top-K (the loop's last
    # collapse, or the all-sentinel init when ``c_upper == 0``); one more topk
    # extracts it as a dense [K] vector.
    top_packed = tl.topk(running, K)

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
        # Default 256: halves the number of c-tiles vs block_c=128 at V4 dims
        # and still keeps SMEM under the 232 KB Hopper limit (kv_tile
        # [256, 128] bf16 = 64 KB + qk [n_heads=64, 256] fp32 = 64 KB +
        # all_packed [T_PADDED] uint64 ~ 16 KB + intermediates ~ 200 KB total
        # under one pipeline stage).
        block_c = 256
    if block_c & (block_c - 1) != 0:
        raise ValueError(f"block_c must be a power of two; got {block_c}")
    # ``tl.dot`` requires both inner dims ≥ 16 (Triton tile-mma minimum).
    # The score path multiplies q ``(N_HEADS, HEAD_DIM)`` by kv-transpose
    # ``(HEAD_DIM, BLOCK_C)``; head_dim is fixed by the model (≥ 64 in
    # practice) so we only need to guard the two we control here.
    if n_heads < 16:
        raise ValueError(
            f"Triton indexer kernel requires n_heads >= 16 (tensor-core tile floor); "
            f"got n_heads={n_heads}. Use backend='native' for small head counts."
        )
    if block_c < 16:
        raise ValueError(f"block_c must be >= 16 for tl.dot; got {block_c}")

    # ``T_PADDED`` is the static upper bound on a sample's compressed length, used
    # only as the modulus for the pack/unpack idx-inversion — every sample-local
    # position must fit, so it is ``next_pow2(total_c)``. ``K_TILE`` is the running
    # top-K buffer width (``next_pow2(K + block_c)``): it must hold the K outputs
    # plus one staged tile, and is what the per-c-tile collapse cost scales with.
    t_padded = 1
    while t_padded < max(total_c, index_topk + block_c):
        t_padded *= 2
    k_tile = 1
    while k_tile < index_topk + block_c:
        k_tile *= 2

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
        K_TILE=k_tile,
        # ``num_stages=1`` keeps SMEM under the 232 KB Hopper limit by
        # disabling kv-load pipelining (the default pipeline would otherwise
        # double the kv_tile footprint). At V4 dims with BLOCK_C=256 the
        # pipelined version is what tips us over.
        num_stages=1,
    )

    return cast(torch.Tensor, out.to(torch.long))
