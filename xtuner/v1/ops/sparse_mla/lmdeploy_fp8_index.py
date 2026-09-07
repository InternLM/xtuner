# Copyright (c) OpenMMLab. All rights reserved.
"""LMDeploy-compatible Triton FP8 Indexer score kernel.

The kernel body is kept aligned with LMDeploy's ``cuda/ds_index.py``. XTuner
vendors it so the training environment does not need a runtime lmdeploy import;
only the dense-to-paged adapter below is XTuner-specific.
"""

from functools import lru_cache

import torch
import triton
import triton.language as tl


@lru_cache(maxsize=None)
def get_device_props(device=None):
    """Return the small device-property subset used by LMDeploy's kernel."""
    if device is None:
        device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    # Keep LMDeploy's occupancy heuristic. H100/H200 (SM90) use 64 warps/SM.
    warps_per_sm = {(9, 0): 64}.get((props.major, props.minor), 32)
    return {
        "multi_processor_count": props.multi_processor_count,
        "warps_per_sm": warps_per_sm,
    }


@triton.jit(do_not_specialize=["max_q_seqlen", "num_split"])
def _fp8_index_kernel(
    q_ptr,
    q_s_ptr,
    k_cache_ptr,
    k_s_cache_ptr,
    cu_seqlen_q_ptr,
    k_seqlen_ptr,
    block_offset_ptr,
    raw_k_seqlen_ptr,
    row_k_seqlen_out_ptr,
    out_ptr,
    stride_qm: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_qsm: tl.constexpr,
    stride_qsh: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_ksb: tl.constexpr,
    stride_ksn: tl.constexpr,
    stride_boff0,
    stride_boff1: tl.constexpr,
    stride_om,
    stride_on: tl.constexpr,
    max_q_seqlen,
    causal: tl.constexpr,
    use_raw_causal_seqlen: tl.constexpr,
    return_row_k_seqlen: tl.constexpr,
    trim_causal_tail: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    num_split,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fp8 index kernel."""
    m_id = tl.program_id(0).to(tl.int64)
    split_id = tl.program_id(1).to(tl.int64)

    assert stride_qd == 1
    assert stride_kd == 1

    batch_id = m_id // max_q_seqlen
    q_id = m_id % max_q_seqlen
    q_start = tl.load(cu_seqlen_q_ptr + batch_id)
    q_seqlen = tl.load(cu_seqlen_q_ptr + batch_id + 1) - q_start
    if q_id >= q_seqlen:
        return

    q_pos = q_start + q_id
    k_seqlen = tl.load(k_seqlen_ptr + batch_id)
    if k_seqlen <= 0:
        if return_row_k_seqlen and split_id == 0:
            tl.store(row_k_seqlen_out_ptr + q_pos, tl.full((), 0, tl.int32))
        return

    causal_k_seqlen = k_seqlen
    if causal:
        if use_raw_causal_seqlen:
            raw_k_seqlen = tl.load(raw_k_seqlen_ptr + batch_id).to(tl.int64)
            raw_q_start = raw_k_seqlen - q_seqlen
            causal_k_seqlen = (raw_q_start + q_id + 1) // COMPRESS_RATIO
        else:
            causal_k_seqlen = k_seqlen - q_seqlen + q_id + 1
        causal_k_seqlen = tl.minimum(tl.maximum(causal_k_seqlen, 0), k_seqlen)

    if return_row_k_seqlen and split_id == 0:
        tl.store(row_k_seqlen_out_ptr + q_pos, causal_k_seqlen.to(tl.int32))

    if trim_causal_tail:
        loop_k_seqlen = causal_k_seqlen
    else:
        loop_k_seqlen = k_seqlen
    if loop_k_seqlen <= 0:
        return

    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)

    q_ptrs = q_ptr + q_pos * stride_qm + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd
    q_s_ptrs = q_s_ptr + q_pos * stride_qsm + offs_h * stride_qsh
    q = tl.load(q_ptrs)
    q_s = tl.load(q_s_ptrs)

    k_ptrs = k_cache_ptr + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
    k_s_ptrs = k_s_cache_ptr + offs_n * stride_ksn
    o_ptrs = out_ptr + q_pos * stride_om + offs_n * stride_on + split_id * BLOCK_N * stride_on
    boff_ptr = block_offset_ptr + batch_id * stride_boff0 + split_id * stride_boff1

    num_blocks = tl.cdiv(loop_k_seqlen, BLOCK_N)
    split_count = num_split.to(tl.int64)
    boff_id = split_id
    while boff_id < num_blocks:
        boff = tl.load(boff_ptr).to(tl.int64)

        k = tl.load(k_ptrs + boff * stride_kb)
        k_s = tl.load(k_s_ptrs + boff * stride_ksb)

        logits = tl.zeros((BLOCK_H, BLOCK_N), dtype=tl.float32)
        logits = tl.dot(q, k, acc=logits)
        logits = tl.maximum(logits, 0) * q_s[:, None]
        logits_sum = tl.sum(logits, axis=0) * k_s

        if causal:
            mask_off = boff_id * BLOCK_N + offs_n
            mask = mask_off < causal_k_seqlen
            logits_sum = tl.where(mask, logits_sum, float("-inf"))

        tl.store(o_ptrs, logits_sum, mask=offs_n + boff_id * BLOCK_N < loop_k_seqlen)
        boff_id += split_count
        boff_ptr += split_count * stride_boff1
        o_ptrs += split_count * BLOCK_N * stride_on


def fp8_index(
    q: torch.Tensor,
    q_s: torch.Tensor,
    k_cache: torch.Tensor,
    k_s_cache: torch.Tensor,
    cu_seqlen_q: torch.Tensor,
    k_seqlens: torch.Tensor,
    block_offset: torch.Tensor,
    max_q_seqlen: int = None,
    max_k_seqlen: int = None,
    causal: bool = False,
    raw_k_seqlens: torch.Tensor | None = None,
    compress_ratio: int = 1,
    return_row_k_seqlens: bool = False,
    trim_causal_tail: bool = False,
):
    """Fp8 index.

    q: (cum_seqlen, num_heads, head_dim)
    q_s: (cum_seqlen, num_heads)
    k_cache: (num_blocks, block_size, head_dim)
    k_s_cache: (num_blocks, block_size)
    cu_seqlen_q: (batch_size,)
    cu_seqlen_k: (batch_size,)
    block_offset: (batch_size, num_blocks)
    raw_k_seqlens: optional uncompressed KV lengths for compressed-cache
        causal masking.
    """
    assert q.dim() == 3
    assert k_cache.dim() == 3
    assert q_s.dim() == 2
    assert k_s_cache.dim() == 2
    assert compress_ratio >= 1
    cum_seqlen, num_heads, head_dim = q.shape
    block_size = k_cache.size(1)
    batch_size = k_seqlens.numel()
    use_raw_causal_seqlen = raw_k_seqlens is not None
    if use_raw_causal_seqlen:
        assert raw_k_seqlens.dim() == 1
        assert raw_k_seqlens.numel() == batch_size
    is_decoding = batch_size == cum_seqlen
    if max_k_seqlen is None:
        max_num_blocks = k_cache.size(0)
        max_k_seqlen = max_num_blocks * block_size

    # max q seqlen
    if is_decoding:
        if max_q_seqlen is None:
            max_q_seqlen = 1
        assert max_q_seqlen == 1
    elif max_q_seqlen is None:
        max_q_seqlen = cum_seqlen

    assert q.stride(-1) == 1 and k_cache.stride(-1) == 1

    out = q.new_empty((cum_seqlen, max_k_seqlen), dtype=torch.float32)
    row_k_seqlens = None
    if return_row_k_seqlens:
        row_k_seqlens = torch.empty((cum_seqlen,), dtype=torch.int32, device=q.device)
    raw_k_seqlens_arg = raw_k_seqlens if use_raw_causal_seqlen else k_seqlens
    row_k_seqlens_arg = row_k_seqlens if row_k_seqlens is not None else k_seqlens

    num_warps = 4
    device_idx = q.device.index
    props = get_device_props(device_idx)
    num_sm = props["multi_processor_count"]
    # estimated occupancy 12.5%
    warps_per_sm = props["warps_per_sm"] // 8
    assert warps_per_sm >= num_warps
    cta_per_sm = warps_per_sm // num_warps
    cta_per_device = num_sm * cta_per_sm
    # we better have a tensor to indicate batch id of each q
    M = max_q_seqlen * batch_size
    num_split = max(1, triton.cdiv(cta_per_device, M))
    grid = (M, num_split)

    _fp8_index_kernel[grid](
        q,
        q_s,
        k_cache,
        k_s_cache,
        cu_seqlen_q,
        k_seqlens,
        block_offset,
        raw_k_seqlens_arg,
        row_k_seqlens_arg,
        out,
        *q.stride(),
        *q_s.stride(),
        *k_cache.stride(),
        *k_s_cache.stride(),
        *block_offset.stride(),
        *out.stride(),
        max_q_seqlen=max_q_seqlen,
        causal=causal,
        use_raw_causal_seqlen=use_raw_causal_seqlen,
        return_row_k_seqlen=return_row_k_seqlens,
        trim_causal_tail=trim_causal_tail,
        COMPRESS_RATIO=compress_ratio,
        num_split=num_split,
        BLOCK_H=num_heads,
        BLOCK_N=block_size,
        BLOCK_D=head_dim,
        num_warps=num_warps,
    )
    if return_row_k_seqlens:
        return out, row_k_seqlens
    return out


_PAGE_SIZE = 128


def _sequence_ranges(cu: torch.Tensor) -> list[tuple[int, int]]:
    """Read packed boundaries once for the correctness adapter.

    The production LMDeploy path already owns a paged cache.  XTuner's SFT Indexer receives a dense, gathered K tensor,
    so this bridge materializes temporary pages.  Boundary reads are intentionally kept out of the Triton kernel and
    happen once per invocation.
    """
    return [(int(start), int(end)) for start, end in zip(cu[:-1].tolist(), cu[1:].tolist())]


def _build_paged_k_cache(
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    k_ranges: list[tuple[int, int]],
    *,
    page_size: int = _PAGE_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Pack dense per-request K/scale rows into LMDeploy's split cache
    layout."""
    if not k_ranges:
        raise ValueError("at least one K sequence is required")
    lengths = [max(0, end - start) for start, end in k_ranges]
    max_blocks = max((length + page_size - 1) // page_size for length in lengths)
    max_blocks = max(max_blocks, 1)
    num_blocks = len(k_ranges) * max_blocks
    head_dim = k_fp8.shape[-1]
    k_cache = torch.zeros((num_blocks, page_size, head_dim), device=k_fp8.device, dtype=k_fp8.dtype)
    k_s_cache = torch.ones((num_blocks, page_size), device=k_scale.device, dtype=torch.float32)
    block_offset = torch.arange(num_blocks, device=k_fp8.device, dtype=torch.int32).view(len(k_ranges), max_blocks)
    for seq_id, ((start, end), length) in enumerate(zip(k_ranges, lengths)):
        if length == 0:
            continue
        base = seq_id * max_blocks
        pages = (length + page_size - 1) // page_size
        values = k_fp8[start:end]
        scales = k_scale[start:end]
        for page_id in range(pages):
            page_start = page_id * page_size
            page_end = min(page_start + page_size, length)
            k_cache[base + page_id, : page_end - page_start].copy_(values[page_start:page_end])
            k_s_cache[base + page_id, : page_end - page_start].copy_(scales[page_start:page_end])
    k_seqlens = torch.tensor(lengths, device=k_fp8.device, dtype=torch.int32)
    return k_cache, k_s_cache, k_seqlens, block_offset, lengths


def _lmdeploy_fp8_indexer_topk_impl(
    q_fp8: torch.Tensor,
    q_scale: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    cu_seq_lens_q: torch.Tensor,
    cu_seq_lens_k: torch.Tensor,
    shard_start: int,
    index_head_dim: int,
    index_topk: int,
) -> torch.Tensor:
    """Dense XTuner input -> LMDeploy ``fp8_index`` -> global top-k ids.

    This is a correctness bridge for the current packed SFT path.  It uses
    the exact LMDeploy Triton score kernel, while retaining XTuner's public
    ``[S, 1, K]`` output and packed global K indices. ``weights`` are the raw
    per-head gates; the head-count and head-dimension factors are folded into
    one FP32 scale exactly as in LMDeploy's fused Q preprocessing.
    """
    if q_fp8.ndim != 4 or q_fp8.size(0) != 1:
        raise RuntimeError(f"LMDeploy FP8 Indexer expects q=(1,S,H,D), got {tuple(q_fp8.shape)}")
    if q_scale.shape != q_fp8.shape[:-1] or weights.shape != q_fp8.shape[:-1]:
        raise RuntimeError("q_scale and weights must have shape [1, S, H]")
    if k_fp8.ndim != 3 or k_fp8.size(0) != 1 or k_scale.shape != k_fp8.shape[:2]:
        raise RuntimeError("LMDeploy FP8 Indexer expects k=(1,S_k,D), k_scale=(1,S_k)")
    if index_head_dim != 128 or q_fp8.size(-1) != 128 or k_fp8.size(-1) != 128:
        raise RuntimeError("LMDeploy GLM-5.2 FP8 Indexer requires head_dim=128")
    if q_fp8.dtype != torch.float8_e4m3fn or k_fp8.dtype != torch.float8_e4m3fn:
        raise RuntimeError("LMDeploy FP8 Indexer requires E4M3 Q/K")
    if q_scale.dtype != torch.float32 or k_scale.dtype != torch.float32 or weights.dtype != torch.float32:
        raise RuntimeError("LMDeploy FP8 Indexer scales and weights must be float32")
    if index_topk <= 0:
        raise ValueError(f"index_topk must be positive, got {index_topk}")

    q_fp8 = q_fp8.squeeze(0).contiguous()
    q_scale = q_scale.squeeze(0).contiguous()
    weights = weights.squeeze(0).contiguous()
    k_fp8 = k_fp8.squeeze(0).contiguous()
    k_scale = k_scale.squeeze(0).contiguous()

    q_ranges = _sequence_ranges(cu_seq_lens_q.to(device="cpu"))
    k_ranges = _sequence_ranges(cu_seq_lens_k.to(device="cpu"))
    if len(q_ranges) != len(k_ranges):
        raise RuntimeError("Q/K packed sequence counts must match")
    q_total = q_fp8.size(0)
    q_global_begin = int(shard_start)
    q_global_end = q_global_begin + q_total

    active_q: list[tuple[int, int, int, int]] = []
    # (local q start, local q end, sequence id, K offset of first query)
    for seq_id, ((q_start, q_end), (k_start, k_end)) in enumerate(zip(q_ranges, k_ranges)):
        overlap_start = max(q_start, q_global_begin)
        overlap_end = min(q_end, q_global_end)
        if overlap_start >= overlap_end:
            continue
        q_len = max(q_end - q_start, 0)
        k_len = max(k_end - k_start, 0)
        q_offset = overlap_start - q_start
        # For prefill q_len == k_len.  For decode, Q is the suffix of K.
        k_offset = max(k_len - q_len, 0) + q_offset
        local_start = overlap_start - q_global_begin
        local_end = overlap_end - q_global_begin
        active_q.append((local_start, local_end, seq_id, k_offset))

    result = torch.full((q_total, 1, index_topk), -1, device=q_fp8.device, dtype=torch.int32)
    if not active_q:
        return result

    # Build the exact LMDeploy argument layout.  Q is concatenated in the
    # active packed order; K pages remain request-local to prevent leakage.
    q_parts = [q_fp8[start:end] for start, end, _, _ in active_q]
    qs_parts = [q_scale[start:end] for start, end, _, _ in active_q]
    weight_parts = [weights[start:end] for start, end, _, _ in active_q]
    q_flat = torch.cat(q_parts, dim=0).contiguous()
    q_s = torch.cat(qs_parts, dim=0).contiguous()
    q_weight = torch.cat(weight_parts, dim=0).contiguous()
    active_seq_ids = [seq_id for _, _, seq_id, _ in active_q]
    active_k_ranges = [k_ranges[seq_id] for seq_id in active_seq_ids]
    if not any(end > start for start, end in active_k_ranges):
        return result
    k_cache, k_s_cache, k_seqlens, block_offset, _ = _build_paged_k_cache(k_fp8, k_scale, active_k_ranges)

    q_lens = [end - start for start, end, _, _ in active_q]
    cu_q = torch.tensor(
        [0, *torch.cumsum(torch.tensor(q_lens, device="cpu"), dim=0).tolist()],
        device=q_fp8.device,
        dtype=torch.int32,
    )
    # raw_k_seqlens is request-local in LMDeploy's causal formula.  It is the
    # end position of the current query suffix in that request's K sequence.
    raw_k_seqlens = torch.tensor(
        [k_offset + (end - start) for start, end, _, k_offset in active_q],
        device=q_fp8.device,
        dtype=torch.int32,
    )
    max_q_seqlen = max(q_lens)
    max_k_seqlen = max(int(k_cache_len) for k_cache_len in k_seqlens.tolist())
    score_scale = (q_fp8.size(1) ** -0.5) * (index_head_dim**-0.5)
    scores, row_k_seqlens = fp8_index(
        q_flat,
        q_s * q_weight * score_scale,
        k_cache,
        k_s_cache,
        cu_q,
        k_seqlens,
        block_offset,
        max_q_seqlen=max_q_seqlen,
        max_k_seqlen=max_k_seqlen,
        causal=True,
        raw_k_seqlens=raw_k_seqlens,
        compress_ratio=1,
        return_row_k_seqlens=True,
        trim_causal_tail=True,
    )

    # With ``trim_causal_tail=True`` LMDeploy deliberately leaves the padded
    # suffix uninitialized and passes ``row_k_seqlens`` to its top-k kernel.
    # The XTuner adapter masks that suffix before invoking the same selector;
    # otherwise arbitrary allocator contents can evict valid keys.
    column_ids = torch.arange(scores.size(1), device=scores.device)
    scores = scores.masked_fill(column_ids[None, :] >= row_k_seqlens[:, None], -torch.inf)
    width = min(index_topk, scores.size(1))
    if width == index_topk:
        # LMDeploy uses its TileLang byte-radix selector for GLM-5.2's
        # production K=2048 (and K=512 variants).  Keep the same selector so
        # near-tied boundary values receive the same deterministic set.  The
        # per-row lengths are passed explicitly because causal score tails are
        # padded to -inf before selection.
        try:
            from .lmdeploy_sparse_index_topk import (
                is_sparse_index_topk_supported,
                sparse_index_topk,
            )
        except ImportError:

            def is_sparse_index_topk_supported(_k: int) -> bool:
                return False

        if is_sparse_index_topk_supported(index_topk):
            local_ids = sparse_index_topk(
                scores,
                torch.ones_like(row_k_seqlens),
                row_k_seqlens,
                index_topk,
                fill=-1,
                descending=True,
                sorted=False,
            )
            valid = local_ids >= 0
        else:
            values, local_ids = scores.topk(width, dim=-1, sorted=True)
            valid = (local_ids < row_k_seqlens[:, None]) & torch.isfinite(values)
    else:
        values, local_ids = scores.topk(width, dim=-1, sorted=True)
        valid = (local_ids < row_k_seqlens[:, None]) & torch.isfinite(values)
    # Map each request-local page coordinate back to XTuner's packed K space.
    row_k_base = torch.repeat_interleave(
        torch.tensor([k_ranges[seq_id][0] for seq_id in active_seq_ids], device=q_fp8.device),
        torch.tensor(q_lens, device=q_fp8.device),
    )
    global_ids = local_ids.to(torch.int32) + row_k_base[:, None].to(torch.int32)
    global_ids = global_ids.masked_fill(~valid, -1)
    if width < index_topk:
        padded = torch.full((global_ids.size(0), index_topk - width), -1, device=q_fp8.device, dtype=torch.int32)
        global_ids = torch.cat((global_ids, padded), dim=-1)

    cursor = 0
    for local_start, local_end, _, _ in active_q:
        count = local_end - local_start
        result[local_start:local_end, 0] = global_ids[cursor : cursor + count]
        cursor += count
    return result


@torch.library.custom_op("sparse_mla::lmdeploy_fp8_indexer_topk", mutates_args=(), device_types="cuda")
def lmdeploy_fp8_indexer_topk(
    q_fp8: torch.Tensor,
    q_scale: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    cu_seq_lens_q: torch.Tensor,
    cu_seq_lens_k: torch.Tensor,
    shard_start: int,
    index_head_dim: int,
    index_topk: int,
) -> torch.Tensor:
    return _lmdeploy_fp8_indexer_topk_impl(
        q_fp8,
        q_scale,
        k_fp8,
        k_scale,
        weights,
        cu_seq_lens_q,
        cu_seq_lens_k,
        shard_start,
        index_head_dim,
        index_topk,
    )


@lmdeploy_fp8_indexer_topk.register_fake
def _lmdeploy_fp8_indexer_topk_fake(
    q_fp8: torch.Tensor,
    q_scale: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    cu_seq_lens_q: torch.Tensor,
    cu_seq_lens_k: torch.Tensor,
    shard_start: int,
    index_head_dim: int,
    index_topk: int,
) -> torch.Tensor:
    return torch.empty((q_fp8.size(1), 1, index_topk), device=q_fp8.device, dtype=torch.int32)


__all__ = ["fp8_index", "lmdeploy_fp8_indexer_topk"]
