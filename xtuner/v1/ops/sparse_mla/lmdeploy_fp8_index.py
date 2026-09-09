# Copyright (c) OpenMMLab. All rights reserved.
"""LMDeploy-compatible DeepGEMM FP8 Indexer score path.

The adapter uses dense K tensors and requires DeepGEMM's contiguous prefill MQA API. It does not build a paged cache.
"""

from functools import lru_cache

import torch


def _sequence_ranges(cu: torch.Tensor) -> list[tuple[int, int]]:
    """Read packed boundaries once for the correctness adapter.

    XTuner's SFT Indexer receives dense, gathered K tensors. Boundary reads happen once before the DeepGEMM calls.
    """
    return [(int(start), int(end)) for start, end in zip(cu[:-1].tolist(), cu[1:].tolist())]


@lru_cache(maxsize=1)
def _get_deep_gemm():
    try:
        import deep_gemm
    except ImportError:
        return None
    has_mqa = hasattr(deep_gemm, "fp8_mqa_logits") or hasattr(deep_gemm, "fp8_fp4_mqa_logits")
    return deep_gemm if has_mqa else None


def _deep_gemm_scores(
    q_flat: torch.Tensor,
    q_s: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    k_ranges: list[tuple[int, int]],
    q_lens: list[int],
    raw_k_seqlens: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run LMDeploy's contiguous DeepGEMM MQA Indexer kernel."""
    deep_gemm = _get_deep_gemm()
    if deep_gemm is None:
        raise RuntimeError(
            "indexer_quant_mode='ue8m0_fp8' requires DeepGEMM's contiguous "
            "fp8_mqa_logits API; the FP8 Indexer path does not fall back to Triton"
        )
    max_k = max((end - start for start, end in k_ranges), default=1)
    scores = torch.full((q_flat.size(0), max_k), -torch.inf, device=q_flat.device, dtype=torch.float32)
    row_lens = torch.zeros((q_flat.size(0),), device=q_flat.device, dtype=torch.int32)
    q_cursor = 0
    for (k_start, k_end), q_len, raw_len in zip(k_ranges, q_lens, raw_k_seqlens):
        k_seq = k_fp8[k_start:k_end].contiguous()
        ks_seq = k_scale[k_start:k_end].contiguous()
        # A contiguous packed slice may still have an unaligned scale pointer.
        # DeepGEMM's TMA input requires 16-byte alignment.
        if ks_seq.data_ptr() % 16:
            ks_seq = ks_seq.clone()
        k_len = k_seq.size(0)
        q_end = q_cursor + q_len
        # Each sequence is passed as an independent contiguous K tensor, so
        # DeepGEMM starts at zero.  ``raw_len`` identifies the query suffix
        # position for decode; prefill has raw_len == q_len.
        q_start = max(raw_len - q_len, 0)
        starts = torch.zeros((q_len,), device=q_flat.device, dtype=torch.int32)
        ends = (q_start + torch.arange(1, q_len + 1, device=q_flat.device, dtype=torch.int32)).clamp(max=k_len)
        if k_len == 0:
            q_cursor = q_end
            continue
        mqa = getattr(deep_gemm, "fp8_mqa_logits", None)
        if mqa is not None:
            seq_scores = mqa(
                q_flat[q_cursor:q_end], (k_seq, ks_seq), q_s[q_cursor:q_end], starts, ends, clean_logits=False
            )
        else:
            seq_scores = deep_gemm.fp8_fp4_mqa_logits(
                q=(q_flat[q_cursor:q_end], None),
                kv=(k_seq, ks_seq),
                weights=q_s[q_cursor:q_end],
                cu_seq_len_k_start=starts,
                cu_seq_len_k_end=ends,
                clean_logits=False,
                max_seqlen_k=max(k_len, 1),
                logits_dtype=torch.float32,
            )
        scores[q_cursor:q_end, : seq_scores.size(1)].copy_(seq_scores)
        row_lens[q_cursor:q_end] = ends
        q_cursor = q_end
    return scores, row_lens


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
    """Dense XTuner input -> LMDeploy DeepGEMM score -> global top-k ids.

    This is a correctness bridge for the current packed SFT path. It uses
    LMDeploy's contiguous DeepGEMM MQA kernel, while retaining XTuner's public
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

    # Build the exact LMDeploy argument layout. Q is concatenated in the
    # active packed order; each DeepGEMM call receives a request-local dense K.
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
    q_lens = [end - start for start, end, _, _ in active_q]
    # raw_k_seqlens is request-local in LMDeploy's causal formula.  It is the
    # end position of the current query suffix in that request's K sequence.
    raw_k_seqlens = torch.tensor(
        [k_offset + (end - start) for start, end, _, k_offset in active_q],
        device=q_fp8.device,
        dtype=torch.int32,
    )
    score_scale = (q_fp8.size(1) ** -0.5) * (index_head_dim**-0.5)
    weighted_q_scale = q_s * q_weight * score_scale
    deepgemm_result = _deep_gemm_scores(
        q_flat,
        weighted_q_scale,
        k_fp8.squeeze(0),
        k_scale.squeeze(0),
        active_k_ranges,
        q_lens,
        raw_k_seqlens.tolist(),
    )
    scores, row_k_seqlens = deepgemm_result

    # Mask the causal suffix before top-k. The per-row lengths are passed to
    # the LMDeploy-compatible selector when it is available.
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

            def is_sparse_index_topk_supported(k: int) -> bool:
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
            column_ids = torch.arange(scores.size(1), device=scores.device)
            scores.masked_fill_(column_ids[None, :] >= row_k_seqlens[:, None], -torch.inf)
            values, local_ids = scores.topk(width, dim=-1, sorted=True)
            valid = (local_ids < row_k_seqlens[:, None]) & torch.isfinite(values)
    else:
        column_ids = torch.arange(scores.size(1), device=scores.device)
        scores.masked_fill_(column_ids[None, :] >= row_k_seqlens[:, None], -torch.inf)
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


__all__ = ["lmdeploy_fp8_indexer_topk"]
