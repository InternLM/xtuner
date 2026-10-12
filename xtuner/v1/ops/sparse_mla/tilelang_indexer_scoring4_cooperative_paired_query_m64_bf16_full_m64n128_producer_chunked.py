# Copyright (c) OpenMMLab. All rights reserved.
"""Opt-in query-chunked BF16 scoring and DeepSelect with reusable GPU scratch."""

import torch
from torch import Tensor

from . import tilelang_indexer_scoring4_cooperative_paired_query_m64_bf16_full_m64n128_producer as full


def _query_chunk_capacity(query_len: int, chunks: int | None, chunk_size: int | None) -> int:
    if chunks is not None and chunk_size is not None:
        raise ValueError("Specify either chunks or chunk_size, not both.")
    if chunk_size is not None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        return chunk_size
    chunks = 4 if chunks is None else chunks
    if chunks <= 0:
        raise ValueError("chunks must be positive.")
    return (query_len + chunks - 1) // chunks


def allocate_score_scratch(
    query_len: int,
    key_len: int,
    device,
    chunks: int | None = None,
    *,
    chunk_size: int | None = None,
) -> Tensor:
    """Allocate one aligned FP32 scratch; default capacity is ceil(Q/4).

    Supply chunk_size for an explicit query capacity, or chunks for a
    requested number of chunks. The two options are mutually exclusive.
    """
    capacity = _query_chunk_capacity(query_len, chunks, chunk_size)
    return full.allocate_scores(capacity, key_len, device)


def chunked_topk_interface(
    q: Tensor,
    k: Tensor,
    weights: Tensor,
    starts: Tensor,
    ends: Tensor,
    topk: int = 512,
    *,
    chunks: int | None = None,
    chunk_size: int | None = None,
    score_scratch: Tensor | None = None,
    out: Tensor | None = None,
    math_registers: int = 112,
) -> Tensor:
    """Return top-k IDs using sequential query chunks and one score scratch.

    The installed DeepSelect requires zero starts. Ends retain their
    original key indices when query views are sliced; do not recompute
    ends from a chunk's local query indices. The result contains absolute
    key IDs, with -1 padding on short/empty rows.

    Default behavior uses four chunks. Supply chunk_size=128 for a fixed
    query capacity instead; chunks and chunk_size are mutually exclusive.
    Pass score_scratch from allocate_score_scratch() and contiguous int32
    out[Q,topk] to exclude allocation. Each chunk writes directly into an
    out view. Scoring/selection and the next scratch reuse are ordered on
    the current CUDA stream; there are no per-chunk synchronizations.
    Zero-start validation performs one GPU-to-host boolean check before
    chunk launches. Only the final IDs are returned; scratch is reusable.
    """
    import deep_select

    if q.ndim != 3 or q.shape[1:] != (32, 128) or k.ndim != 2 or k.shape[1:] != (128,) or not k.shape[0]:
        raise ValueError("Chunked BF16 scorer requires Q[Q,32,128] and nonempty K[K,128].")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or weights.dtype != torch.float32:
        raise ValueError("Chunked BF16 scorer requires BF16 Q/K and FP32 weights.")
    if starts.dtype != torch.int32 or ends.dtype != torch.int32:
        raise ValueError("Chunked DeepSelect requires int32 ranges.")
    if weights.shape != q.shape[:2] or starts.shape != q.shape[:1] or ends.shape != q.shape[:1]:
        raise ValueError("Chunked weights/ranges must match query rows.")
    if not q.is_cuda or any(t.device != q.device for t in (k, weights, starts, ends)):
        raise ValueError("Chunked scorer requires one CUDA device.")
    if bool((starts != 0).any()):
        raise ValueError("Installed DeepSelect requires zero starts.")
    query_len, key_len = q.shape[0], k.shape[0]
    capacity = _query_chunk_capacity(query_len, chunks, chunk_size)
    if score_scratch is None:
        score_scratch = allocate_score_scratch(query_len, key_len, q.device, chunks, chunk_size=chunk_size)
    if (
        score_scratch.device != q.device
        or score_scratch.dtype != torch.float32
        or score_scratch.ndim != 2
        or score_scratch.shape[0] < capacity
        or score_scratch.shape[1] != key_len
        or score_scratch.stride(1) != 1
        or score_scratch.stride(0) < key_len
        or score_scratch.stride(0) * 4 % deep_select.get_stride_requirement()[0]
    ):
        raise ValueError("score_scratch must have sufficient aligned FP32 query/key storage.")
    if out is None:
        output_alignment = deep_select.get_stride_requirement()[1]
        if topk * 4 % output_alignment:
            raise ValueError("topk must admit a contiguous aligned int32 output row.")
        out = torch.empty((query_len, topk), device=q.device, dtype=torch.int32)
    if (
        out.device != q.device
        or out.dtype != torch.int32
        or out.shape != (query_len, topk)
        or not out.is_contiguous()
        or out.stride(0) * 4 % deep_select.get_stride_requirement()[1]
    ):
        raise ValueError("out must be aligned contiguous int32[Q,topk].")
    if not query_len:
        return out
    for begin in range(0, query_len, capacity):
        end = min(begin + capacity, query_len)
        scores = score_scratch[: end - begin]
        full.scorer_full_interface(
            q[begin:end],
            k,
            weights[begin:end],
            starts[begin:end],
            ends[begin:end],
            out=scores,
            math_registers=math_registers,
        )
        deep_select.topk(
            scores,
            topk,
            end=ends[begin:end].contiguous(),
            indices_type=torch.int32,
            sorted=False,
            sorted_index=False,
            return_value=False,
            output_idx=out[begin:end],
            idx_oob_fill_value=-1,
        )
    return out
