# Copyright (c) OpenMMLab. All rights reserved.
"""Packed/SP adapter for the SM90 cooperative BF16 scorer and DeepSelect."""

import torch
from torch import Tensor


@torch.library.custom_op("sparse_mla::cooperative_kpool_topk", mutates_args=(), device_types="cuda")
def cooperative_kpool_topk(
    q: Tensor,
    k: Tensor,
    weights: Tensor,
    cu_seq_lens: Tensor,
    shard_start: int,
    pool_size: int,
    topk: int,
    chunk_size: int,
) -> Tensor:
    from .tilelang_indexer_scoring4_cooperative_paired_query_m64_bf16_full_m64n128_producer_chunked import (
        chunked_topk_interface,
    )

    if isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("Cooperative KPool query chunk size must be positive.")
    # DeepSelect's contiguous int32 output rows must be aligned to 32 bytes.
    # Keep the requested budget (rather than rounding it and changing the selection).
    if topk <= 0 or topk % 8:
        raise ValueError("Cooperative KPool requires a pool top-k budget divisible by 8.")
    result = torch.empty((q.shape[0], topk), dtype=torch.int32, device=q.device)
    # DeepSelect has no begin support. Rebase each document's keys and queries,
    # preserving absolute causal positions even when a query shard starts mid-document.
    boundaries = cu_seq_lens.tolist()
    pool_begin = 0
    shard_end = shard_start + q.shape[0]
    for doc_begin, doc_end in zip(boundaries[:-1], boundaries[1:]):
        pool_count = (doc_end - doc_begin + pool_size - 1) // pool_size
        query_begin, query_end = max(doc_begin, shard_start), min(doc_end, shard_end)
        if query_begin < query_end:
            lo, hi = query_begin - shard_start, query_end - shard_start
            positions = torch.arange(
                query_begin - doc_begin + 1, query_end - doc_begin + 1, dtype=torch.int32, device=q.device
            )
            ends = (positions // pool_size).contiguous()
            starts = torch.zeros_like(ends)
            ids = chunked_topk_interface(
                q[lo:hi],
                k[pool_begin : pool_begin + pool_count],
                weights[lo:hi],
                starts,
                ends,
                topk,
                chunk_size=chunk_size,
                out=result[lo:hi],
            )
            if pool_begin:
                ids.add_(pool_begin)
                ids.masked_fill_(ids == pool_begin - 1, -1)
        pool_begin += pool_count
    # Preserve the original selector's short-key output contract.
    return result[:, : min(topk, k.shape[0])].unsqueeze(1).contiguous()


@cooperative_kpool_topk.register_fake
def _fake(
    q: Tensor,
    k: Tensor,
    weights: Tensor,
    cu_seq_lens: Tensor,
    shard_start: int,
    pool_size: int,
    topk: int,
    chunk_size: int,
) -> Tensor:
    return torch.empty((q.shape[0], 1, min(topk, k.shape[0])), device=q.device, dtype=torch.int32)
