# Copyright (c) OpenMMLab. All rights reserved.
"""DeepGEMM FP8 Indexer score path.

The adapter uses dense K tensors and requires DeepGEMM's contiguous prefill MQA API. It does not build a paged cache.
"""

from dataclasses import dataclass

import torch

from xtuner.v1.data_proto import SequenceContext


try:
    import deep_gemm
except ImportError:
    deep_gemm = None


# Head counts accepted by DeepGEMM's contiguous FP8 MQA kernel at
# ``index_head_dim=128``: the kernel asserts ``block_qh % num_heads == 0``,
# with block_qh=128 in the pinned DeepGEMM build (H=48/80/96/112 fail in
# ``smxx_fp8_mqa_logits.hpp``).  Re-verify this allowlist when bumping DeepGEMM.
DEEPGEMM_MQA_SUPPORTED_HEADS: tuple[int, ...] = (32, 64, 128)


def _packed_query_metadata(
    seq_ctx: SequenceContext,
    query_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    query_starts, query_ends = seq_ctx.packed_causal_query_ranges(query_len, device)
    cu_seq_lens_k = seq_ctx.cu_seq_lens_k.to(device=device, dtype=torch.int32).contiguous()
    sequence_indices = torch.searchsorted(cu_seq_lens_k, query_starts, right=True) - 1
    sequence_indices = sequence_indices.clamp(min=0, max=cu_seq_lens_k.numel() - 2)
    k_starts = cu_seq_lens_k[sequence_indices]
    k_ends = cu_seq_lens_k[sequence_indices + 1]
    return query_starts, query_ends, k_starts, k_ends


def _validate_fp8_indexer_inputs(
    q_fp8: torch.Tensor,
    q_scale: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    query_starts: torch.Tensor,
    query_ends: torch.Tensor,
    k_starts: torch.Tensor,
    k_ends: torch.Tensor,
    index_head_dim: int,
    index_topk: int,
) -> None:
    if q_fp8.ndim != 4 or q_fp8.size(0) != 1:
        raise RuntimeError(f"DeepGEMM FP8 Indexer expects q=(1,S,H,D), got {tuple(q_fp8.shape)}")
    if q_scale.shape != q_fp8.shape[:-1] or weights.shape != q_fp8.shape[:-1]:
        raise RuntimeError("q_scale and weights must have shape [1, S, H]")
    if k_fp8.ndim != 3 or k_fp8.size(0) != 1 or k_scale.shape != k_fp8.shape[:2]:
        raise RuntimeError("DeepGEMM FP8 Indexer expects k=(1,S_k,D), k_scale=(1,S_k)")
    if index_head_dim != 128 or q_fp8.size(-1) != 128 or k_fp8.size(-1) != 128:
        raise RuntimeError("DeepGEMM GLM-5.2 FP8 Indexer requires head_dim=128")
    if q_fp8.dtype != torch.float8_e4m3fn or k_fp8.dtype != torch.float8_e4m3fn:
        raise RuntimeError("DeepGEMM FP8 Indexer requires E4M3 Q/K")
    if q_scale.dtype != torch.float32 or k_scale.dtype != torch.float32 or weights.dtype != torch.float32:
        raise RuntimeError("DeepGEMM FP8 Indexer scales and weights must be float32")
    query_len = q_fp8.size(1)
    for name, value in (
        ("query_starts", query_starts),
        ("query_ends", query_ends),
        ("k_starts", k_starts),
        ("k_ends", k_ends),
    ):
        if value.ndim != 1 or value.numel() != query_len:
            raise RuntimeError(f"{name} must have shape [S]")
        if value.dtype != torch.int32:
            raise RuntimeError(f"{name} must be int32")
    if index_topk <= 0:
        raise ValueError(f"index_topk must be positive, got {index_topk}")


@dataclass
class _LocalIndexerRequest:
    """Packed query metadata needed by the contiguous DeepGEMM adapter."""

    q_flat: torch.Tensor
    q_scale: torch.Tensor
    q_weight: torch.Tensor
    k_ranges: list[tuple[int, int]]
    q_lens: list[int]
    raw_k_seqlens: list[int]
    row_k_base: torch.Tensor
    slots: list[tuple[int, int]]

    @classmethod
    def build(
        cls,
        q_fp8: torch.Tensor,
        q_scale: torch.Tensor,
        weights: torch.Tensor,
        query_starts: torch.Tensor,
        query_ends: torch.Tensor,
        k_starts: torch.Tensor,
        k_ends: torch.Tensor,
    ) -> "_LocalIndexerRequest":
        q_flat = q_fp8.squeeze(0).contiguous()
        q_s = q_scale.squeeze(0).contiguous()
        q_weight = weights.squeeze(0).contiguous()
        row_k_base = k_starts.contiguous()
        if q_flat.size(0) == 0:
            return cls(q_flat, q_s, q_weight, [], [], [], row_k_base, [])

        segment_change = torch.ones(query_starts.numel(), device=query_starts.device, dtype=torch.bool)
        segment_change[1:] = (query_starts[1:] != query_starts[:-1]) | (k_starts[1:] != k_starts[:-1])
        slot_starts = torch.nonzero(segment_change).flatten().tolist()
        slots: list[tuple[int, int]] = []
        k_ranges: list[tuple[int, int]] = []
        q_lens: list[int] = []
        raw_k_seqlens: list[int] = []
        total = query_starts.numel()
        for slot_idx, slot_start in enumerate(slot_starts):
            slot_end = slot_starts[slot_idx + 1] if slot_idx + 1 < len(slot_starts) else total
            k_start = int(k_starts[slot_start])
            k_end = int(k_ends[slot_start])
            slots.append((slot_start, slot_end))
            k_ranges.append((k_start, k_end))
            q_lens.append(slot_end - slot_start)
            raw_k_seqlens.append(int(query_ends[slot_end - 1]) - k_start)
        return cls(q_flat, q_s, q_weight, k_ranges, q_lens, raw_k_seqlens, row_k_base, slots)

    def to_global_packed_ids(self, local_ids: torch.Tensor, valid: torch.Tensor, index_topk: int) -> torch.Tensor:
        result = torch.full(
            (self.q_flat.size(0), 1, index_topk),
            -1,
            device=self.q_flat.device,
            dtype=torch.int32,
        )
        if not self.slots:
            return result
        global_ids = local_ids.to(torch.int32) + self.row_k_base[:, None].to(torch.int32)
        global_ids = global_ids.masked_fill(~valid, -1)
        if global_ids.size(-1) < index_topk:
            padded = torch.full(
                (global_ids.size(0), index_topk - global_ids.size(-1)),
                -1,
                device=global_ids.device,
                dtype=torch.int32,
            )
            global_ids = torch.cat((global_ids, padded), dim=-1)
        cursor = 0
        for local_start, local_end in self.slots:
            count = local_end - local_start
            result[local_start:local_end, 0] = global_ids[cursor : cursor + count]
            cursor += count
        return result


def _deep_gemm_scores(
    q_flat: torch.Tensor,
    q_s: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    request: _LocalIndexerRequest,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run DeepGEMM's contiguous FP8 MQA Indexer kernel."""
    if deep_gemm is None or not (hasattr(deep_gemm, "fp8_mqa_logits") or hasattr(deep_gemm, "fp8_fp4_mqa_logits")):
        raise RuntimeError(
            "indexer_backend='deep_gemm_fp8' requires DeepGEMM's contiguous "
            "fp8_mqa_logits API; the FP8 Indexer path does not fall back to Triton"
        )
    max_k = max((end - start for start, end in request.k_ranges), default=1)
    scores = torch.full((q_flat.size(0), max_k), -torch.inf, device=q_flat.device, dtype=torch.float32)
    row_lens = torch.zeros((q_flat.size(0),), device=q_flat.device, dtype=torch.int32)
    q_cursor = 0
    for (k_start, k_end), q_len, raw_len in zip(request.k_ranges, request.q_lens, request.raw_k_seqlens):
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


def _select_topk(
    scores: torch.Tensor,
    row_k_seqlens: torch.Tensor,
    index_topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select local top-k IDs and mark rows that are valid under causal
    lengths."""
    width = min(index_topk, scores.size(1))
    if width == index_topk:
        try:
            from .lmdeploy_sparse_index_topk import is_sparse_index_topk_supported, sparse_index_topk
        except ImportError:
            pass
        else:
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
                return local_ids, local_ids >= 0

    column_ids = torch.arange(scores.size(1), device=scores.device)
    masked_scores = scores.masked_fill(column_ids[None, :] >= row_k_seqlens[:, None], -torch.inf)
    values, local_ids = masked_scores.topk(width, dim=-1, sorted=True)
    valid = (local_ids < row_k_seqlens[:, None]) & torch.isfinite(values)
    return local_ids, valid


def _lmdeploy_fp8_indexer_topk_impl(
    q_fp8: torch.Tensor,
    q_scale: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    query_starts: torch.Tensor,
    query_ends: torch.Tensor,
    k_starts: torch.Tensor,
    k_ends: torch.Tensor,
    index_head_dim: int,
    index_topk: int,
) -> torch.Tensor:
    """Run the FP8 Indexer score and selection pipeline for packed prefill."""
    _validate_fp8_indexer_inputs(
        q_fp8,
        q_scale,
        k_fp8,
        k_scale,
        weights,
        query_starts,
        query_ends,
        k_starts,
        k_ends,
        index_head_dim,
        index_topk,
    )
    # Remove only the validated singleton batch dimension. Unlike a generic
    # squeeze, indexing keeps the sequence dimension when K has length one.
    k_flat = k_fp8[0].contiguous()
    k_s = k_scale[0].contiguous()
    request = _LocalIndexerRequest.build(
        q_fp8,
        q_scale,
        weights,
        query_starts,
        query_ends,
        k_starts,
        k_ends,
    )
    if not request.slots:
        return request.to_global_packed_ids(
            torch.empty((0, index_topk), device=q_fp8.device, dtype=torch.int32),
            torch.empty((0, index_topk), device=q_fp8.device, dtype=torch.bool),
            index_topk,
        )
    score_scale = (request.q_flat.size(1) ** -0.5) * (index_head_dim**-0.5)
    scores, row_k_seqlens = _deep_gemm_scores(
        request.q_flat,
        request.q_scale * request.q_weight * score_scale,
        k_flat,
        k_s,
        request,
    )
    local_ids, valid = _select_topk(scores, row_k_seqlens, index_topk)
    return request.to_global_packed_ids(local_ids, valid, index_topk)


def lmdeploy_fp8_dsa_topk_indices(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    seq_ctx: SequenceContext,
    *,
    index_head_dim: int,
    index_topk: int,
) -> torch.Tensor:
    """Run the DeepGEMM FP8 Indexer through the common DSA seam.

    The public DSA protocol keeps logical BF16 Q/K inputs and raw gates. FP8 quantization, packed sequence conversion
    and DeepGEMM invocation stay private to this adapter.
    """
    if not q.is_cuda or not k.is_cuda or not weights.is_cuda:
        raise RuntimeError("DeepGEMM FP8 Indexer requires CUDA q, k, and weights")
    if q.ndim != 4 or q.size(0) != 1 or k.ndim != 3 or k.size(0) != 1:
        raise RuntimeError("DeepGEMM FP8 Indexer expects q=(1,S,H,D) and k=(1,S_k,D)")
    if index_head_dim != 128 or q.size(-1) != 128 or k.size(-1) != 128:
        raise RuntimeError("DeepGEMM GLM-5.2 FP8 Indexer requires head_dim=128")
    if weights.shape != q.shape[:-1]:
        raise RuntimeError("DeepGEMM FP8 Indexer expects weights with shape [1, S, H]")
    # Keep the historical adapter behavior: quantization is defined from a
    # BF16 logical tensor even when an upstream projection runs in another
    # floating-point dtype.
    q = q.to(torch.bfloat16) if q.dtype != torch.bfloat16 else q
    k = k.to(torch.bfloat16) if k.dtype != torch.bfloat16 else k

    from .indexer_fp8_quant import indexer_fp8_quant

    q_fp8, q_scale = indexer_fp8_quant(q.contiguous())
    k_fp8, k_scale = indexer_fp8_quant(k.contiguous())
    query_starts, query_ends, k_starts, k_ends = _packed_query_metadata(seq_ctx, q.size(1), q.device)
    return lmdeploy_fp8_indexer_topk(
        q_fp8,
        q_scale,
        k_fp8,
        k_scale,
        weights.float().contiguous(),
        query_starts,
        query_ends,
        k_starts,
        k_ends,
        index_head_dim,
        index_topk,
    )


@torch.library.custom_op("sparse_mla::lmdeploy_fp8_indexer_topk", mutates_args=(), device_types="cuda")
def lmdeploy_fp8_indexer_topk(
    q_fp8: torch.Tensor,
    q_scale: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    query_starts: torch.Tensor,
    query_ends: torch.Tensor,
    k_starts: torch.Tensor,
    k_ends: torch.Tensor,
    index_head_dim: int,
    index_topk: int,
) -> torch.Tensor:
    return _lmdeploy_fp8_indexer_topk_impl(
        q_fp8,
        q_scale,
        k_fp8,
        k_scale,
        weights,
        query_starts,
        query_ends,
        k_starts,
        k_ends,
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
    query_starts: torch.Tensor,
    query_ends: torch.Tensor,
    k_starts: torch.Tensor,
    k_ends: torch.Tensor,
    index_head_dim: int,
    index_topk: int,
) -> torch.Tensor:
    return torch.empty((q_fp8.size(1), 1, index_topk), device=q_fp8.device, dtype=torch.int32)


__all__ = ["DEEPGEMM_MQA_SUPPORTED_HEADS", "lmdeploy_fp8_dsa_topk_indices", "lmdeploy_fp8_indexer_topk"]
