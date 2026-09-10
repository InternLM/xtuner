# Copyright (c) OpenMMLab. All rights reserved.

import functools
import subprocess
import sys

import torch
import torch.nn.functional as F
from torch import Tensor

from xtuner.v1.data_proto import SequenceContext

from .protocol import SparseMLAOutputs


def tilelang_sparse_mla(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    scaling: float | None,
    value_dim: int | None = None,
) -> SparseMLAOutputs:
    _validate_tilelang_sparse_mla_inputs(q, kv, indices, value_dim)
    indices = indices.to(torch.int32).contiguous()
    raw_output, softmax_lse, _ = _tilelang_sparse_mla_forward(q, kv, indices, scaling)
    return SparseMLAOutputs(raw_output=raw_output, softmax_lse=softmax_lse)


def _validate_tilelang_sparse_mla_inputs(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    value_dim: int | None,
) -> None:
    if not q.is_cuda or not kv.is_cuda or not indices.is_cuda:
        raise RuntimeError("TileLang SparseMLA requires q, kv, and indices to be CUDA tensors.")
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        raise RuntimeError("TileLang SparseMLA requires bfloat16 q and kv tensors.")
    if q.ndim != 3 or kv.ndim != 3 or indices.ndim != 3:
        raise RuntimeError("TileLang SparseMLA expects q=(S,H,D), kv=(S,K,D), indices=(S,K,topk).")
    if q.shape[-1] != 576 or kv.shape[-1] != 576:
        raise RuntimeError("TileLang SparseMLA supports GLM-5.2 DSA dim_plus_tail_dim=576 only.")
    if value_dim not in (None, 512):
        raise RuntimeError("TileLang SparseMLA supports value_dim=512 only.")
    if indices.shape[-1] % 64 != 0:
        raise RuntimeError("TileLang SparseMLA requires topk to be divisible by 64.")
    if not q.is_contiguous() or not kv.is_contiguous() or not indices.is_contiguous():
        raise RuntimeError("TileLang SparseMLA requires contiguous q, kv, and indices tensors.")


@torch.library.custom_op("sparse_mla::tilelang_sparse_mla_forward", mutates_args=(), device_types="cuda")
def _tilelang_sparse_mla_forward(
    q: Tensor,
    kv: Tensor,
    indices: Tensor,
    scaling: float | None,
) -> tuple[Tensor, Tensor, Tensor]:
    from .tilelang_sparse_mla_fwd import sparse_mla_fwd_interface

    q = q.contiguous()
    kv = kv.contiguous()
    indices = indices.contiguous()
    out, lse_log2 = sparse_mla_fwd_interface(q, kv, indices, sm_scale=scaling)
    # TileLang stores LSE in log2 space for its exp2-based backward. The public
    # sparse_mla contract follows PyTorch's natural-log logsumexp, but backward
    # keeps the raw log2 LSE to match the original autograd.Function path.
    return out, lse_log2 * 0.6931471805599453, lse_log2


@_tilelang_sparse_mla_forward.register_fake
def _(
    q: Tensor,
    kv: Tensor,
    indices: Tensor,
    scaling: float | None,
) -> tuple[Tensor, Tensor, Tensor]:
    out = q.new_empty((*q.shape[:-1], 512))
    softmax_lse = q.new_empty(q.shape[:-1], dtype=torch.float32)
    lse_log2 = q.new_empty(q.shape[:-1], dtype=torch.float32)
    return out, softmax_lse, lse_log2


def _setup_tilelang_sparse_mla_context(ctx, inputs, output) -> None:
    q, kv, indices, scaling = inputs
    raw_output, _, lse_log2 = output
    ctx.scaling = scaling
    ctx.save_for_backward(q, kv, indices, raw_output, lse_log2)


def _tilelang_sparse_mla_backward(ctx, grad_output: Tensor, grad_lse: Tensor, grad_lse_log2: Tensor):
    q, kv, indices, raw_output, lse_log2 = ctx.saved_tensors
    dq, dkv = _tilelang_sparse_mla_backward_op(
        q,
        kv,
        raw_output,
        grad_output.contiguous(),
        indices,
        lse_log2,
        ctx.scaling,
    )
    return dq, dkv, None, None


_tilelang_sparse_mla_forward.register_autograd(
    _tilelang_sparse_mla_backward, setup_context=_setup_tilelang_sparse_mla_context
)


@torch.library.custom_op("sparse_mla::tilelang_sparse_mla_backward", mutates_args=(), device_types="cuda")
def _tilelang_sparse_mla_backward_op(
    q: Tensor,
    kv: Tensor,
    raw_output: Tensor,
    grad_output: Tensor,
    indices: Tensor,
    lse_log2: Tensor,
    scaling: float | None,
) -> tuple[Tensor, Tensor]:
    from .tilelang_sparse_mla_bwd import sparse_mla_bwd

    dq, dkv = sparse_mla_bwd(q, kv, raw_output, grad_output, indices, lse_log2, sm_scale=scaling)
    return dq, dkv.to(kv.dtype)


@_tilelang_sparse_mla_backward_op.register_fake
def _(
    q: Tensor,
    kv: Tensor,
    raw_output: Tensor,
    grad_output: Tensor,
    indices: Tensor,
    lse_log2: Tensor,
    scaling: float | None,
) -> tuple[Tensor, Tensor]:
    return torch.empty_like(q), torch.empty_like(kv)


def tilelang_dsa_topk_indices(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    seq_ctx: SequenceContext,
    *,
    index_head_dim: int,
    index_topk: int,
    query_chunk_size: int | None = None,
) -> torch.Tensor:
    """Select fixed-ID Top-K supports, optionally bounded by query chunks.

    ``query_chunk_size=None`` keeps a single selector launch; a non-aligned
    tail is internally padded for kernel safety.  A positive value limits the
    transient ``[query_chunk, seq_k]`` logits tile and preserves the fixed-ID
    Top-K semantics while retaining only the final int32 IDs across chunks.

    Args:
        q (torch.Tensor): Indexer query features shaped ``(1, S, H, D)``.
        k (torch.Tensor): Indexer key features shaped ``(1, S_k, D)``.
        weights (torch.Tensor): Per-query head weights shaped ``(1, S, H)``.
        seq_ctx (SequenceContext): Packed-sequence and sequence-parallel metadata.
        index_head_dim (int): Indexer head dimension used for score scaling.
        index_topk (int): Maximum number of source IDs to select per query.
        query_chunk_size (int | None): Maximum query rows per selector launch.

    Returns:
        torch.Tensor: Contiguous int32 IDs shaped ``(S, 1, min(index_topk, S_k))``.
    """

    if query_chunk_size is not None and (
        isinstance(query_chunk_size, bool) or not isinstance(query_chunk_size, int) or query_chunk_size <= 0
    ):
        raise ValueError(f"query_chunk_size must be a positive integer, got {query_chunk_size!r}")
    if isinstance(index_topk, bool) or not isinstance(index_topk, int) or index_topk <= 0:
        raise ValueError(f"index_topk must be a positive integer, got {index_topk!r}")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise RuntimeError("TileLang DSA indexer requires bfloat16 q and k tensors.")
    if not q.is_cuda or not k.is_cuda or not weights.is_cuda:
        raise RuntimeError("TileLang DSA indexer requires CUDA tensors.")

    q = q.squeeze(0).contiguous()
    k = k.squeeze(0).contiguous()
    weights = (weights.squeeze(0) * (index_head_dim**-0.5)).contiguous()
    starts, ends = seq_ctx.packed_causal_query_ranges(q.shape[0], q.device)
    if q.shape[1] <= 0 or 128 % q.shape[1] != 0:
        raise ValueError(
            "TileLang DSA indexer requires the number of index heads to divide the 128-thread query tile, "
            f"got {q.shape[1]}"
        )
    block_q = 128 // q.shape[1]
    k_eff = min(index_topk, k.shape[0])

    # Keep the historical one-shot path when chunking is disabled.  The
    # primitive has no tail guard, so pad a non-aligned final block with empty
    # causal ranges and crop those rows after the kernel returns.
    if query_chunk_size is None or query_chunk_size >= q.shape[0]:
        if q.shape[0] % block_q == 0:
            return _tilelang_dsa_topk_indices_from_ranges(q, k, weights, starts, ends, index_topk)
        q, weights, starts, ends, valid_rows = _pad_tilelang_indexer_query_chunk(
            q,
            weights,
            starts,
            ends,
            block_q=block_q,
        )
        return _tilelang_dsa_topk_indices_from_ranges(
            q,
            k,
            weights,
            starts,
            ends,
            index_topk,
        )[:valid_rows].contiguous()

    # Retain only one final int32 ID tensor.  Dense logits and top-k scratch
    # from each query chunk become unreachable before the next chunk starts.
    final_ids = torch.empty((q.shape[0], 1, k_eff), device=q.device, dtype=torch.int32)
    for lo in range(0, q.shape[0], query_chunk_size):
        hi = min(lo + query_chunk_size, q.shape[0])
        q_chunk, weights_chunk, starts_chunk, ends_chunk, valid_rows = _pad_tilelang_indexer_query_chunk(
            q[lo:hi],
            weights[lo:hi],
            starts[lo:hi],
            ends[lo:hi],
            block_q=block_q,
        )
        chunk_ids = _tilelang_dsa_topk_indices_from_ranges(
            q_chunk,
            k,
            weights_chunk,
            starts_chunk,
            ends_chunk,
            index_topk,
        )
        final_ids[lo:hi].copy_(chunk_ids[:valid_rows])
        del q_chunk, weights_chunk, starts_chunk, ends_chunk, chunk_ids
    return final_ids.contiguous()


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _pad_tilelang_indexer_query_chunk(
    q: torch.Tensor,
    weights: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    *,
    block_q: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Pad an unaligned query tail with rows whose causal range is empty.

    The padding range is placed at the last real row's exclusive end.  The
    TileLang primitive reduces the range bounds across all rows in a query
    block, so using a global sentinel such as ``seq_len_kv`` would unnecessarily
    extend the KV scan for every non-aligned chunk.
    """

    valid_rows = q.shape[0]
    padded_rows = _round_up(valid_rows, block_q)
    padding = padded_rows - valid_rows
    if padding == 0:
        return (
            q.contiguous(),
            weights.contiguous(),
            starts.contiguous(),
            ends.contiguous(),
            valid_rows,
        )
    return (
        F.pad(q, (0, 0, 0, 0, 0, padding)).contiguous(),
        F.pad(weights, (0, 0, 0, padding)).contiguous(),
        # start=end marks an empty row.  Reusing the final real end keeps both
        # block-wide range reductions bounded by the real rows in this chunk.
        torch.cat((starts, ends[-1:].expand(padding)), dim=0).contiguous(),
        torch.cat((ends, ends[-1:].expand(padding)), dim=0).contiguous(),
        valid_rows,
    )


@torch.library.custom_op("sparse_mla::tilelang_dsa_topk_indices", mutates_args=(), device_types="cuda")
def _tilelang_dsa_topk_indices_from_ranges(
    q: Tensor,
    k: Tensor,
    weights: Tensor,
    starts: Tensor,
    ends: Tensor,
    index_topk: int,
) -> Tensor:
    from .tilelang_indexer_fwd import indexer_fwd_interface

    logits = indexer_fwd_interface(q, k, weights, starts, ends, clean_logits=True)
    topk = min(index_topk, k.shape[0])
    topk_scores, topk_indices = logits.topk(topk, dim=-1)
    topk_indices = topk_indices.masked_fill(topk_scores == -torch.inf, -1)
    return topk_indices.to(torch.int32).unsqueeze(1)


@_tilelang_dsa_topk_indices_from_ranges.register_fake
def _(
    q: Tensor,
    k: Tensor,
    weights: Tensor,
    starts: Tensor,
    ends: Tensor,
    index_topk: int,
) -> Tensor:
    topk = min(index_topk, k.shape[0])
    return torch.empty((q.shape[0], 1, topk), device=q.device, dtype=torch.int32)


@functools.cache
def _tilelang_runtime_import_error() -> str | None:
    result = subprocess.run(
        [sys.executable, "-c", "import tilelang"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return None
    return result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown import failure"


def ensure_tilelang_runtime_available() -> None:
    detail = _tilelang_runtime_import_error()
    if detail is not None:
        raise RuntimeError(f"TileLang SparseMLA runtime is unavailable: {detail}")
