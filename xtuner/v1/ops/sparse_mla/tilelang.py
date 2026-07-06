# Copyright (c) OpenMMLab. All rights reserved.

import functools
import subprocess
import sys

import torch

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
    raw_output, softmax_lse = _TileLangSparseMLAFunction.apply(q, kv, indices, scaling)
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


class _TileLangSparseMLAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, scaling: float | None):
        from .tilelang_sparse_mla_fwd import sparse_mla_fwd_interface

        q = q.contiguous()
        kv = kv.contiguous()
        indices = indices.to(torch.int32).contiguous()
        out, lse = sparse_mla_fwd_interface(q, kv, indices, sm_scale=scaling)
        ctx.scaling = scaling
        ctx.save_for_backward(q, kv, indices, out, lse)
        # TileLang stores LSE in log2 space for its exp2-based backward. The public
        # sparse_mla contract follows PyTorch's natural-log logsumexp.
        return out, lse * 0.6931471805599453

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, grad_lse: torch.Tensor):
        from .tilelang_sparse_mla_bwd import sparse_mla_bwd

        q, kv, indices, out, lse = ctx.saved_tensors
        dq, dkv = sparse_mla_bwd(q, kv, out, grad_output.contiguous(), indices, lse, sm_scale=ctx.scaling)
        return dq, dkv, None, None


def tilelang_dsa_topk_indices(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    seq_ctx: SequenceContext,
    *,
    index_head_dim: int,
    index_topk: int,
) -> torch.Tensor:
    from .tilelang_indexer_fwd import indexer_fwd_interface

    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise RuntimeError("TileLang DSA indexer requires bfloat16 q and k tensors.")
    if not q.is_cuda or not k.is_cuda or not weights.is_cuda:
        raise RuntimeError("TileLang DSA indexer requires CUDA tensors.")

    q = q.squeeze(0).contiguous()
    k = k.squeeze(0).contiguous()
    weights = (weights.squeeze(0) * (index_head_dim**-0.5)).contiguous()
    starts, ends = _packed_causal_start_end(seq_ctx, q.shape[0], q.device)
    logits = indexer_fwd_interface(q, k, weights, starts, ends, clean_logits=True)

    topk = min(index_topk, q.shape[0])
    topk_scores, topk_indices = logits.topk(topk, dim=-1)
    topk_indices = topk_indices.masked_fill(topk_scores == -torch.inf, -1)
    return topk_indices.to(torch.int64).unsqueeze(1)


def _packed_causal_start_end(
    seq_ctx: SequenceContext, seq_len: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    cu_seq_lens = seq_ctx.cu_seq_lens_q.to(device)
    token_indices = torch.arange(seq_len, device=device)
    seq_indices = torch.searchsorted(cu_seq_lens, token_indices, right=True) - 1
    starts = cu_seq_lens[seq_indices]
    ends = token_indices + 1
    return starts.to(torch.int32), ends.to(torch.int32)


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
