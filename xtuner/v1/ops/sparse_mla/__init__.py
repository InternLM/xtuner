# Copyright (c) OpenMMLab. All rights reserved.
import torch

from xtuner.v1.data_proto import SequenceContext

from .protocol import DSATopKIndicesProtocol, SparseMLABackend, SparseMLAOutputs, SparseMLAProtocol
from .pytorch import torch_dsa_topk_indices, torch_sparse_mla


def get_sparse_mla(backend: SparseMLABackend) -> SparseMLAProtocol:
    if backend == "torch":
        return torch_sparse_mla
    if backend == "tilelang":
        from .tilelang import tilelang_sparse_mla

        return tilelang_sparse_mla
    raise ValueError(f"Unsupported SparseMLA backend: {backend}")


def sparse_mla(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    scaling: float | None,
    value_dim: int | None = None,
    backend: SparseMLABackend = "torch",
) -> SparseMLAOutputs:
    return get_sparse_mla(backend)(q, kv, indices, scaling=scaling, value_dim=value_dim)


def get_dsa_topk_indices(backend: SparseMLABackend) -> DSATopKIndicesProtocol:
    if backend == "torch":
        return torch_dsa_topk_indices
    if backend == "tilelang":
        from .tilelang import tilelang_dsa_topk_indices

        return tilelang_dsa_topk_indices
    raise ValueError(f"Unsupported DSA indexer backend: {backend}")


def dsa_topk_indices(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    seq_ctx: SequenceContext,
    *,
    index_head_dim: int,
    index_topk: int,
    backend: SparseMLABackend = "torch",
) -> torch.Tensor:
    return get_dsa_topk_indices(backend)(
        q,
        k,
        weights,
        seq_ctx,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
    )


def ensure_tilelang_runtime_available() -> None:
    from .tilelang import ensure_tilelang_runtime_available as _impl

    return _impl()


def sparse_mla_fwd_interface(*args, **kwargs):
    from .tilelang_sparse_mla_fwd import sparse_mla_fwd_interface as _impl

    return _impl(*args, **kwargs)


def sparse_mla_bwd(*args, **kwargs):
    from .tilelang_sparse_mla_bwd import sparse_mla_bwd as _impl

    return _impl(*args, **kwargs)


def indexer_fwd_interface(*args, **kwargs):
    from .tilelang_indexer_fwd import indexer_fwd_interface as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "DSATopKIndicesProtocol",
    "SparseMLABackend",
    "SparseMLAOutputs",
    "SparseMLAProtocol",
    "dsa_topk_indices",
    "ensure_tilelang_runtime_available",
    "get_dsa_topk_indices",
    "get_sparse_mla",
    "indexer_fwd_interface",
    "sparse_mla",
    "sparse_mla_bwd",
    "sparse_mla_fwd_interface",
    "torch_dsa_topk_indices",
    "torch_sparse_mla",
]
