# Copyright (c) OpenMMLab. All rights reserved.
from typing import Literal, NamedTuple, Protocol

import torch

from xtuner.v1.data_proto import SequenceContext


SparseMLABackend = Literal["torch", "tilelang", "cudnn_dsa", "flash_mla", "flash_mla_cudnn"]
# ``deep_gemm_fp8`` names the runtime dependency and its FP8 MQA score path.
DSAIndexerBackend = Literal["torch", "tilelang", "cudnn_dsa", "flash_mla", "deep_gemm_fp8", "cute_dsl"]
# GLM-5.3-Flash's KPool indexer (design doc F5.a) only has these two: "torch" is the eager
# reference path, "tilelang" is the only production kernel today -- unlike GLM-5.2's per-token
# DSAIndexerBackend, there's no cudnn_dsa/flash_mla/deep_gemm_fp8/cute_dsl KPool kernel.
KPoolIndexerBackend = Literal["torch", "tilelang"]


class SparseMLAOutputs(NamedTuple):
    """SparseMLA op outputs.

    Attributes:
        raw_output: Sparse attention output before GLM-5.2's final value
            projection, shaped ``(seq_len, num_heads, value_dim)``.
        softmax_lse: Natural-log logsumexp of the sparse attention scores,
            shaped ``(seq_len, num_heads)``.
    """

    raw_output: torch.Tensor
    softmax_lse: torch.Tensor


class SparseMLAProtocol(Protocol):
    def __call__(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        indices: torch.Tensor,
        scaling: float | None,
        value_dim: int | None = None,
    ) -> SparseMLAOutputs: ...


class DSATopKIndicesProtocol(Protocol):
    """Computes GLM-5.2 DSA sparse source indices.

    Inputs use logical tensors: ``q`` is shaped ``(bsz, S, Ni, Di)``, ``k`` is
    shaped ``(bsz, T, Di)``, and ``weights`` contain raw gates shaped
    ``(bsz, S, Ni)``. Implementations own the full Indexer score scaling,
    including ``Ni**-0.5`` and ``Di**-0.5``.

    Returns:
        ``torch.int32`` tensor shaped ``(seq_len, kv_group, topk)``. Invalid
        slots are padded with ``-1``. For packed inputs, every valid index stays
        inside its sequence and respects causal order.
    """

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
        seq_ctx: SequenceContext,
        *,
        index_head_dim: int,
        index_topk: int,
        query_chunk_size: int | None = None,
    ) -> torch.Tensor: ...


class KPoolTopKIndicesProtocol(Protocol):
    """Computes GLM-5.3-Flash's KPool DSA sparse source indices (design doc
    F5.a).

    Pools ``index_kpool`` consecutive tokens before scoring/top-k, then expands the selected
    pools back to token ids and appends the current incomplete trailing pool. See
    :mod:`xtuner.v1.ops.sparse_mla.kpool` for the pooling math.

    Returns:
        ``torch.int32`` tensor shaped ``(seq_len, 1, kpool_output_width(...))``. Invalid slots
        are padded with ``-1``.
    """

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        gate_scores: torch.Tensor,
        weights: torch.Tensor,
        kpool_ape: torch.Tensor,
        seq_ctx: SequenceContext,
        *,
        index_head_dim: int,
        index_topk: int,
        index_kpool: int = 4,
        always_select_tail: bool = True,
        alignment: int = 512,
        query_chunk_size: int | None = None,
    ) -> torch.Tensor: ...
