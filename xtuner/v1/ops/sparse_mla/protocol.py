# Copyright (c) OpenMMLab. All rights reserved.
from typing import Literal, NamedTuple, Protocol

import torch

from xtuner.v1.data_proto import SequenceContext


SparseMLABackend = Literal["torch", "tilelang", "cudnn_dsa"]


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

    Returns:
        ``torch.int64`` tensor shaped ``(seq_len, kv_group, topk)``. Invalid
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
    ) -> torch.Tensor: ...
