# Copyright (c) OpenMMLab. All rights reserved.
"""DeepSeek Sparse Attention (DSA) — the compressor / indexer / sparse-attn
stack.

Cohesive home for everything DSA: the :class:`DeepSeekSparseAttention` module and
its config, the lightning :class:`Indexer`, the :class:`KVCompressor`, the native
``sparse_attn`` reference, and the private rope / top-k / FlashMLA kernels they
share. Public symbols are re-exported here so call sites keep importing from
``xtuner.v1.module.attention.dsa``.
"""

from .dsa import DeepSeekSparseAttention, DSAConfig
from .indexer import Indexer, IndexerConfig
from .sparse_attn import sparse_attn


__all__ = [
    "DeepSeekSparseAttention",
    "DSAConfig",
    "Indexer",
    "IndexerConfig",
    "sparse_attn",
]
