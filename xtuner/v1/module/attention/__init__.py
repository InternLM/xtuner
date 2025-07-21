# Copyright (c) OpenMMLab. All rights reserved.
from typing import Protocol

import torch

from xtuner.v1.config import TransformerConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.utils import ForwardState

from .mha import MHAConfig, MultiHeadAttention
from .mla import MLAConfig, MultiLatentAttention


class AttentionProtocol(Protocol):
    """Protocol for attention modules."""

    def __init__(
        self,
        config: TransformerConfig,
        layer_idx: int = 0,
    ): ...

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
    ) -> torch.Tensor:
        """Forward pass of the attention module."""
        ...

    def __call__(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
        past_key_values: list[list[torch.Tensor]] | None = None,
        state: ForwardState = ForwardState.TRAINING,
    ) -> torch.Tensor:
        """Forward pass of the attention module."""
        ...

    def prefilling(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
        past_key_values: list[list[torch.Tensor]],
    ) -> torch.Tensor: ...

    def decoding(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
        past_key_values: list[list[torch.Tensor]],
    ) -> torch.Tensor: ...

    def build_kv_cache(
        self, max_batch_size: int | None = None, max_length: int | None = None, block_size: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Shape of the key-value cache."""
        ...


__all__ = [
    "MultiLatentAttention",
    "MultiHeadAttention",
    "MHAConfig",
    "MLAConfig",
    "AttentionProtocol",
]
