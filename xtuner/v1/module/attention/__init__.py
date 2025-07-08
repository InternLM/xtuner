# Copyright (c) OpenMMLab. All rights reserved.
from typing import Mapping, Protocol, Type

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
        past_key_values: list[list[torch.Tensor]] | None = None,
        state: ForwardState = ForwardState.TRAINING,
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


def build_attnention(
    config: TransformerConfig,
    layer_idx: int = 0,
) -> AttentionProtocol:
    """Build attention module based on the configuration."""
    if isinstance(config.attention, MHAConfig):
        return MultiHeadAttention(config, layer_idx)
    elif isinstance(config.attention, MLAConfig):
        return MultiLatentAttention(config, layer_idx)
    else:
        raise ValueError(f"Unsupported attention type: {type(config.attention)}")


__all__ = [
    "MultiLatentAttention",
    "MultiHeadAttention",
    "MHAConfig",
    "MLAConfig",
    "AttentionProtocol",
]
