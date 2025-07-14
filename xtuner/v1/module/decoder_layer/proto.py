from typing import Protocol

import torch

from xtuner.v1.data_proto import SequenceContext


class DecoderLayerProto(Protocol):
    """Protocol for decoder layer modules."""

    def __init__(self, layer_idx: int = 0): ...

    def decoding(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
    ) -> torch.Tensor: ...

    def prefilling(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
        past_key_values: list[list[torch.Tensor]] | None = None,
    ) -> torch.Tensor: ...

    def __call__(
        self,
        hidden_states: list[float],
        position_embeddings: tuple[list[float], list[float]],
        seq_ctx: dict[str, float],
    ) -> list[float]: ...

    def build_kv_cache(
        self, max_batch_size: int | None = None, max_length: int | None = None, block_size: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
