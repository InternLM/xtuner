from typing import Protocol, TypeVar

import torch
from torch.distributed.device_mesh import DeviceMesh

from xtuner.v1.config.base_model import (
    MoEModelOutputs,
    TransformerConfig,
)
from xtuner.v1.data_proto import SequenceContext


T = TypeVar("T", bound=TransformerConfig, covariant=True)


class ModelProtocol(Protocol[T]):
    def __init__(self, config: T, ep_mesh: DeviceMesh | None = None):
        """Initialize the model with the given configuration and device
        mesh."""
        ...

    def __call__(
        self,
        seq_ctx: SequenceContext,
        labels: torch.LongTensor,
        return_router_results: bool = False,
        return_hidden_states: bool = False,
    ) -> MoEModelOutputs: ...

    def forward(
        self,
        seq_ctx: SequenceContext,
        labels: torch.LongTensor,
        return_router_results: bool = False,
        return_hidden_states: bool = False,
    ) -> MoEModelOutputs:
        """Forward pass of the model."""
        ...
