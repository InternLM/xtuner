from typing import Protocol

import torch
from torch.distributed.device_mesh import DeviceMesh

from xtuner.v1.config.base_model import (
    MoEConfig,
    MoEModelOutputs,
    TransformerConfig,
)
from xtuner.v1.data_proto import SequenceContext


class ModelProtocol(Protocol):
    def __init__(self, config: TransformerConfig, device_mesh: DeviceMesh | None = None):
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


def build_model(config: TransformerConfig, device_mesh: DeviceMesh | None = None):
    if isinstance(config, MoEConfig):
        from .moe.moe import MoE
        from .moe.qwen3 import Qwen3MoE

        if config.model_type is None:
            return MoE(config, ep_mesh=device_mesh)
        elif config.model_type == "qwen":
            return Qwen3MoE(config, ep_mesh=device_mesh)
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")
    else:
        raise ValueError(f"Unsupported model configuration: {type(config)}")
