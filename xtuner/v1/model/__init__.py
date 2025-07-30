from torch.distributed.device_mesh import DeviceMesh

from xtuner.v1.config import InternS1Config
from xtuner.v1.config.base_model import (
    MoEConfig,
    TransformerConfig,
)

from .base import BaseModel


def build_model(config: TransformerConfig, device_mesh: DeviceMesh | None = None) -> BaseModel:
    if isinstance(config, MoEConfig):
        from .moe.moe import MoE
        from .moe.qwen3 import Qwen3MoE

        if config.model_type is None:
            return MoE(config)
        elif config.model_type == "qwen":
            return Qwen3MoE(config)
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")
    elif isinstance(config, InternS1Config):
        from .interns1 import InternS1ForConditionalGeneration

        return InternS1ForConditionalGeneration(config)
    else:
        raise ValueError(f"Unsupported model configuration: {type(config)}")
