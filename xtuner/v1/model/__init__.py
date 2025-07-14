from torch.distributed.device_mesh import DeviceMesh

from xtuner.v1.config.base_model import (
    MoEConfig,
    TransformerConfig,
)

from .proto import ModelProtocol


def build_model(config: TransformerConfig, device_mesh: DeviceMesh | None = None) -> ModelProtocol[TransformerConfig]:
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
