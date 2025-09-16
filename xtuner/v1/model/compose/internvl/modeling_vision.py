from torch import nn
import numpy as np


from .internvl_config import InternVLVisionConfig
from ..intern_s1.modeling_vision import InternS1VisionAttention, InternS1VisionLayer, InternS1VisionEncoder, InternS1VisionModel,InternS1VisionMLP


class InternVLVisionAttention(InternS1VisionAttention):
    pass


class InternVLVisionMLP(InternS1VisionMLP):
    pass


class InternVLVisionLayer(InternS1VisionLayer):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: InternVLVisionConfig, drop_path_rate: float) -> None:
        super().__init__(config, drop_path_rate)
        self.attention = InternVLVisionAttention(config)
        self.mlp = InternVLVisionMLP(config)


class InternVLVisionEncoder(InternS1VisionEncoder):
    def __init__(self, config: InternVLVisionConfig) -> None:
        super().__init__(config)
        dpr = np.linspace(0.0, float(config.drop_path_rate), int(config.num_hidden_layers))
        self.layer = nn.ModuleList([
            InternVLVisionLayer(config, dpr[idx]) for idx in range(config.num_hidden_layers)])


class InternVLVisionModel(InternS1VisionModel):
    config: InternVLVisionConfig

    def __init__(self, config: InternVLVisionConfig) -> None:
        super().__init__(config)

        self.encoder = InternVLVisionEncoder(config)

        self._hf_prefix = "vision_tower."
        self._init_load_spec()
