from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, ConfigDict

from xtuner.v1.config.base_model import MoEConfig, TransformerConfig
from xtuner.v1.config.float8 import Float8Config


if TYPE_CHECKING:
    from xtuner.v1.model.interns1.modeling_interns1 import InternS1ForConditionalGeneration


class InternS1VisionConfig(BaseModel):
    model_config = ConfigDict(
        title="Base model config for xtuner",
        extra="allow",
    )
    num_channels: int = 3
    patch_size: tuple[int, int] = (14, 14)
    image_size: tuple[int, int] = (448, 448)
    hidden_size: int = 1024
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    use_qk_norm: bool = False
    num_hidden_layers: int = 24
    hidden_act: str = "gelu"
    norm_type: str = "layer_norm"
    layer_norm_eps: float = 1e-6
    dropout: float = 0.0
    drop_path_rate: float = 0.1
    attention_bias: bool = True
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    initializer_factor: float = 0.1
    layer_scale_init_value: float = 0.1
    hidden_dropout_prob: float = 0.0
    projection_dropout: float = 0.0
    use_absolute_position_embeddings: bool = True
    use_mask_token: bool = False
    use_mean_pooling: bool = True
    float8_cfg: Optional["Float8Config"] = None

    def build(self):
        from xtuner.v1.model.interns1.modeling_vision import InternS1VisionModel

        return InternS1VisionModel(self)


class InternS1ProjectorConfig(BaseModel):
    vision_hidden_size: int = 1024
    text_hidden_size: int = 4096
    downsample_ratio: float = 0.5
    float8_cfg: Optional["Float8Config"] = None

    def build(self):
        from xtuner.v1.model.interns1.modeling_projector import InternS1MultiModalProjector

        return InternS1MultiModalProjector(self)


class InternS1Config(BaseModel):
    model_config = ConfigDict(
        title="Base model config for xtuner",
        extra="allow",
    )
    vision_config: InternS1VisionConfig
    projector_config: InternS1ProjectorConfig
    text_config: MoEConfig | TransformerConfig
    vision_feature_layer: int = -1
    downsample_ratio: float = 0.5
    dynamic_image_size: bool = True
    use_thumbnail: bool = True
    min_dynamic_patch: int = 1
    max_dynamic_patch: int = 12
    projector_hidden_act: str = "gelu"
    image_token_id: int = 152957

    def build(self) -> "InternS1ForConditionalGeneration":
        from xtuner.v1.model.interns1.modeling_interns1 import InternS1ForConditionalGeneration

        return InternS1ForConditionalGeneration(self)
