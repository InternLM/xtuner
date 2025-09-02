from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, ConfigDict

from xtuner.v1.config.base_model import MoEConfig, TransformerConfig
from xtuner.v1.config.float8 import Float8Config
from xtuner.v1.model.dense.qwen3 import Qwen3Dense8BConfig
from xtuner.v1.model.moe.qwen3 import Qwen3MoE235BA22Config


if TYPE_CHECKING:
    from .modeling_intern_s1 import InternS1ForConditionalGeneration


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
    drop_path_rate: float = 0.0
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
        from .modeling_vision import InternS1VisionModel

        return InternS1VisionModel(self)


class InternS1ProjectorConfig(BaseModel):
    vision_hidden_size: int = 1024
    text_hidden_size: int = 4096
    downsample_ratio: float = 0.5
    float8_cfg: Optional["Float8Config"] = None

    def build(self):
        from .modeling_projector import InternS1MultiModalProjector

        return InternS1MultiModalProjector(self)


class InternS1BaseConfig(BaseModel):
    model_config = ConfigDict(
        title="Base model config for xtuner",
        extra="allow",
    )
    vision_config: InternS1VisionConfig
    projector_config: InternS1ProjectorConfig
    text_config: TransformerConfig

    vision_feature_layer: int = -1
    downsample_ratio: float = 0.5
    dynamic_image_size: bool = True
    use_thumbnail: bool = True
    min_dynamic_patch: int = 1
    max_dynamic_patch: int = 12
    projector_hidden_act: str = "gelu"
    image_token_id: int = 152957
    freeze_vision: bool = False
    freeze_projector: bool = False
    freeze_language: bool = False

    def build(self) -> "InternS1ForConditionalGeneration":
        from .modeling_intern_s1 import InternS1ForConditionalGeneration

        return InternS1ForConditionalGeneration(self)


class InternS1Config(InternS1BaseConfig):
    vision_config: InternS1VisionConfig = InternS1VisionConfig(
        hidden_size=3200, intermediate_size=12800, num_hidden_layers=45, use_qk_norm=True, num_attention_heads=25
    )
    projector_config: InternS1ProjectorConfig = InternS1ProjectorConfig(
        vision_hidden_size=3200, text_hidden_size=12800
    )
    text_config: MoEConfig = Qwen3MoE235BA22Config(vocab_size=153216)


class InternS1MiniConfig(InternS1BaseConfig):
    vision_config: InternS1VisionConfig = InternS1VisionConfig()
    projector_config: InternS1ProjectorConfig = InternS1ProjectorConfig()
    text_config: Qwen3Dense8BConfig = Qwen3Dense8BConfig(vocab_size=153216)
