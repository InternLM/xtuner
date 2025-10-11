from pydantic import BaseModel, ConfigDict
from typing_extensions import Self
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional
from xtuner.v1.float8 import Float8Config
from xtuner.v1.model import TransformerConfig, Qwen3MoE30BA3Config
from mmengine import is_installed
from xtuner.v1.utils import get_logger

logger = get_logger()


class Qwen3VLVisionConfig(BaseModel):
    model_config = ConfigDict(
        title="Base model config for xtuner",
        extra="allow",
    )
    num_channels: int = 3
    depth: int = 27
    hidden_size: int = 1152
    num_attention_heads: int = 16
    intermediate_size: int = 4304
    num_hidden_layers: int = 24
    hidden_act: str = "gelu_pytorch_tanh"
    patch_size: int = 16
    spatial_merge_size: int = 2,
    temporal_patch_size: int = 2,
    num_position_embeddings: int = 2304,
    deepstack_visual_indexes: list[int] = [8, 16, 24],
    initializer_range: float = 0.02
    float8_cfg: Optional["Float8Config"] = None
    attn_impl: Literal["flash_attention", "flex_attention", "eager_attention"] = "flash_attention"

    def model_post_init(self, _):
        if not is_installed("flash-attn") and self.attn_impl == "flash_attention":
            logger.warning("flash-attn is not installed, using `flex_attention` instead.")
            self.attn_impl = "flex_attention"
        return self

    def build(self):
        from .modeling_vision import InternS1VisionModel

        return InternS1VisionModel(self)


class Qwen3VLProjectorConfig(BaseModel):
    vision_hidden_size: int = 1152
    text_hidden_size: int = 3584
    spatial_merge_size: int = 2
    use_postshuffle_norm: bool = True
    float8_cfg: Optional["Float8Config"] = None

    def build(self):
        from .modeling_projector import InternS1MultiModalProjector

        return InternS1MultiModalProjector(self)


class Qwen3VLBaseConfig(BaseModel):
    model_config = ConfigDict(
        title="Base model config for xtuner",
        extra="allow",
    )
    vision_config: Qwen3VLVisionConfig
    projector_config: Qwen3VLProjectorConfig
    text_config: TransformerConfig

    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    freeze_vision: bool = False
    freeze_projector: bool = False
    freeze_language: bool = False

    def build(self) -> "InternS1ForConditionalGeneration":
        from .modeling_intern_s1 import InternS1ForConditionalGeneration

        return InternS1ForConditionalGeneration(self)

    @classmethod
    def from_hf(cls, hf_path: str | Path) -> Self:
        raise NotImplementedError


class Qwen3VLMoE30BA3Config(Qwen3VLBaseConfig):
    vision_config: Qwen3VLVisionConfig = Qwen3VLVisionConfig()
    projector_config: Qwen3VLProjectorConfig = Qwen3VLProjectorConfig()
    text_config: Qwen3MoE30BA3Config = Qwen3MoE30BA3Config(
        max_position_embeddings=262144,
        rope_scaling={
            "mrope_interleaved": True,
            "mrope_section": [
                24,
                20,
                20
            ],
            "rope_type": "default"
        },
        rope_theta=5000000
    )
