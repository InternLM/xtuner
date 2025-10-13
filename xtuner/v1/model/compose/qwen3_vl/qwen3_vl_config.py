from pydantic import BaseModel, ConfigDict
from typing_extensions import Self
from pathlib import Path
from typing import Literal, Optional
from xtuner.v1.float8 import Float8Config
from xtuner.v1.model.base import TransformerConfig
from xtuner.v1.model.moe.qwen3vl_text import Qwen3VLTextMoE30BA3Config
from mmengine import is_installed
from xtuner.v1.utils import get_logger
from xtuner.v1.module.rope import RopeScalingConfig

logger = get_logger()


class Qwen3VLVisionConfig(BaseModel):
    model_config = ConfigDict(
        title="Base model config for xtuner",
        extra="allow",
    )
    in_channels: int = 3
    depth: int = 27
    hidden_size: int = 1152
    num_attention_heads: int = 16
    intermediate_size: int = 4304
    num_hidden_layers: int = 24
    hidden_act: str = "gelu_pytorch_tanh"
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    num_position_embeddings: int = 2304
    deepstack_visual_indexes: list[int] = [8, 16, 24]
    initializer_range: float = 0.02
    float8_cfg: Optional["Float8Config"] = None
    attn_impl: Literal["flash_attention", "flex_attention", "eager_attention"] = "flash_attention"

    def model_post_init(self, _):
        if not is_installed("flash-attn") and self.attn_impl == "flash_attention":
            logger.warning("flash-attn is not installed, using `flex_attention` instead.")
            self.attn_impl = "flex_attention"
        return self

    def build(self):
        from .modeling_vision import Qwen3VLVisionModel

        return Qwen3VLVisionModel(self)


class Qwen3VLProjectorConfig(BaseModel):
    vision_hidden_size: int = 1152
    text_hidden_size: int = 2048
    spatial_merge_size: int = 2
    deepstack_visual_indexes: list[int] = [8, 16, 24]
    float8_cfg: Optional["Float8Config"] = None

    def build(self):
        from .modeling_projector import Qwen3VLProjector

        return Qwen3VLProjector(self)


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

    def build(self):
        from .modeling_qwen3_vl import Qwen3VLForConditionalGeneration

        return Qwen3VLForConditionalGeneration(self)

    @classmethod
    def from_hf(cls, hf_path: str | Path) -> Self:
        raise NotImplementedError


class Qwen3VLMoE30BA3Config(Qwen3VLBaseConfig):
    vision_config: Qwen3VLVisionConfig = Qwen3VLVisionConfig()
    projector_config: Qwen3VLProjectorConfig = Qwen3VLProjectorConfig()
    text_config: Qwen3VLTextMoE30BA3Config = Qwen3VLTextMoE30BA3Config(
        rope_type='qwen3_vl',
        max_position_embeddings=262144,
        rope_theta=5000000,
        rope_scaling_cfg=RopeScalingConfig(rope_type='qwen3_vl',
                                           mrope_section=[24, 20, 20])
    )
