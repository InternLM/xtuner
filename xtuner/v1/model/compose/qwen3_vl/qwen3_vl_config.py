from pathlib import Path
from typing import Literal

from mmengine import is_installed
from pydantic import ConfigDict
from typing_extensions import Self

from xtuner.v1.model.base import TransformerConfig, XTunerBaseModelConfig
from xtuner.v1.model.dense.qwen3vl_text import Qwen3VLTextDense4BConfig, Qwen3VLTextDense8BConfig
from xtuner.v1.model.moe.qwen3 import Qwen3MoE235BA22Config
from xtuner.v1.model.moe.qwen3vl_text import Qwen3VLTextMoE30BA3Config, Qwen3VLTextMoE235BA22Config
from xtuner.v1.module.rope import RopeScalingConfig
from xtuner.v1.utils import get_device, get_logger

from ..base import BaseComposeConfig


logger = get_logger()


class Qwen3VLVisionConfig(XTunerBaseModelConfig):
    model_config = ConfigDict(
        title="Base model config for xtuner",
        extra="forbid",
    )
    in_channels: int = 3
    depth: int = 27
    hidden_size: int = 1152
    num_attention_heads: int = 16
    intermediate_size: int = 4304
    hidden_act: str = "gelu_pytorch_tanh"
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    num_position_embeddings: int = 2304
    deepstack_visual_indexes: list[int] = [8, 16, 24]
    initializer_range: float = 0.02
    attn_impl: Literal["flash_attention", "flex_attention", "eager_attention"] = "flash_attention"

    def model_post_init(self, _):
        if not is_installed("flash-attn") and self.attn_impl == "flash_attention" and get_device() == "cuda":
            logger.warning("flash-attn is not installed, using `flex_attention` instead.")
            self.attn_impl = "flex_attention"
        return self

    def build(self):
        from .modeling_vision import Qwen3VLVisionModel

        return Qwen3VLVisionModel(self)

    @property
    def hf_config(self):
        return None


class Qwen3VLProjectorConfig(XTunerBaseModelConfig):
    model_config = ConfigDict(extra="forbid")
    vision_hidden_size: int = 1152
    text_hidden_size: int = 2048
    spatial_merge_size: int = 2
    deepstack_visual_indexes: list[int] = [8, 16, 24]

    def build(self):
        from .modeling_projector import Qwen3VLProjector

        return Qwen3VLProjector(self)

    @property
    def hf_config(self):
        return None


class Qwen3VLBaseConfig(BaseComposeConfig):
    model_config = ConfigDict(
        title="Base model config for xtuner",
        extra="forbid",
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
    dcp_ignore_frozen_params: bool = True

    def build(self):
        from .modeling_qwen3_vl import Qwen3VLForConditionalGeneration

        return Qwen3VLForConditionalGeneration(self)

    @classmethod
    def from_hf(cls, hf_path: str | Path) -> Self:
        raise NotImplementedError

    @property
    def hf_config(self):
        # TODO(pppppM) Support saving HuggingFace format config
        logger.warning(
            f"{type(self)} does not support conversion to HuggingFace config format. "
            "Only the original HuggingFace config will be retained in the saved HuggingFace format checkpoint. "
            f"If you have changed the default values in {type(self)}, it may cause the config in the saved "
            "HuggingFace format checkpoint to not match the weights."
        )
        return None


class Qwen3VLMoE30BA3Config(Qwen3VLBaseConfig):
    vision_config: Qwen3VLVisionConfig = Qwen3VLVisionConfig()
    projector_config: Qwen3VLProjectorConfig = Qwen3VLProjectorConfig()
    text_config: Qwen3VLTextMoE30BA3Config = Qwen3VLTextMoE30BA3Config(
        max_position_embeddings=262144,
        rope_theta=5000000,
        rope_scaling_cfg=RopeScalingConfig(type="qwen3_vl", mrope_section=[24, 20, 20]),
    )


class Qwen3VLMoE235BA22Config(Qwen3VLBaseConfig):
    vision_config: Qwen3VLVisionConfig = Qwen3VLVisionConfig()
    projector_config: Qwen3VLProjectorConfig = Qwen3VLProjectorConfig(text_hidden_size=4096)
    text_config: Qwen3MoE235BA22Config = Qwen3VLTextMoE235BA22Config(
        max_position_embeddings=262144,
        rope_theta=5000000,
        rope_scaling_cfg=RopeScalingConfig(type="qwen3_vl", mrope_section=[24, 20, 20]),
    )


class Qwen3VLDense4BConfig(Qwen3VLBaseConfig):
    vision_config: Qwen3VLVisionConfig = Qwen3VLVisionConfig(
        depth=24, hidden_size=1024, intermediate_size=4096, deepstack_visual_indexes=[5, 11, 17]
    )
    projector_config: Qwen3VLProjectorConfig = Qwen3VLProjectorConfig(
        vision_hidden_size=1024, text_hidden_size=2560, deepstack_visual_indexes=[5, 11, 17]
    )
    text_config: Qwen3VLTextDense4BConfig = Qwen3VLTextDense4BConfig(
        max_position_embeddings=262144,
        rope_theta=5000000,
        rope_scaling_cfg=RopeScalingConfig(type="qwen3_vl", mrope_section=[24, 20, 20]),
    )


class Qwen3VLDense8BConfig(Qwen3VLBaseConfig):
    vision_config: Qwen3VLVisionConfig = Qwen3VLVisionConfig()
    projector_config: Qwen3VLProjectorConfig = Qwen3VLProjectorConfig(text_hidden_size=4096)
    text_config: Qwen3VLTextDense8BConfig = Qwen3VLTextDense8BConfig(
        max_position_embeddings=262144,
        rope_theta=5000000,
        rope_scaling_cfg=RopeScalingConfig(type="qwen3_vl", mrope_section=[24, 20, 20]),
    )
