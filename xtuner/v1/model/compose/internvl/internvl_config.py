from typing import TYPE_CHECKING, Literal, Optional

from mmengine import is_installed
from pydantic import BaseModel, ConfigDict

from xtuner.v1.float8 import Float8Config
from xtuner.v1.model.dense.qwen3 import Qwen3Dense0P6BConfig, Qwen3Dense8BConfig
from xtuner.v1.model.moe.moe import TransformerConfig
from xtuner.v1.model.moe.qwen3 import Qwen3MoE30BA3Config
from xtuner.v1.utils import get_device, get_logger


if TYPE_CHECKING:
    from .modeling_internvl import InternVLForConditionalGeneration

logger = get_logger()


class InternVLVisionConfig(BaseModel):
    model_config = ConfigDict(
        title="Base model config for xtuner",
        extra="forbid",
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
    attn_impl: Literal["flash_attention", "flex_attention", "eager_attention"] = "flash_attention"

    def model_post_init(self, _):
        if not is_installed("flash-attn") and self.attn_impl == "flash_attention" and get_device() == "cuda":
            logger.warning("flash-attn is not installed, using `flex_attention` instead.")
            self.attn_impl = "flex_attention"
        return self

    def build(self):
        from .modeling_vision import InternVLVisionModel

        return InternVLVisionModel(self)


class InternVLProjectorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    vision_hidden_size: int = 1024
    text_hidden_size: int = 4096
    downsample_ratio: float = 0.5
    hidden_act: str = "gelu"
    float8_cfg: Optional["Float8Config"] = None

    def build(self):
        from .modeling_projector import InternVLMultiModalProjector

        return InternVLMultiModalProjector(self)


class InternVLBaseConfig(BaseModel):
    model_config = ConfigDict(
        title="Base model config for xtuner",
        extra="forbid",
    )
    vision_config: InternVLVisionConfig
    projector_config: InternVLProjectorConfig
    text_config: TransformerConfig

    vision_feature_layer: int = -1
    downsample_ratio: float = 0.5
    dynamic_image_size: bool = True
    use_thumbnail: bool = True
    min_dynamic_patch: int = 1
    max_dynamic_patch: int = 12
    image_token_id: int = 151671
    freeze_vision: bool = False
    freeze_projector: bool = False
    freeze_language: bool = False
    hf_save_worker: int = 16
    dcp_ignore_frozen_params: bool = True

    def build(self) -> "InternVLForConditionalGeneration":
        from .modeling_internvl import InternVLForConditionalGeneration

        return InternVLForConditionalGeneration(self)


class InternVL3P5Dense8BConfig(InternVLBaseConfig):
    vision_config: InternVLVisionConfig = InternVLVisionConfig()
    projector_config: InternVLProjectorConfig = InternVLProjectorConfig()
    text_config: Qwen3Dense8BConfig = Qwen3Dense8BConfig()

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


class InternVL3P5MoE30BA3Config(InternVLBaseConfig):
    vision_config: InternVLVisionConfig = InternVLVisionConfig()
    projector_config: InternVLProjectorConfig = InternVLProjectorConfig(text_hidden_size=2049)
    text_config: Qwen3MoE30BA3Config = Qwen3MoE30BA3Config()

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


class InternVL3P5Dense1BConfig(InternVLBaseConfig):
    vision_config: InternVLVisionConfig = InternVLVisionConfig()
    projector_config: InternVLProjectorConfig = InternVLProjectorConfig(text_hidden_size=1024)
    text_config: Qwen3Dense0P6BConfig = Qwen3Dense0P6BConfig()

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
