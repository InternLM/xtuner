from xtuner.v1.model.base import TransformerConfig
from xtuner.v1.model.moe.qwen3_5_text import Qwen3_5_VLTextMoE35BA3BConfig
from xtuner.v1.utils import get_logger

from ..qwen3_vl.qwen3_vl_config import Qwen3VLVisionConfig, Qwen3VLProjectorConfig, Qwen3VLBaseConfig

logger = get_logger()

class Qwen3_5_VisionConfig(Qwen3VLVisionConfig):
    deepstack_visual_indexes: list[int] = []

class Qwen3_5_ProjectorConfig(Qwen3VLProjectorConfig):
    deepstack_visual_indexes: list[int] = []

class Qwen3_5_BaseConfig(Qwen3VLBaseConfig):
    vision_config: Qwen3_5_VisionConfig
    projector_config: Qwen3_5_ProjectorConfig
    text_config: TransformerConfig

    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054

class Qwen3_5_VLMoE35BA3Config(Qwen3_5_BaseConfig):
    vision_config: Qwen3_5_VisionConfig = Qwen3_5_VisionConfig()
    projector_config: Qwen3_5_ProjectorConfig = Qwen3_5_ProjectorConfig()
    text_config: TransformerConfig = Qwen3_5_VLTextMoE35BA3BConfig()
