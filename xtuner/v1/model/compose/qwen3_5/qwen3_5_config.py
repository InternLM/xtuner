from xtuner.v1.model.dense.qwen3_5_text import Qwen3_5_VLTextDense4BConfig, Qwen3_5_VLTextDenseConfig
from xtuner.v1.model.moe.qwen3_5_text import Qwen3_5_VLTextMoE35BA3BConfig, Qwen3_5_VLTextMoEConfig
from xtuner.v1.model.moe.qwen3_5_text_split import (
    Qwen3_5_VLTextMoE35BA3BSplitConfig,
    Qwen3_5_VLTextMoE397BA17BSplitConfig,
    Qwen3_5_VLTextMoESplitConfig,
)
from xtuner.v1.utils import get_logger

from ..qwen3_vl.qwen3_vl_config import Qwen3VLBaseConfig, Qwen3VLProjectorConfig, Qwen3VLVisionConfig


logger = get_logger()


class Qwen3_5_VisionConfig(Qwen3VLVisionConfig):
    deepstack_visual_indexes: list[int] = []


class Qwen3_5_ProjectorConfig(Qwen3VLProjectorConfig):
    deepstack_visual_indexes: list[int] = []


Qwen3_5TextConfig = Qwen3_5_VLTextDenseConfig | Qwen3_5_VLTextMoEConfig | Qwen3_5_VLTextMoESplitConfig


class Qwen3_5_BaseConfig(Qwen3VLBaseConfig):
    vision_config: Qwen3_5_VisionConfig
    projector_config: Qwen3_5_ProjectorConfig
    text_config: Qwen3_5TextConfig

    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054


class Qwen3_5_VLMoE35BA3Config(Qwen3_5_BaseConfig):
    vision_config: Qwen3_5_VisionConfig = Qwen3_5_VisionConfig()
    projector_config: Qwen3_5_ProjectorConfig = Qwen3_5_ProjectorConfig()
    text_config: Qwen3_5_VLTextMoE35BA3BConfig = Qwen3_5_VLTextMoE35BA3BConfig()


class Qwen3_5_VLDense4BConfig(Qwen3_5_BaseConfig):
    vision_config: Qwen3_5_VisionConfig = Qwen3_5_VisionConfig(depth=24, hidden_size=1024, intermediate_size=4096)
    projector_config: Qwen3_5_ProjectorConfig = Qwen3_5_ProjectorConfig(vision_hidden_size=1024, text_hidden_size=2560)
    text_config: Qwen3_5_VLTextDense4BConfig = Qwen3_5_VLTextDense4BConfig()


class Qwen3_5_VLMoE35BA3SplitConfig(Qwen3_5_BaseConfig):
    vision_config: Qwen3_5_VisionConfig = Qwen3_5_VisionConfig()
    projector_config: Qwen3_5_ProjectorConfig = Qwen3_5_ProjectorConfig()
    text_config: Qwen3_5_VLTextMoE35BA3BSplitConfig = Qwen3_5_VLTextMoE35BA3BSplitConfig(
        hf_key_mapping={r"^model\.": "model.language_model."}
    )


class Qwen3_5TimeSeriesMoE35BA3Config(Qwen3_5_VLMoE35BA3Config):
    time_series_encoder_path: str | None = None
    ts_token_id: int = 248093


class Qwen3_5_VLMoE397BA17SplitConfig(Qwen3_5_BaseConfig):
    vision_config: Qwen3_5_VisionConfig = Qwen3_5_VisionConfig(fully_shard=False)
    projector_config: Qwen3_5_ProjectorConfig = Qwen3_5_ProjectorConfig(text_hidden_size=4096, fully_shard=False)
    text_config: Qwen3_5_VLTextMoE397BA17BSplitConfig = Qwen3_5_VLTextMoE397BA17BSplitConfig(
        hf_key_mapping={r"^model\.": "model.language_model."}
    )
