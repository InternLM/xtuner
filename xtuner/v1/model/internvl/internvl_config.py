from pydantic import BaseModel, ConfigDict
from xtuner.v1.config.base_model import MoEConfig


class InternVLVisionConfig(BaseModel):
    model_config = ConfigDict(
        title="Base model config for xtuner",
        extra="allow",
    )
    num_channels = 3
    patch_size = 14
    image_size = 224
    qkv_bias = False
    hidden_size = 3200
    num_attention_heads = 25
    intermediate_size = 12800
    qk_normalization = True
    num_hidden_layers = 48
    use_flash_attn = True
    hidden_act = "gelu"
    norm_type = "rms_norm"
    layer_norm_eps = 1e-6
    dropout = 0.0
    drop_path_rate = 0.0
    attention_dropout = 0.0
    initializer_range = 0.02
    initializer_factor = 0.1


class InternVLConfig(BaseModel):
    model_config = ConfigDict(
        title="Base model config for xtuner",
        extra="allow",
    )
    vision_config: InternVLVisionConfig
    llm_config: MoEConfig
    select_layer = -1
    force_image_size = None
    downsample_ratio = 0.5
    dynamic_image_size = True
    use_thumbnail = True
    min_dynamic_patch = 1
    max_dynamic_patch = 6
