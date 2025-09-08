from torch import nn

from .internvl_config import InternVLProjectorConfig
from ..intern_s1.modeling_projector import InternS1MultiModalProjector

try:
    from transformers.models.internvl.modeling_internvl import ACT2FN
except:
    ACT2FN = None


class InternVLMultiModalProjector(InternS1MultiModalProjector):
    config: InternVLProjectorConfig

    def __init__(self, config: InternVLProjectorConfig):
        super(InternS1MultiModalProjector, self).__init__()
        self.layer_norm = nn.LayerNorm(config.vision_config.hidden_size * int(1 / config.downsample_ratio) ** 2)
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size * int(1 / config.downsample_ratio) ** 2, config.text_config.hidden_size
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size)

        self._hf_prefix = "multi_modal_projector."
        self._init_load_spec()
