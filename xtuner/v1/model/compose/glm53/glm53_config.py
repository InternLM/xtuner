# Copyright (c) OpenMMLab. All rights reserved.
"""GLM-5.3-Flash vision configs, see doc/xtuner_glm5p3flash_design.md F2
§5.1/§3.7.4.

Field names/defaults are checked against the real checkpoint's ``vision_config`` (transformers
5.17.0, `zai-org/GLM-5.3-Flash`), not copied from the design doc's placeholder table:

- the checkpoint field is ``num_heads``, not ``num_attention_heads`` (HF aliases the latter via
  ``attribute_map``; XTuner has no such alias mechanism, so the field here is named to match the
  checkpoint directly);
- the checkpoint has no ``rope_parameters`` key at all -- HF's ``AutoConfig`` fills the default
  ``{"rope_theta": 10000.0, "rope_type": "axial"}`` when absent, so that's the default here too
  (declaring it required would block construction from the real config.json);
- the checkpoint field is ``out_hidden_size``, not ``text_hidden_size``.
"""

from pathlib import Path
from typing import Literal

from pydantic import ConfigDict
from typing_extensions import Self

from xtuner.v1.model.base import XTunerBaseModelConfig
from xtuner.v1.model.compose.base import BaseComposeConfig
from xtuner.v1.model.moe.glm53.glm53 import Glm53TextMoEConfig


class Glm53VisionConfig(XTunerBaseModelConfig):
    model_config = ConfigDict(title="GLM-5.3-Flash vision config for xtuner", extra="forbid")

    in_channels: int = 3
    depth: int = 24
    hidden_size: int = 1024
    num_heads: int = 16
    intermediate_size: int = 4096
    patch_size: int = 14
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    rms_norm_eps: float = 1e-5
    hidden_act: str = "silu"
    swiglu_limit: float = 10.0
    rope_parameters: dict = {"rope_theta": 10000.0, "rope_type": "axial"}
    attention_bias: bool = True
    attention_dropout: float = 0.0
    attn_impl: Literal["flash_attention", "flex_attention", "eager_attention"] = "flash_attention"
    fully_shard: bool = True

    def build(self):
        from .modeling_vision import Glm53VisionModel

        return Glm53VisionModel(self)

    @property
    def hf_config(self):
        return None


class Glm53ProjectorConfig(XTunerBaseModelConfig):
    model_config = ConfigDict(title="GLM-5.3-Flash projector config for xtuner", extra="forbid")

    vision_hidden_size: int = 1024
    out_hidden_size: int = 4096
    spatial_merge_size: int = 2
    projection_intermediate_size: int = 10240
    hidden_act: str = "silu"
    swiglu_limit: float = 10.0
    fully_shard: bool = True

    def build(self):
        from .modeling_projector import Glm53Projector

        return Glm53Projector(self)

    @property
    def hf_config(self):
        return None


class Glm53BaseConfig(BaseComposeConfig):
    """GLM-5.3-Flash compose config, see doc/xtuner_glm5p3flash_design.md F6.

    ``image_token_id``/``video_start_token_id``/``video_end_token_id`` are kept only for
    reference/debugging; the splice itself uses the global ``mm_token_type_ids`` (produced by
    ``Glm53VLTokenizeFunction`` via the real HF processor's own ``create_mm_token_type_ids``, see
    F1.b) to separate image (1) from video (2) positions, never
    ``input_ids == video_token_id`` -- that token never appears in the expanded sequence.
    """

    model_config = ConfigDict(title="GLM-5.3-Flash compose config for xtuner", extra="forbid")
    vision_config: Glm53VisionConfig = Glm53VisionConfig()
    projector_config: Glm53ProjectorConfig = Glm53ProjectorConfig()
    text_config: Glm53TextMoEConfig = Glm53TextMoEConfig()

    image_token_id: int = 154854
    video_token_id: int = 154855
    video_start_token_id: int = 154832
    video_end_token_id: int = 154833
    only_llm_forward: bool = False

    def build(self):
        from .modeling_glm53 import Glm53ForConditionalGeneration

        return Glm53ForConditionalGeneration(self)

    @classmethod
    def from_hf(cls, hf_path: str | Path) -> Self:
        raise NotImplementedError

    @property
    def hf_config(self):
        from xtuner.v1.utils import log_rank0

        log_rank0.warning(
            f"{type(self)} does not support conversion to HuggingFace config format. Only the "
            "original HuggingFace config will be retained in the saved HuggingFace format checkpoint."
        )
        return None
