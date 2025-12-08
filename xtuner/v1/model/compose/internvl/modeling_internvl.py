import types
from pathlib import Path
import torch

from .modeling_vision import InternVLVisionModel
from .modeling_projector import InternVLMultiModalProjector
from .internvl_config import InternVLBaseConfig
from ..intern_s1.modeling_intern_s1 import InternS1ForConditionalGeneration, to_hf_key_list_wrapper


def convert_llm_to_hf_keys(key):
    if 'lm_head' in key:
        key = key.replace('lm_head', 'language_model.lm_head')
    else:
        key = key.replace('model.', 'language_model.model.')
    return key


class InternVLForConditionalGeneration(InternS1ForConditionalGeneration):
    def __init__(self, config: InternVLBaseConfig):
        super(InternS1ForConditionalGeneration, self).__init__(config)
        self.select_layer = config.vision_feature_layer
        self.downsample_ratio = config.downsample_ratio
        self.image_size = config.vision_config.image_size[0]

        vision_config = config.vision_config
        text_config = config.text_config
        projector_config= config.projector_config

        self.vision_tower = InternVLVisionModel(vision_config)
        self.multi_modal_projector = InternVLMultiModalProjector(projector_config)

        self.language_model = text_config.build()

        # TODO(YHC): This is a hack to make the language model compatible with HF
        self.language_model.to_hf_key_list = types.MethodType(to_hf_key_list_wrapper(  # type: ignore
            fn=self.language_model.to_hf_key_list,
            convertor=convert_llm_to_hf_keys),
            self.language_model)
        self.language_model._init_load_spec()

        self.img_context_token_id = config.image_token_id
        self._hf_path: Path | None = None

        # Note: global load spec mapping for save_hf
        self.load_spec_mapping = {}
        for key, value in self.vision_tower.load_spec_mapping.items():
            self.load_spec_mapping['vision_tower.' + key] = value
        for key, value in self.multi_modal_projector.load_spec_mapping.items():
            self.load_spec_mapping['multi_modal_projector.' + key] = value
        for key, value in self.language_model.load_spec_mapping.items():
            self.load_spec_mapping['language_model.' + key] = value

        self._maybe_enable_compile(self.compile_cfg)
        self._freeze_modules()
