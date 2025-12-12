import types

from .internvl_config import InternVLBaseConfig
from ..intern_s1.modeling_intern_s1 import InternS1ForConditionalGeneration
from ..base import to_hf_key_list_wrapper


def convert_llm_to_hf_keys(key):
    if 'lm_head' in key:
        key = key.replace('lm_head', 'language_model.lm_head')
    else:
        key = key.replace('model.', 'language_model.model.')
    return key


class InternVLForConditionalGeneration(InternS1ForConditionalGeneration):
    def __init__(self, config: InternVLBaseConfig):
        super(InternS1ForConditionalGeneration, self).__init__(config)
        # TODO(YHC): This is a hack to make the language model compatible with HF
        self.language_model.to_hf_key_list = types.MethodType(to_hf_key_list_wrapper(  # type: ignore
            fn=self.language_model.to_hf_key_list,
            convertor=convert_llm_to_hf_keys),
            self.language_model)
        self.language_model._init_load_spec()

        self.img_context_token_id = config.image_token_id
        self.select_layer = config.vision_feature_layer
        self.downsample_ratio = config.downsample_ratio
        self.image_size = config.vision_config.image_size[0]
