import os
from typing import Dict, Optional, Union

import torch

from transformers import AutoConfig as HfAutoConfig
from transformers import AutoModelForCausalLM as HfAutoModelForCausalLM
from transformers import BitsAndBytesConfig

from xtuner.model.modules.dispatch import SUPPORT_FLASH1, SUPPORT_FLASH2




class AutoModelForCausalLM:

    @classmethod
    def from_config(cls,
                    pretrained_model_name_or_path: str,
                    trust_remote_code: bool = True,
                    **kwargs):
        return HfAutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            **kwargs)

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: str,
            trust_remote_code: bool = True,
            quantization_config: Optional[BitsAndBytesConfig] = None,
            **kwargs):

        config = cls.from_config(
            pretrained_model_name_or_path, trust_remote_code=True)
        attn_kwargs = cls._flash_attn_kwargs(config)
        kwargs.update(attn_kwargs)

        if torch.cuda.is_bf16_supported():
            kwargs.update(torch_dtype=torch.bfloat16)
        else:
            kwargs.update(torch_dtype=torch.float16)

        model = HfAutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            quantization_config=quantization_config,
            **kwargs)

        return model

    @staticmethod
    def _flash_attn_kwargs(config):
        cls_name = type(config).__name__
        _built_in_flash_attn_1 = ('LlamaConfig', 'GemmaConfig',
                                  'MistralConfig', 'MixtralConfig',
                                  'Qwen2Config', 'Starcoder2Config',
                                  'Starcoder2Config')

        _built_in_flash_attn_2 = ('InternLMConfig', 'InternLM2Config',
                                  'LlamaConfig', 'GemmaConfig',
                                  'MistralConfig', 'MixtralConfig',
                                  'Qwen2Config', 'Starcoder2Config',
                                  'Starcoder2Config')

        attn_kwargs = {}
        if SUPPORT_FLASH2 and cls_name in _built_in_flash_attn_2:
            attn_kwargs.update(attn_implementation='flash_attention_2')
        elif SUPPORT_FLASH1 and cls_name in _built_in_flash_attn_1:
            attn_kwargs.update(attn_implementation='sdpa')

        return attn_kwargs