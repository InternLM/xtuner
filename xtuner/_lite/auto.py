import math
import os
from typing import Literal, Optional

import torch
from transformers import BitsAndBytesConfig, PretrainedConfig

from xtuner.model.modules.dispatch import SUPPORT_FLASH1, SUPPORT_FLASH2

if os.environ.get('XTUNER_USE_MODELSCOPE'):
    from modelscope import AutoTokenizer  # noqa: F401
    from modelscope import AutoConfig
    from modelscope import AutoModelForCausalLM as OriAutoModelForCausalLM
else:
    from transformers import AutoTokenizer  # noqa: F401
    from transformers import AutoConfig
    from transformers import AutoModelForCausalLM as OriAutoModelForCausalLM


def download_model_from_hub(
    model_name_or_path: str,
    from_hub: Literal['huggingface', 'modelscope'] = 'huggingface',
    cache_dir: Optional[str] = None,
) -> str:
    """Automatically download model from the HUB.

    Note:
        If `model_name_or_path` is a local path, it will return the path
        directly without downloading it again.

    Args:
        model_name_or_path (str): The model name, model path or repo id.
        config (str | None): The config path. Default is None.
        from_hub (str): The model hosting hub, modelscope, or huggingface.
            Default is huggingface.
        cache_dir (str | None):
            The save path when downloading the model. If it is None, it
            will be stored in the default location of the HUB. For
            Huggingface, it's ~/.cache/huggingface/hub, for ModelScope,
            it's ~/.cache/modelscope/hub.

    Returns:
        str: The local path of the model.
    """
    if os.path.isdir(model_name_or_path):
        model_path = model_name_or_path
    elif from_hub == 'huggingface':
        from huggingface_hub import snapshot_download
        model_path = snapshot_download(
            repo_id=model_name_or_path, cache_dir=cache_dir)
    elif from_hub == 'modelscope':
        from modelscope import snapshot_download
        model_path = snapshot_download(
            model_id=model_name_or_path, cache_dir=cache_dir)
    else:
        # TODO support openxlab
        raise NotImplementedError('The model does not support downloading '
                                  f'from {from_hub}, it only supports '
                                  '`huggingface` and `modelscope`.')

    return model_path


class AutoModelForCausalLM:
    """Enhanced version of Huggingface's `AutoModelForCausalLM`.

    Compared to HuggingFace's `AutoModelForCausalLM`, the following three
    features have been added:

        1. Load the model from either HuggingFace or ModelScope based on the
           environment variable `XTUNER_USE_MODELSCOPE` (bool).
        2. Automatically enables Flash Attention. If `flash-attn` is already
           installed, Flash Attention 2 will be used. If there is no
           `flash-attn`, use Flash Attention 1 when torch version is less than
           2.2. When torch version is greater than or equal to 2.2, use Flash
           Attention 2.
        3. When the length of the target sequence during training exceeds the
           maximum length of the original model, the rope scaling is
           automatically set to the `linear` type with a factor of 1."

    Note:
        If the model is built through `from_config`, it will not automatically
        enable flash attention or modify rope scaling.

    Note:
        If you want to load the model on ModelScope, please set the
        environment variable `XTUNER_USE_MODELSCOPE=1`.
    """

    @classmethod
    def from_config(cls,
                    pretrained_model_name_or_path: str,
                    trust_remote_code: bool = True,
                    **kwargs):
        """Consistent with the usage of HuggingFace's AutoModelForCausalLM."""
        return AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            **kwargs)

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: str,
            trust_remote_code: bool = True,
            quantization_config: Optional[BitsAndBytesConfig] = None,
            max_position_embeddings: Optional[int] = None,
            **kwargs):
        """Consistent with the usage of HuggingFace's AutoModelForCausalLM."""
        config = cls.from_config(
            pretrained_model_name_or_path, trust_remote_code=True)

        attn_kwargs = cls._flash_attn_kwargs(config)
        kwargs.update(attn_kwargs)

        if max_position_embeddings:
            long_ctx_kwargs = cls._long_ctx_kwargs(config,
                                                   max_position_embeddings)
            kwargs.update(long_ctx_kwargs)

        if 'torch_dtype' not in kwargs:
            if torch.cuda.is_bf16_supported():
                kwargs.update(torch_dtype=torch.bfloat16)
            else:
                kwargs.update(torch_dtype=torch.float16)

        model = OriAutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            quantization_config=quantization_config,
            **kwargs)

        from xtuner._lite.accelerate import dispatch_modules
        dispatch_modules(model, use_varlen_attn=True)

        return model

    @staticmethod
    def _flash_attn_kwargs(config: PretrainedConfig) -> dict:
        """Arguments Required to Enable Flash Attention."""
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

    @staticmethod
    def _long_ctx_kwargs(config: PretrainedConfig,
                         max_position_embeddings: int) -> dict:
        """Arguments Required for Long Context Training."""
        ori_rope_scaling = getattr(config, 'rope_scaling', None)
        if ori_rope_scaling is None:
            ori_rope_scaling = {'factor': 1}

        if 'factor' in ori_rope_scaling.keys():
            ori_rope_scaling_factor = ori_rope_scaling['factor']
        else:
            ori_rope_scaling_factor = 1

        ori_ctx_len = getattr(config, 'max_position_embeddings', None)

        long_text_kwargs = {}
        if ori_ctx_len:
            ori_ctx_len *= ori_rope_scaling_factor
            if max_position_embeddings > ori_ctx_len:
                scaling_factor = float(
                    math.ceil(max_position_embeddings / ori_ctx_len))

                new_rope_scaling = {'type': 'linear', 'factor': scaling_factor}
                long_text_kwargs.update(dict(rope_scaling=new_rope_scaling))
        return long_text_kwargs
