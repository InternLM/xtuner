import os
from typing import Literal, Optional

import torch
from mmengine import Config, print_log
from mmengine.config.lazy import LazyAttr
from mmengine.runner import find_latest_checkpoint
from transformers import AutoConfig as HfAutoConfig
from transformers import AutoModelForCausalLM as HfAutoModelForCausalLM
from transformers import BitsAndBytesConfig

from xtuner.model.base import BaseAlgorithm
from xtuner.model.modules.dispatch import SUPPORT_FLASH1, SUPPORT_FLASH2
from xtuner.registry import BUILDER


def download_model_from_hub(model_name_or_path: str,
                            from_hub: Literal['huggingface',
                                              'modelscope'] = 'huggingface'):
    if os.path.isdir(model_name_or_path):
        model_name_or_path = model_name_or_path
    elif from_hub == 'huggingface':
        from huggingface_hub import snapshot_download
        model_name_or_path = snapshot_download(repo_id=model_name_or_path)
    elif from_hub == 'modelscope':
        from modelscope import snapshot_download
        model_name_or_path = snapshot_download(model_id=model_name_or_path)
    else:
        # TODO support openxlab
        raise NotImplementedError

    return model_name_or_path


class AutoAlgorithm():

    @classmethod
    def from_config(self, config: str):
        if isinstance(config, str):
            config = Config.fromfile(config)
        model: BaseAlgorithm = BUILDER.build(config.model)
        return model

    @classmethod
    def from_workdir(cls, workdir: str):

        config = [f for f in os.listdir(workdir) if f.endswith('.py')]
        assert len(config) == 1

        checkpoint = find_latest_checkpoint(workdir)
        if checkpoint is None:
            raise RuntimeError

        config = Config.fromfile(config[0])
        model: BaseAlgorithm = BUILDER.build(config.model)
        model.load_checkpoint(checkpoint)
        print_log(f'Auto loaded from {checkpoint}.', logger='current')

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str,
        config: Optional[str] = None,
        from_hub: Literal['huggingface', 'modelscope'] = 'huggingface'
    ) -> BaseAlgorithm:
        checkpoint = download_model_from_hub(checkpoint, from_hub)
        xtuner_conf = os.path.join(checkpoint, 'xtuner_config.py')
        has_conf = os.path.exists(xtuner_conf)

        if config and has_conf:
            # TODO add warning
            conf_path = config
        elif config and not has_conf:
            conf_path = config
        elif not config and has_conf:
            conf_path = xtuner_conf
        else:
            raise RuntimeError

        config = Config.fromfile(conf_path)

        model_cls = config.model.type
        if isinstance(model_cls, LazyAttr):
            model_cls = model_cls.build()

        if not issubclass(model_cls, BaseAlgorithm):
            raise TypeError

        return model_cls.from_pretrained(checkpoint, conf_path, from_hub)


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


if __name__ == '__main__':

    config = 'xtuner/configs/internlm/internlm2_chat_1_8b/example.py'
    model = AutoAlgorithm.from_config(config)
    model.save_pretrained('test_saved', config)
    model.cuda()
    print(model.chat('Hello'))

    model = AutoAlgorithm.from_pretrained('test_saved')
    model.cuda()
    print(model.chat('Hello'))
