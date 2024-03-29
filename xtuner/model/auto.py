import os
from typing import Dict, Optional, Union

import torch
from mmengine import Config, print_log
from mmengine.runner import find_latest_checkpoint
from transformers import AutoConfig as HfAutoConfig
from transformers import AutoModelForCausalLM as HfAutoModelForCausalLM
from transformers import BitsAndBytesConfig

from xtuner.model.base import BaseTune
from xtuner.model.modules.dispatch import SUPPORT_FLASH1, SUPPORT_FLASH2
from xtuner.registry import BUILDER


def download_model_from_hub(model_name_or_path, from_hub: str = 'huggingface'):
    if os.path.isdir(model_name_or_path):
        model_name_or_path = model_name_or_path
    elif from_hub == 'huggingface' or from_hub is True:
        from huggingface_hub import snapshot_download
        model_name_or_path = snapshot_download(repo_id=model_name_or_path)
    elif from_hub == 'modelscope':
        from modelscope import snapshot_download
        model_name_or_path = snapshot_download(model_id=model_name_or_path)
    else:
        raise NotImplementedError

    return model_name_or_path


class AutoXTunerModel():

    @classmethod
    def from_config(self, config: Union[str, Dict]):
        if isinstance(config, str):
            config = Config.fromfile(config)
        model: BaseTune = BUILDER.build(config.model)
        return model

    @classmethod
    def _from_mmengine_work_dir(cls, work_dir: str):

        config = [f for f in os.listdir(work_dir) if f.endswith('.py')]
        assert len(config) == 1

        checkpoint = find_latest_checkpoint(work_dir)
        if checkpoint is None:
            raise RuntimeError

        config = Config.fromfile(config[0])
        model: BaseTune = BUILDER.build(config.model)
        model.load_checkpoint(checkpoint)
        print_log(f'Auto loaded from {checkpoint}.', logger='current')

    @classmethod
    def from_pretrained(cls,
                        checkpoint: str,
                        config: Optional[str] = None,
                        from_hub: bool = False):
        config = Config.fromfile(config)

        # Huggingface format
        if from_hub:
            checkpoint = download_model_from_hub(checkpoint, from_hub)

            if os.path.exists(os.path.join(checkpoint, 'config.py')):
                config = Config.fromfile(os.path.join(checkpoint, 'config.py'))
            elif config is not None:
                config = config = Config.fromfile(config)
            else:
                raise RuntimeError

            model: BaseTune = BUILDER.build(config.model)
            model.load_checkpoint(checkpoint, from_hub)
            print_log(f'Auto loaded from {checkpoint}.', logger='current')

        elif checkpoint.endswith('.pth'):
            config = Config.fromfile(config)
            model: BaseTune = BUILDER.build(config.model)
            model.load_checkpoint(checkpoint)
            print_log(f'Auto loaded from {checkpoint}.', logger='current')
        elif os.path.isdir(checkpoint):
            model = cls._from_mmengine_work_dir(checkpoint)
        else:
            raise RuntimeError

        return model


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

    from transformers import AutoTokenizer

    from xtuner.model import ChatFinetune
    from xtuner.types import ChatMessages, ChatTemplate

    model_name = 'internlm/internlm2-chat-1_8b'

    config = dict(
        type=ChatFinetune,
        tokenizer=dict(
            type=AutoTokenizer.from_pretrained,
            pretrained_model_name_or_path=model_name,
            trust_remote_code=True,
            padding_side='right'),
        chat_template=dict(
            type=ChatTemplate,
            system='<|im_start|>system\n{system}<|im_end|>\n',
            user='<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n',
            assistant='{assistant}<|im_end|>\n',
            stop_words=['<|im_end|>']),
        llm=dict(
            type=AutoModelForCausalLM.from_pretrained,
            pretrained_model_name_or_path=model_name,
            trust_remote_code=True))

    model = AutoXTunerModel.from_config(config)

    data = {
        'messages': [
            {
                'role': 'user',
                'content': 'hello'
            },
        ]
    }

    messages = ChatMessages.from_dict(data)

    model.cuda()
    response = model.chat(messages)

    print(response)
