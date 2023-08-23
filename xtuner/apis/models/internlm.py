from mmengine.config import Config

from xtuner.registry import BUILDER
from .base import model_qlora as model_qlora_cfg_dict
from .base import tokenizer as tokenizer_cfg_dict


def internlm_7b_qlora(model_name_or_path=None,
                      quantization_config=None,
                      lora_config=None,
                      return_tokenizer=True):
    if model_name_or_path is None:
        model_name_or_path = 'internlm/internlm-7b'
    model_cfg = Config(model_qlora_cfg_dict)
    model_cfg.llm.pretrained_model_name_or_path = model_name_or_path
    if quantization_config:
        model_cfg.llm.quantization_config = quantization_config
    if lora_config:
        model_cfg.lora = lora_config

    model = BUILDER.build(model_cfg)
    if return_tokenizer:
        tokenizer_cfg = Config(tokenizer_cfg_dict)
        tokenizer_cfg.pretrained_model_name_or_path = model_name_or_path
        tokenizer = BUILDER.build(tokenizer_cfg)
        return model.llm, tokenizer
    else:
        return model.llm


def internlm_chat_7b_qlora(model_name_or_path=None,
                           quantization_config=None,
                           lora_config=None,
                           return_tokenizer=True):
    if model_name_or_path is None:
        model_name_or_path = 'internlm/internlm-chat-7b'
    model_cfg = Config(model_qlora_cfg_dict)
    model_cfg.llm.pretrained_model_name_or_path = model_name_or_path
    if quantization_config:
        model_cfg.llm.quantization_config = quantization_config
    if lora_config:
        model_cfg.lora = lora_config

    model = BUILDER.build(model_cfg)
    if return_tokenizer:
        tokenizer_cfg = Config(tokenizer_cfg_dict)
        tokenizer_cfg.pretrained_model_name_or_path = model_name_or_path
        tokenizer = BUILDER.build(tokenizer_cfg)
        return model.llm, tokenizer
    else:
        return model.llm
