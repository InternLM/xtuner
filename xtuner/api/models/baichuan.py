from mmengine.config import Config

from xtuner.registry import MODELS, TOKENIZER
from .base import model_qlora as model_qlora_cfg_dict
from .base import tokenizer as tokenizer_cfg_dict


def baichuan_7b_qlora(llm_kwargs={},
                      lora_kwargs={},
                      tokenizer_kwargs={},
                      return_tokenizer=True):
    default_name = 'baichuan-inc/Baichuan-7B'
    if 'pretrained_model_name_or_path' not in llm_kwargs:
        llm_kwargs['pretrained_model_name_or_path'] = default_name
    if 'pretrained_model_name_or_path' not in tokenizer_kwargs:
        tokenizer_kwargs['pretrained_model_name_or_path'] = default_name
    model_cfg = Config(model_qlora_cfg_dict)
    model_cfg.llm.update(llm_kwargs)
    model_cfg.lora.update(lora_kwargs)
    model = MODELS.build(model_cfg)
    if return_tokenizer:
        tokenizer_cfg = Config(tokenizer_cfg_dict)
        tokenizer_cfg.update(tokenizer_kwargs)
        tokenizer = TOKENIZER.build(tokenizer_cfg)
        return model.llm, tokenizer
    else:
        return model.llm


def baichuan_13b_base_qlora(llm_kwargs={},
                            lora_kwargs={},
                            tokenizer_kwargs={},
                            return_tokenizer=True):
    default_name = 'baichuan-inc/Baichuan-13B-Base'
    if 'pretrained_model_name_or_path' not in llm_kwargs:
        llm_kwargs['pretrained_model_name_or_path'] = default_name
    if 'pretrained_model_name_or_path' not in tokenizer_kwargs:
        tokenizer_kwargs['pretrained_model_name_or_path'] = default_name
    model_cfg = Config(model_qlora_cfg_dict)
    model_cfg.llm.update(llm_kwargs)
    model_cfg.lora.update(lora_kwargs)
    model = MODELS.build(model_cfg)
    if return_tokenizer:
        tokenizer_cfg = Config(tokenizer_cfg_dict)
        tokenizer_cfg.update(tokenizer_kwargs)
        tokenizer = TOKENIZER.build(tokenizer_cfg)
        return model.llm, tokenizer
    else:
        return model.llm


def baichuan_13b_chat_qlora(llm_kwargs={},
                            lora_kwargs={},
                            tokenizer_kwargs={},
                            return_tokenizer=True):
    default_name = 'baichuan-inc/Baichuan-13B-Chat'
    if 'pretrained_model_name_or_path' not in llm_kwargs:
        llm_kwargs['pretrained_model_name_or_path'] = default_name
    if 'pretrained_model_name_or_path' not in tokenizer_kwargs:
        tokenizer_kwargs['pretrained_model_name_or_path'] = default_name
    model_cfg = Config(model_qlora_cfg_dict)
    model_cfg.llm.update(llm_kwargs)
    model_cfg.lora.update(lora_kwargs)
    model = MODELS.build(model_cfg)
    if return_tokenizer:
        tokenizer_cfg = Config(tokenizer_cfg_dict)
        tokenizer_cfg.update(tokenizer_kwargs)
        tokenizer = TOKENIZER.build(tokenizer_cfg)
        return model.llm, tokenizer
    else:
        return model.llm
