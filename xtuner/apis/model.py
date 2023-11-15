# Copyright (c) OpenMMLab. All rights reserved.
import torch
from peft import LoraConfig
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from xtuner.model import SupervisedFinetune

__all__ = ['build_model', 'build_lora_model', 'build_qlora_model']


def build_qlora_model(model_name_or_path,
                      quantization_config=None,
                      lora_config=None,
                      return_tokenizer=True):

    if quantization_config is None:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4')
    if lora_config is None:
        lora_config = LoraConfig(
            r=64,
            lora_alpha=16,
            lora_dropout=0.1,
            bias='none',
            task_type='CAUSAL_LM')

    llm = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        quantization_config=quantization_config)

    model = SupervisedFinetune(llm, lora=lora_config)

    if return_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            encode_special_tokens=True)
        return model.llm, tokenizer
    else:
        return model.llm


def build_lora_model(model_name_or_path,
                     lora_config=None,
                     return_tokenizer=True):
    if lora_config is None:
        lora_config = LoraConfig(
            r=64,
            lora_alpha=16,
            lora_dropout=0.1,
            bias='none',
            task_type='CAUSAL_LM')

    llm = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch.float16, trust_remote_code=True)

    model = SupervisedFinetune(llm, lora=lora_config)

    if return_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            encode_special_tokens=True)
        return model.llm, tokenizer
    else:
        return model.llm


def build_model(model_name_or_path, return_tokenizer=True):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch.float16, trust_remote_code=True)

    if return_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            encode_special_tokens=True)
        return model, tokenizer
    else:
        return model
