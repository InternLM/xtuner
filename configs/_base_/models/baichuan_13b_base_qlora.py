import torch
from mmengine.model import BaseDataPreprocessor
from peft import LoraConfig
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from mmchat.models import SupervisedFinetune

pretrained_model_name_or_path = 'baichuan-inc/Baichuan-13B-Base'

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    padding_side='right',
    trust_remote_code=True)

model = dict(
    type=SupervisedFinetune,
    data_preprocessor=dict(type=BaseDataPreprocessor),
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        quantization_config=dict(
            type=BitsAndBytesConfig,
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4')),
    lora=dict(
        type=LoraConfig,
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias='none',
        task_type='CAUSAL_LM'),
    tokenizer=tokenizer)
