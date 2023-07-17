import torch
from mmengine.config import read_base
from peft import LoraConfig
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

from mmchat.models import DataProcesorForCausalLM, SupervisedQloraFinetune

with read_base():
    from .._base_.datasets.mmlu_fs import *  # noqa: F401,F403
    from .._base_.datasets.oasst1 import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403
    from .._base_.schedules.guanaco import *  # noqa: F401,F403

pretrained_model_name_or_path = '/nvme/share_data/llama-7b'
model = dict(
    type=SupervisedQloraFinetune,
    data_preprocessor=dict(
        type=DataProcesorForCausalLM,
        tokenizer=dict(
            type=AutoTokenizer.from_pretrained,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            use_fast=False,
            padding_side='right',
        ),
        source_max_len=2048,
        target_max_len=512,
        train_on_source=False,
        predict_with_generate=False,
    ),
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        device_map='auto',
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
        task_type='CAUSAL_LM'))

val_evaluator['tokenizer'] = dict(  # noqa: F405
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    use_fast=False,
    padding_side='right')

test_evaluator['tokenizer'] = dict(  # noqa: F405
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    use_fast=False,
    padding_side='right')
