from mmengine.config import read_base
from transformers import AutoModelForCausalLM, AutoTokenizer
from mmchat.models import SupervisedQloraFinetune, DataProcesorForCausalLM
from transformers import BitsAndBytesConfig
from peft import LoraConfig
from dataclasses import dataclass
import torch
with read_base():
    from .._base_.datasets.aplaca import *
    from .._base_.schedules.guanaco import *
    from .._base_.default_runtime import *

model = dict(
    type = SupervisedQloraFinetune,
    data_preprocessor = dict(
        type=DataProcesorForCausalLM,
        tokenizer=dict(
            type=AutoTokenizer.from_pretrained,
            pretrained_model_name_or_path = '/share/gaojianfei/merged_chinese_lora_7b',
            use_fast = False,
        ),
        source_max_len = 512,
        target_max_len = 512,
        train_on_source = False,
        predict_with_generate = False,
    ),
    llm = dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path = '/share/gaojianfei/merged_chinese_lora_7b',
        quantization_config=dict(
            type = BitsAndBytesConfig,
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type = 'nf4'
        )
    ),
    lora=dict(
        type=LoraConfig,
        r = 64,
        lora_alpha = 16,
        lora_dropout = 0.1,
        bias = 'none',
        task_type = 'CAUSAL_LM'
    )

)

