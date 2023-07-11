from mmengine.config import read_base
from transformers import AutoModelForCausalLM, AutoTokenizer
from mmchat.models import SupervisedFinetune, DataProcesorForCausalLM
from transformers import BitsAndBytesConfig
from peft import LoraConfig
from dataclasses import dataclass
import torch
with read_base():
    from .._base_.datasets.aplaca import *
    from .._base_.schedules.guanaco import *
    from .._base_.default_runtime import *

model = dict(
    type = SupervisedFinetune,
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
    ),

)

