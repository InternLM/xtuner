from mmengine.config import read_base
from transformers import AutoModelForCausalLM, AutoTokenizer

from mmchat.models import SupervisedFinetune

with read_base():
    from .._base_.datasets.aplaca import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403
    from .._base_.schedules.guanaco import *  # noqa: F401,F403

pretrained_model_name_or_path = '/share/gaojianfei/merged_chinese_lora_7b'
model = dict(
    type=SupervisedFinetune,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=pretrained_model_name_or_path))

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    use_fast=False,
    padding_side='right')

train_dataloader['collate_fn']['tokenizer'] = tokenizer  # noqa: F405
