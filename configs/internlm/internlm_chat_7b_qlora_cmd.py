import torch
from mmengine.config import read_base
from mmengine.model import BaseDataPreprocessor
from peft import LoraConfig
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from mmchat.engine import SampleGenerateHook
from mmchat.models import SupervisedQloraFinetune

with read_base():
    from .._base_.datasets.cmd import *  # noqa: F401,F403
    from .._base_.default_runtime import *  # noqa: F401,F403
    from .._base_.schedules.internlm import *  # noqa: F401,F403

pretrained_model_name_or_path = './models/internlm-chat-7b'
model = dict(
    type=SupervisedQloraFinetune,
    data_preprocessor=dict(type=BaseDataPreprocessor),
    llm=dict(
        type=AutoModel.from_pretrained,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        device_map='auto',
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
        task_type='CAUSAL_LM'))

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    use_fast=False,
    padding_side='right',
    trust_remote_code=True)

train_dataloader.dataset.tokenizer = tokenizer  # noqa: F405

default_hooks.checkpoint.update(  # noqa: F405
    dict(by_epoch=False, interval=500, max_keep_ckpts=2))

custom_hooks = [
    dict(
        type=SampleGenerateHook,
        tokenizer=tokenizer,
        every_n_iters=500,
        sample_inputs=[
            '我有家族遗传性的过敏，请问可以可以献血吗？', '我爷爷有高血压，请问他可以喝咖啡吗？',
            '我女儿今年3岁了，从昨天晚上九点开始腹泻，到现在已经八个小时了，请问应该怎么办？'
        ],
        prompt='请从一名专业医生的角度，对下述医学问题给出安全、可靠的回答\n\n'
        '问：{input}\n\n答：')
]
