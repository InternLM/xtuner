# Copyright (c) OpenMMLab. All rights reserved.
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, Trainer, TrainingArguments)

from xtuner.dataset import process_hf_dataset
from xtuner.dataset.map_fns import alpaca_map_fn, template_map_fn_factory
from xtuner.utils import PROMPT_TEMPLATE

framework = 'huggingface'
pretrained_model_name_or_path = 'meta-llama/Llama-2-70b-hf'
dataset_name_or_path = 'garage-bAInd/Open-Platypus'
max_length = 2048
pack_to_max_length = True
prompt_template = PROMPT_TEMPLATE.llama2_chat

trainer = Trainer

training_args = dict(
    type=TrainingArguments,
    do_train=True,
    learning_rate=3e-4,
    weight_decay=0,
    lr_scheduler_type='cosine',
    warmup_steps=100,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    fp16=True,
    logging_steps=1,
    optim='adamw_torch',
    save_strategy='steps',
    save_steps=1000,
    save_total_limit=2,
    ddp_find_unused_parameters=False)

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    trust_remote_code=True,
    padding_side='right')

model = dict(
    type=AutoModelForCausalLM.from_pretrained,
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    quantization_config=dict(
        type=BitsAndBytesConfig,
        load_in_4bit=True,
        load_in_8bit=False,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'))

lora = dict(
    type=LoraConfig,
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=['gate_proj', 'down_proj', 'up_proj'],
    bias='none',
    task_type='CAUSAL_LM')

train_dataset = dict(
    type=process_hf_dataset,
    dataset=dict(type=load_dataset, path=dataset_name_or_path),
    tokenizer=tokenizer,
    max_length=max_length,
    dataset_map_fn=alpaca_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    remove_unused_columns=True,
    shuffle_before_pack=True,
    pack_to_max_length=pack_to_max_length)
