# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from data_utils import get_train_dataloader
from peft import LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig, Trainer

from xtuner.models import SupervisedFinetune


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default='internlm/internlm-7b')
    use_qlora: bool = field(default=True)
    use_lora: bool = field(default=False)


@dataclass
class DataArguments:
    dataset_cfg_path: str = field(
        default='../configs/_base_/datasets/alpaca.py')


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default='adamw_torch')
    model_max_length: int = field(
        default=512,
        metadata={
            'help':
            'Maximum sequence length. Sequences will be right padded '
            '(and possibly truncated).'
        },
    )


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # build model
    if model_args.use_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4')
        lora_config = LoraConfig(
            r=64,
            lora_alpha=16,
            lora_dropout=0.1,
            bias='none',
            task_type='CAUSAL_LM')
    elif not model_args.use_qlora and model_args.use_lora:
        quantization_config = None
        lora_config = LoraConfig(
            r=64,
            lora_alpha=16,
            lora_dropout=0.1,
            bias='none',
            task_type='CAUSAL_LM')
    else:
        quantization_config = None
        lora_config = None

    llm = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        quantization_config=quantization_config,
        trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True)

    model = SupervisedFinetune(llm=llm, lora=lora_config, tokenizer=tokenizer)

    # build trainer_hf
    train_dataloader = get_train_dataloader(data_args.dataset_cfg_path,
                                            tokenizer)
    train_dataset = train_dataloader.dataset
    data_collator = train_dataloader.collate_fn
    data_module = dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator)
    trainer = Trainer(model=model, args=training_args, **data_module)

    # training
    trainer.train()

    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == '__main__':
    train()
