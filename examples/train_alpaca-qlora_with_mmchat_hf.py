from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from mmengine.config import Config
from mmengine.runner import Runner
from peft import LoraConfig
from transformers import BitsAndBytesConfig, Trainer

from mmchat.models import SupervisedFinetuneLoRA


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default='facebook/opt-125m')


@dataclass
class DataArguments:
    dataset_cfg_path: str = field(
        default='../configs/alpaca/alpaca_standford_llama-7b.py')


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
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        load_in_8bit=False,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4')
    llm = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        quantization_config=quantization_config,
    )

    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias='none',
        task_type='CAUSAL_LM')

    model = SupervisedFinetuneLoRA(llm=llm, lora=lora_config)

    # build trainer_hf
    dataset_cfg = Config.fromfile(data_args.dataset_cfg_path)
    train_dataloader = Runner.build_dataloader(dataset_cfg.train_dataloader)
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
