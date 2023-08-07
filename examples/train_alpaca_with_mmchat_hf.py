from dataclasses import dataclass, field
from typing import Optional

import transformers
from transformers import Trainer, AutoTokenizer

from mmchat.models import SupervisedFinetune
from data_utils import get_train_dataloader


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default='internlm/internlm-7b')


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
    llm = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,
                                              trust_remote_code=True)

    model = SupervisedFinetune(llm, tokenizer=tokenizer)

    # build trainer_hf
    train_dataloader = get_train_dataloader(data_args.dataset_cfg_path, tokenizer)
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
