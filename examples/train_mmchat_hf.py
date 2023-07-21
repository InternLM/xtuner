#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from mmengine.config import Config
from mmengine.runner import Runner
from mmchat.models.algorithms import SupervisedFinetune
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import transformers
from transformers import Trainer

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    # data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    dataset_cfg_path: str = field(default='/home/humu/MMChat/configs/alpaca/alpaca_standford.py')


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    llm = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    model = SupervisedFinetune(llm)

    dataset_cfg = Config.fromfile(data_args.dataset_cfg_path)
    train_dataloader = Runner.build_dataloader(dataset_cfg.train_dataloader)
    train_dataset = train_dataloader.dataset
    data_collator = train_dataloader.collate_fn
    data_module = dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
    trainer = Trainer(model=model, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
