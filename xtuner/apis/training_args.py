# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass, field
from typing import Union

from transformers import TrainingArguments
from transformers.trainer_utils import IntervalStrategy, SchedulerType

__all__ = ['DefaultTrainingArguments']


@dataclass
class DefaultTrainingArguments(TrainingArguments):
    # custom
    model_name_or_path: str = field(
        default=None,
        metadata={'help': 'model name or path.'},
    )
    dataset_name_or_path: str = field(
        default=None,
        metadata={'help': 'dataset name or path.'},
    )

    # huggingface
    default_output_dir = './work_dirs'
    default_do_train = True
    default_per_device_train_batch_size = 1
    default_learning_rate = 2e-5
    default_save_strategy = 'epoch'
    default_lr_scheduler_type = 'cosine'
    default_logging_steps = 5

    output_dir: str = field(
        default=default_output_dir,
        metadata={
            'help': ('The output directory where the model predictions and '
                     'checkpoints will be written.')
        })
    do_train: bool = field(
        default=default_do_train,
        metadata={'help': 'Whether to run training.'})
    per_device_train_batch_size: int = field(
        default=default_per_device_train_batch_size,
        metadata={'help': 'Batch size per GPU/TPU core/CPU for training.'})
    learning_rate: float = field(
        default=default_learning_rate,
        metadata={'help': 'The initial learning rate for AdamW.'})
    save_strategy: Union[IntervalStrategy, str] = field(
        default=default_save_strategy,
        metadata={'help': 'The checkpoint save strategy to use.'},
    )
    lr_scheduler_type: Union[SchedulerType, str] = field(
        default=default_lr_scheduler_type,
        metadata={'help': 'The scheduler type to use.'},
    )
    logging_steps: float = field(
        default=default_logging_steps,
        metadata={
            'help': ('Log every X updates steps. Should be an integer or a '
                     'float in range `[0,1)`. If smaller than 1, will be '
                     'interpreted as ratio of total training steps.')
        })
