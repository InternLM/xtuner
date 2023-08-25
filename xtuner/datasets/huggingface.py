# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial

import numpy as np
from datasets import DatasetDict
from mmengine.config import Config, ConfigDict

from xtuner.registry import BUILDER
from .map_fns import prompt_template_map_fn
from .utils import Packer, encode_fn


def process_hf_dataset(dataset,
                       tokenizer,
                       max_length,
                       dataset_map_fn=None,
                       prompt_template=None,
                       max_dataset_length=None,
                       split='train',
                       remove_unused_columns=False,
                       rename_maps=[],
                       shuffle_before_pack=True,
                       pack_to_max_length=True,
                       input_ids_with_output=True):

    dataset = BUILDER.build(dataset)
    if isinstance(dataset, DatasetDict):
        dataset = dataset[split]

    # sample `max_dataset_length` items from the original dataset to
    # save time consumed by map function
    if max_dataset_length is not None:
        max_dataset_length = min(max_dataset_length, len(dataset))
        indices = np.random.choice(
            len(dataset), max_dataset_length, replace=False)
        dataset = dataset.select(indices)

    # Extract the useful data for training from the original dataset.
    if dataset_map_fn is not None:
        dataset = dataset.map(dataset_map_fn)

    # Add prompt template, such as ### Human: xxx ###Assistant: xxx
    if prompt_template is not None:
        assert (hasattr(prompt_template, 'INSTRUCTION_START') and
                hasattr(prompt_template, 'INSTRUCTION')), \
                    ('`prompt_template` can be found in '
                     '`xtuner/utils/templates.py`'
                     'The `prompt_template` should consist of two distinct '
                     'keys: `INSTRUCTION_START` and `INSTRUCTION`. '
                     'The `INSTRUCTION_START` serves as the template for '
                     'initiating multi-turn dialogues, whereas `INSTRUCTION` '
                     'applies to templates used in the following rounds of '
                     'communication.')

        template_map_fn = partial(
            prompt_template_map_fn, template=prompt_template)
        dataset = dataset.map(template_map_fn)

    for old, new in rename_maps:
        dataset = dataset.rename_column(old, new)

    # remove unused columns
    remove_unused_columns = pack_to_max_length or remove_unused_columns

    # tokenize
    if isinstance(tokenizer, dict) or isinstance(
            tokenizer, Config) or isinstance(tokenizer, ConfigDict):
        tokenizer = BUILDER.build(tokenizer)
    dataset = dataset.map(
        partial(
            encode_fn,
            tokenizer=tokenizer,
            max_length=max_length,
            input_ids_with_output=input_ids_with_output),
        remove_columns=list(dataset.column_names)
        if remove_unused_columns else None)

    # pack to max length
    if pack_to_max_length and split == 'train':
        if shuffle_before_pack:
            dataset = dataset.shuffle()
            dataset = dataset.flatten_indices()
        dataset = dataset.map(Packer(max_length), batched=True)

    return dataset
