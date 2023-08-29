# Copyright (c) OpenMMLab. All rights reserved.
import logging
from functools import partial

import numpy as np
from datasets import DatasetDict
from mmengine import print_log
from mmengine.config import Config, ConfigDict

from xtuner.registry import BUILDER
from .utils import Packer, encode_fn


def process_hf_dataset(dataset,
                       tokenizer,
                       max_length,
                       dataset_map_fn=None,
                       template_map_fn=None,
                       max_dataset_length=None,
                       split='train',
                       remove_unused_columns=False,
                       rename_maps=[],
                       shuffle_before_pack=True,
                       pack_to_max_length=True,
                       input_ids_with_output=True):
    """Post-process the dataset loaded from the Hugging Face Hub, or a local
    dataset.

    Args:
        dataset: The dataset to be post-processed.
        tokenizer: The tokenizer processes some raw text as input and outputs
            an Encoding.
        max_length: Max length of the sequence.
        dataset_map_fn: Map the original dataset format to the one defined
            by xTuner.
        template_map_fn: Add the prompt template to the dataset
        max_dataset_length: If the length of the dataset is too long, we can
            randomly extract `max_dataset_length` from it.
        split: Which split of the data to load.
            If `None`, will return a `dict` with all splits (typically
            `datasets.Split.TRAIN` and `datasets.Split.TEST`).
            If given, will return a single Dataset.
        remove_unused_columns: Whether to remove columns from the dataset
            that are not used during training.
        rename_maps: Rename the column name of the dataset.
        shuffle_before_pack: Whether to shuffle the dataset before
            packing them.
        pack_to_max_length: Whether to pack the dataset to the `max_length `.
            This usually improves gpu utilization and therefore reduces
            training time.
        input_ids_with_output: Whether to put the groundtruth output
            corresponding to the question into the dataset. Typically set
            it to True during training and False during testing.
    """

    if isinstance(dataset, DatasetDict):
        dataset = dataset[split]
    elif isinstance(dataset, dict) or isinstance(
            dataset, Config) or isinstance(dataset, ConfigDict):
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
    if template_map_fn is not None:
        if isinstance(template_map_fn, dict) or isinstance(
                template_map_fn, Config) or isinstance(template_map_fn,
                                                       ConfigDict):
            template_map_fn = BUILDER.build(template_map_fn)
        dataset = dataset.map(template_map_fn)

    for old, new in rename_maps:
        dataset = dataset.rename_column(old, new)

    # remove unused columns
    if pack_to_max_length and (not remove_unused_columns):
        print_log(
            'We have to remove unused columns if '
            '`pack_to_max_length` is set to True.',
            logger='current',
            level=logging.WARNING)
        remove_unused_columns = True

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
