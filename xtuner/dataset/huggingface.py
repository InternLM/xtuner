# Copyright (c) OpenMMLab. All rights reserved.
import logging
from functools import partial

import numpy as np
from datasets import DatasetDict
from mmengine import print_log
from mmengine.config import Config, ConfigDict
from mmengine.utils.misc import get_object_from_string
from torch import distributed as dist

from xtuner.registry import BUILDER, MAP_FUNC
from .utils import Packer, encode_fn


def process(dataset,
            tokenizer,
            max_length,
            dataset_map_fn=None,
            template_map_fn=None,
            prompt_template=None,
            max_dataset_length=None,
            split='train',
            remove_unused_columns=False,
            rename_maps=[],
            shuffle_before_pack=True,
            pack_to_max_length=True,
            input_ids_with_output=True,
            with_image_token=False,
            map_num_proc=32):
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
        with_image_token: Whether to convert DEFAULT_IMAGE_TOKEN to
            IMAGE_TOKEN_INDEX. Typically set it to True during the training
            of VLM.
        map_num_proc: Max number of processes when mapping the dataset.
    """

    assert template_map_fn is None or prompt_template is None

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

    # rename
    for old, new in rename_maps:
        dataset = dataset.rename_column(old, new)

    # Extract the useful data for training from the original dataset.
    if dataset_map_fn is not None:
        if isinstance(dataset_map_fn, str):
            map_fn_obj = MAP_FUNC.get(
                dataset_map_fn) or get_object_from_string(dataset_map_fn)
            if map_fn_obj is not None:
                dataset_map_fn = map_fn_obj
            else:
                raise TypeError('dataset_map_fn must be a function or a '
                                "registered function's string in MAP_FUNC, "
                                f"but got a string of '{dataset_map_fn}'")

        dataset = dataset.map(dataset_map_fn, num_proc=map_num_proc)

    # Add prompt template, such as <|System|>: xxx <|User|>: xxx <|Bot|>: xxx
    if template_map_fn is not None:
        if isinstance(template_map_fn, dict) or isinstance(
                template_map_fn, Config) or isinstance(template_map_fn,
                                                       ConfigDict):
            template_map_fn = BUILDER.build(template_map_fn)
        dataset = dataset.map(template_map_fn, num_proc=map_num_proc)
    elif prompt_template is not None:
        if isinstance(prompt_template, str):  # for resume
            prompt_template = get_object_from_string(prompt_template)
        dataset = dataset.map(
            prompt_template.template_map_fn_v2, num_proc=map_num_proc)

    # remove invalid data
    if 'conversation' in dataset.column_names:
        dataset = dataset.filter(
            lambda example: len(example['conversation']) > 0,
            num_proc=map_num_proc)
    if 'messages' in dataset.column_names:
        dataset = dataset.filter(
            lambda example: len(example['messages']) > 0,
            num_proc=map_num_proc)

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
    if prompt_template is None or ('messages' not in dataset.column_names
                                   and 'conversation' in dataset.column_names):
        encode_map_fn = encode_fn
    else:
        encode_map_fn = prompt_template.encode_map_fn
    dataset = dataset.map(
        partial(
            encode_map_fn,
            tokenizer=tokenizer,
            max_length=max_length,
            with_image_token=with_image_token,
            input_ids_with_output=input_ids_with_output),
        remove_columns=list(dataset.column_names)
        if remove_unused_columns else None,
        num_proc=map_num_proc)

    if input_ids_with_output:
        # remove data that does not have the valid labels.
        dataset = dataset.filter(
            lambda example: any(label >= 0 for label in example['labels']),
            num_proc=map_num_proc)

    # pack to max length
    if pack_to_max_length and split == 'train':
        if shuffle_before_pack:
            dataset = dataset.shuffle()
            dataset = dataset.flatten_indices(num_proc=map_num_proc)
        dataset = dataset.map(
            Packer(max_length), batched=True, num_proc=map_num_proc)

    # add 'length'
    setattr(dataset, 'length', [len(i['input_ids']) for i in dataset])

    return dataset


def process_hf_dataset(*args, **kwargs):
    if not (dist.is_available() and dist.is_initialized()):
        return process(*args, **kwargs)

    if dist.get_rank() == 0:
        dataset = process(*args, **kwargs)
        objects = [dataset]
    else:
        objects = [None]
    dist.broadcast_object_list(objects, src=0)
    return objects[0]
