# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
from datetime import timedelta
from functools import partial

import numpy as np
from datasets import DatasetDict, concatenate_datasets
from mmengine import print_log
from mmengine.config import Config, ConfigDict
from mmengine.utils.misc import get_object_from_string
from torch import distributed as dist

from xtuner.registry import BUILDER, MAP_FUNC
from .utils import Packer, encode_fn


def get_lengths(example):
    return {'length': len(example['input_ids'])}


def build_origin_dataset(dataset, split):
    if isinstance(dataset, DatasetDict):
        if split is None:
            dataset = concatenate_datasets(dataset.values())
        else:
            dataset = dataset[split]
    elif isinstance(dataset, dict) or isinstance(
            dataset, Config) or isinstance(dataset, ConfigDict):
        dataset = BUILDER.build(dataset)
        if isinstance(dataset, DatasetDict):
            if split is None:
                dataset = concatenate_datasets(dataset.values())
            else:
                dataset = dataset[split]
    return dataset


def map_dataset(dataset, dataset_map_fn, map_num_proc):
    if isinstance(dataset_map_fn, str):
        map_fn_obj = MAP_FUNC.get(dataset_map_fn) or get_object_from_string(
            dataset_map_fn)
        if map_fn_obj is not None:
            dataset_map_fn = map_fn_obj
        else:
            raise TypeError('dataset_map_fn must be a function or a '
                            "registered function's string in MAP_FUNC, "
                            f"but got a string of '{dataset_map_fn}'")

    dataset = dataset.map(dataset_map_fn, num_proc=map_num_proc)
    return dataset


def add_template_to_dataset(dataset, template_map_fn, map_num_proc):
    if isinstance(template_map_fn,
                  dict) or isinstance(template_map_fn, Config) or isinstance(
                      template_map_fn, ConfigDict):
        template_map_fn = BUILDER.build(template_map_fn)
    dataset = dataset.map(template_map_fn, num_proc=map_num_proc)
    # remove invalid data
    dataset = dataset.filter(
        lambda example: len(example['conversation']) > 0,
        num_proc=map_num_proc)
    return dataset


def tokenize_dataset(dataset, tokenizer, max_length, with_image_token,
                     input_ids_with_output, remove_unused_columns,
                     map_num_proc):
    assert (tokenizer is not None) and (max_length is not None), \
        f'({tokenizer}, {max_length})'
    if isinstance(tokenizer, dict) or isinstance(
            tokenizer, Config) or isinstance(tokenizer, ConfigDict):
        tokenizer = BUILDER.build(tokenizer)
    dataset = dataset.map(
        partial(
            encode_fn,
            tokenizer=tokenizer,
            max_length=max_length,
            with_image_token=with_image_token,
            input_ids_with_output=input_ids_with_output),
        remove_columns=list(dataset.column_names)
        if remove_unused_columns else None,
        num_proc=map_num_proc)
    return dataset


def pack_dataset(dataset, max_length, use_varlen_attn, shuffle_before_pack,
                 map_num_proc):
    if shuffle_before_pack:
        dataset = dataset.shuffle()
        dataset = dataset.flatten_indices(num_proc=map_num_proc)
    dataset = dataset.map(
        Packer(max_length, use_varlen_attn=use_varlen_attn),
        batched=True,
        num_proc=map_num_proc)
    return dataset


def process(dataset,
            do_dataset_tokenization=True,
            tokenizer=None,
            max_length=None,
            dataset_map_fn=None,
            template_map_fn=None,
            max_dataset_length=None,
            split='train',
            remove_unused_columns=False,
            rename_maps=[],
            shuffle_before_pack=True,
            pack_to_max_length=True,
            use_varlen_attn=False,
            input_ids_with_output=True,
            with_image_token=False,
            map_num_proc=32):
    """Post-process the dataset loaded from the Hugging Face Hub, or a local
    dataset.

    Args:
        dataset: The dataset to be post-processed.
        do_dataset_tokenization: Whether the dataset need to be tokenized
            in this function. Default to True.
        tokenizer: The tokenizer processes some raw text as input and outputs
            an Encoding. If `do_dataset_tokenization` is True, this argument
            should not be None. Default to None.
        max_length: Max length of the sequence. If `do_dataset_tokenization`
            or `pack_to_max_length` is True, this argument should not be None.
            Default to None.
        dataset_map_fn: Map the original dataset format to the one defined
            by xTuner.
        template_map_fn: Add the prompt template to the dataset
        max_dataset_length: If the length of the dataset is too long, we can
            randomly extract `max_dataset_length` from it.
        split: Which split of the data to load.
            If `None`, will return a single concatenated dataset with all
            splits (typically `datasets.Split.TRAIN` and
            `datasets.Split.TEST`).
            If given, will return a single Dataset.
        remove_unused_columns: Whether to remove columns from the dataset
            that are not used during training.
        rename_maps: Rename the column name of the dataset.
        shuffle_before_pack: Whether to shuffle the dataset before
            packing them.
        pack_to_max_length: Whether to pack the dataset to the `max_length `.
            This usually improves gpu utilization and therefore reduces
            training time.
        use_varlen_attn: If use_varlen_attn is True, we calculate attention
            the actual length of the sequence rather than the actual length
            of the sequence
        input_ids_with_output: Whether to put the groundtruth output
            corresponding to the question into the dataset. Typically set
            it to True during training and False during testing.
        with_image_token: Whether to convert DEFAULT_IMAGE_TOKEN to
            IMAGE_TOKEN_INDEX. Typically set it to True during the training
            of VLM.
        map_num_proc: Max number of processes when mapping the dataset.
    """
    if use_varlen_attn:
        assert pack_to_max_length, \
            '`pack_to_max_length` in `process_hf_dataset` should be set to ' \
            'True if `use_varlen_attn` is True.'
    if pack_to_max_length:
        assert split == 'train' or split is None, \
            ('`split` should be `train` or `None` if `pack_to_max_length` is '
             f'True, but got {split}.')

    dataset = build_origin_dataset(dataset, split)

    # sample `max_dataset_length` items from the original dataset to
    # save time consumed by map function
    if max_dataset_length is not None:
        max_dataset_length = min(max_dataset_length, len(dataset))
        indices = np.random.choice(
            len(dataset), max_dataset_length, replace=False)
        dataset = dataset.select(indices)

    # Extract the useful data for training from the original dataset.
    if dataset_map_fn is not None:
        dataset = map_dataset(dataset, dataset_map_fn, map_num_proc)

    # Add prompt template, such as <|System|>: xxx <|User|>: xxx <|Bot|>: xxx
    if template_map_fn is not None:
        dataset = add_template_to_dataset(dataset, template_map_fn,
                                          map_num_proc)

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

    if do_dataset_tokenization:
        dataset = tokenize_dataset(dataset, tokenizer, max_length,
                                   with_image_token, input_ids_with_output,
                                   remove_unused_columns, map_num_proc)

    if input_ids_with_output:
        assert {'input_ids', 'labels'}.issubset(dataset.column_names)
        # remove data that does not have the valid labels.
        dataset = dataset.filter(
            lambda example: any(label >= 0 for label in example['labels']),
            num_proc=map_num_proc)

    # pack to max length
    if pack_to_max_length:
        dataset = pack_dataset(dataset, max_length, use_varlen_attn,
                               shuffle_before_pack, map_num_proc)

    # add 'length'
    dataset = dataset.map(get_lengths, num_proc=map_num_proc)
    setattr(dataset, 'length', dataset['length'])

    return dataset


def process_hf_dataset(dataset,
                       do_dataset_tokenization=True,
                       tokenizer=None,
                       max_length=None,
                       dataset_map_fn=None,
                       template_map_fn=None,
                       max_dataset_length=None,
                       split='train',
                       remove_unused_columns=False,
                       rename_maps=[],
                       shuffle_before_pack=True,
                       pack_to_max_length=True,
                       use_varlen_attn=False,
                       input_ids_with_output=True,
                       with_image_token=False,
                       map_num_proc=32):
    """Post-process the dataset loaded from the Hugging Face Hub, or a local
    dataset.

    Args:
        dataset: The dataset to be post-processed.
        do_dataset_tokenization: Whether the dataset need to be tokenized
            in this function. Default to True.
        tokenizer: The tokenizer processes some raw text as input and outputs
            an Encoding. If `do_dataset_tokenization` is True, this argument
            should not be None. Default to None.
        max_length: Max length of the sequence. If `do_dataset_tokenization`
            or `pack_to_max_length` is True, this argument should not be None.
            Default to None.
        dataset_map_fn: Map the original dataset format to the one defined
            by xTuner.
        template_map_fn: Add the prompt template to the dataset
        max_dataset_length: If the length of the dataset is too long, we can
            randomly extract `max_dataset_length` from it.
        split: Which split of the data to load.
            If `None`, will return a single concatenated dataset with all
            splits (typically `datasets.Split.TRAIN` and
            `datasets.Split.TEST`).
            If given, will return a single Dataset.
        remove_unused_columns: Whether to remove columns from the dataset
            that are not used during training.
        rename_maps: Rename the column name of the dataset.
        shuffle_before_pack: Whether to shuffle the dataset before
            packing them.
        pack_to_max_length: Whether to pack the dataset to the `max_length `.
            This usually improves gpu utilization and therefore reduces
            training time.
        use_varlen_attn: If use_varlen_attn is True, we calculate attention
            the actual length of the sequence rather than the actual length
            of the sequence
        input_ids_with_output: Whether to put the groundtruth output
            corresponding to the question into the dataset. Typically set
            it to True during training and False during testing.
        with_image_token: Whether to convert DEFAULT_IMAGE_TOKEN to
            IMAGE_TOKEN_INDEX. Typically set it to True during the training
            of VLM.
        map_num_proc: Max number of processes when mapping the dataset.
    """
    kwargs = dict(
        dataset=dataset,
        do_dataset_tokenization=do_dataset_tokenization,
        tokenizer=tokenizer,
        max_length=max_length,
        dataset_map_fn=dataset_map_fn,
        template_map_fn=template_map_fn,
        max_dataset_length=max_dataset_length,
        split=split,
        remove_unused_columns=remove_unused_columns,
        rename_maps=rename_maps,
        shuffle_before_pack=shuffle_before_pack,
        pack_to_max_length=pack_to_max_length,
        use_varlen_attn=use_varlen_attn,
        input_ids_with_output=input_ids_with_output,
        with_image_token=with_image_token,
        map_num_proc=map_num_proc)
    if not (dist.is_available() and dist.is_initialized()):
        return process(**kwargs)

    xtuner_dataset_timeout = timedelta(
        minutes=int(os.getenv('XTUNER_DATASET_TIMEOUT', default=60)))
    print_log(
        f'xtuner_dataset_timeout = {xtuner_dataset_timeout}', logger='current')
    # monitored barrier requires gloo process group to perform host-side sync.
    group_gloo = dist.new_group(backend='gloo', timeout=xtuner_dataset_timeout)

    if dist.get_rank() == 0:
        dataset = process(**kwargs)
        objects = [dataset]
    else:
        objects = [None]

    dist.monitored_barrier(group=group_gloo, timeout=xtuner_dataset_timeout)
    dist.broadcast_object_list(objects, src=0)
    return objects[0]
