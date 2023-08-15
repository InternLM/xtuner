# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial

import numpy as np
from datasets import DatasetDict
from mmengine.config import Config, ConfigDict
from mmengine.config.lazy import LazyObject

from xtuner.registry import DATASETS, TOKENIZER
from .utils import Packer, encode_fn


def process_hf_dataset(dataset,
                       tokenizer,
                       max_length,
                       max_dataset_length=None,
                       split='train',
                       map_fn=None,
                       remove_columns=[],
                       rename_maps=[],
                       pack_to_max_length=True,
                       input_with_labels=True):

    dataset = DATASETS.build(dataset)
    if isinstance(dataset, DatasetDict):
        dataset = dataset[split]

    # sample `max_dataset_length` items from the original dataset to
    # save time consumed by map function
    if max_dataset_length is not None:
        max_dataset_length = min(max_dataset_length, len(dataset))
        indices = np.random.choice(
            len(dataset), max_dataset_length, replace=False)
        dataset = dataset.select(indices)

    if isinstance(map_fn, str):
        map_fn = eval(map_fn)
    if isinstance(map_fn, list):
        assert all(
            [callable(fn) and isinstance(fn, LazyObject) for fn in map_fn])
        for fn in map_fn[:-1]:
            fn = fn.build()
            dataset = dataset.map(fn)
        dataset = dataset.map(
            map_fn[-1].build(), remove_columns=remove_columns)
    elif map_fn is not None:
        dataset = dataset.map(map_fn, remove_columns=remove_columns)
    for old, new in rename_maps:
        dataset = dataset.rename_column(old, new)
    if isinstance(tokenizer, dict) or isinstance(
            tokenizer, Config) or isinstance(tokenizer, ConfigDict):
        tokenizer = TOKENIZER.build(tokenizer)
    dataset = dataset.map(
        partial(
            encode_fn,
            tokenizer=tokenizer,
            max_length=max_length,
            input_with_labels=input_with_labels))
    if pack_to_max_length and split == 'train':
        column_names = list(dataset.column_names)
        dataset = dataset.map(
            Packer(max_length),
            batched=True,
            remove_columns=column_names)
    return dataset
