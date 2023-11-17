# Copyright (c) OpenMMLab. All rights reserved.
import logging
from functools import partial

import numpy as np
from datasets import DatasetDict
from mmengine import print_log
from mmengine.config import Config, ConfigDict
from torch import distributed as dist

from xtuner.registry import BUILDER, MAP_FUNC
import os

from .utils import Packer, encode_fn
from tqdm import tqdm
from datasets import concatenate_datasets, load_dataset, load_from_disk
from xtuner.utils import IGNORE_INDEX
import copy
from mmengine import print_log


def add_labels(example, max_length):
    tokens = example['tokens'][:max_length]
    labels = copy.deepcopy(tokens)
    tokens = list(np.abs(np.array(tokens)))
    labels = np.array(labels)
    labels[labels < 0] = IGNORE_INDEX
    labels = list(labels)
    return {'input_ids': tokens, 'labels': labels}


def process(dataset_folder=None,
            cached_folder=None,
            max_length=2048,
            split='train',
            shuffle_before_pack=True,
            pack_to_max_length=False,
            num_proc=32):
    if cached_folder is not None:
        try:
            return load_from_disk(cached_folder)
        except FileNotFoundError:
            pass
    
    assert dataset_folder is not None
    ds = []
    for root, dirs, files in os.walk(dataset_folder, followlinks=True):
        for fn in tqdm(sorted(files), total=len(files), leave=False):
            if fn.endswith(".bin"):
                fp = os.path.join(root, fn)
                ds.append(load_dataset('json', data_files=fp)[split])
    dataset = concatenate_datasets(ds)
    print_log(f'Find {len(dataset)} samples.', 'current')
    dataset = dataset.map(partial(add_labels, max_length=max_length), remove_columns=list(dataset.column_names), num_proc=num_proc)

    # pack to max length
    if pack_to_max_length:
        if shuffle_before_pack:
            dataset = dataset.shuffle()
            dataset = dataset.flatten_indices()
        dataset = dataset.map(Packer(max_length), batched=True, num_proc=num_proc)
        print_log(f'After packing to {max_length}, '
                  f'the length of dataset is {len(dataset)}.', 'current')
    
    dataset.save_to_disk(cached_folder)
    print_log(f'Processed dataset has been saved in {cached_folder}.', 'current')

    return dataset


def process_tokenized_dataset(*args, **kwargs):
    if not (dist.is_available() and dist.is_initialized()):
        return process(*args, **kwargs)

    if dist.get_rank() == 0:
        dataset = process(*args, **kwargs)
    
    dist.barrier()
    if dist.get_rank() != 0:
        # load processed dataset from `cached_folder`
        dataset = process(*args, **kwargs)
    return dataset
