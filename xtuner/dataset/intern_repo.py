# Copyright (c) OpenMMLab. All rights reserved.
import os

from datasets import concatenate_datasets, load_dataset, load_from_disk
from mmengine import print_log
from torch import distributed as dist
from tqdm import tqdm

from .utils import InternRepoPacker


def process(dataset_folder,
            max_length=2048,
            split='train',
            shuffle_before_pack=True,
            pack_to_max_length=False,
            map_num_proc=32):
    # map_num_proc = 1
    # return load_from_disk('/mnt/petrelfs/share_data/gaojianfei/wenwei_dataset_fix_labels')

    ds = []
    total_length = 0
    for root, dirs, files in os.walk(dataset_folder, followlinks=True):
        for fn in tqdm(sorted(files), total=len(files), leave=False):
            if fn.endswith('.bin'):
                fp = os.path.join(root, fn)
                data = load_dataset('json', data_files=fp)[split]
                data = data.rename_column('tokens', 'input_ids')
                ds.append(data)
                total_length += len(data)
    print_log(f'Find {total_length} samples.', 'current')
    packed_ds = []
    for data in ds:
        if shuffle_before_pack:
            data = data.shuffle()
            data = data.flatten_indices(num_proc=map_num_proc)
        data = data.map(
            InternRepoPacker(max_length), batched=True, num_proc=1, load_from_cache_file=False)
        packed_ds.append(data)
    del ds

    dataset = concatenate_datasets(packed_ds)
    print_log(
        f'After packing to {max_length}, '
        f'the length of dataset is {len(dataset)}.', 'current')

    dataset.save_to_disk('wenwei_dataset_pack_inside')
    return dataset


def process_intern_repo_dataset(*args, **kwargs):
    """Post-process the dataset in InternLM repo
    (https://github.com/InternLM/InternLM) format.

    The training dataset of InternLM is pre-tokenized, and is formatted as
    follows:

    ```
    {"tokens": [1, -333, -352, -1621, ..., 103028, 13, 2]}
    {"tokens": [1, -333, -352, -1621, ..., 103028, 13, 2]}
    ```

    Among them, tokens with negative values are not involved in the calculation
    of loss during the training process.

    Note:
        This function is specifically designed for processing data in the
        internlm format. However, it should not be misconstrued as a tool for
        training the internlm model.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return process(*args, **kwargs)

    if dist.get_rank() == 0:
        dataset = process(*args, **kwargs)
        objects = [dataset]
    else:
        objects = [None]
    dist.broadcast_object_list(objects, src=0)
    return objects[0]
