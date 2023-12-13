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

    # ds = []
    # for root, dirs, files in os.walk(dataset_folder, followlinks=True):
    #     for fn in tqdm(sorted(files), total=len(files), leave=False):
    #         if fn.endswith('.bin') and fn == 'calculate_format_datum_max_num.bin':
    #             fp = os.path.join(root, fn)
    #             ds.append(load_dataset('json', data_files=fp)[split])

    # dataset = concatenate_datasets(ds)
    # dataset = load_dataset('json', data_files='/mnt/petrelfs/share_data/caoweihan/maibao_kaoshi_7_5_v0213_8k_rc8_nowm/train/cn/calculate_format_datum_max_num.bin')[split]
    dataset = load_dataset('json', data_files='/mnt/petrelfs/share_data/caoweihan/maibao_kaoshi_7_5_v0213_8k_rc8_nowm/train/cn/long_seq_data.bin')[split]
    print_log(f'Find {len(dataset)} samples.', 'current')
    dataset = dataset.rename_column('tokens', 'input_ids')

    # pack to max length
    if pack_to_max_length:
        if shuffle_before_pack:
            dataset = dataset.shuffle()
            dataset = dataset.flatten_indices(num_proc=map_num_proc)
        dataset = dataset.map(
            InternRepoPacker(max_length), batched=True, num_proc=map_num_proc, load_from_cache_file=False)
        print_log(
            f'After packing to {max_length}, '
            f'the length of dataset is {len(dataset)}.', 'current')

    return dataset


def process_intern_repo_dataset_debug(*args, **kwargs):
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
