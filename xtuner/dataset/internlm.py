# Copyright (c) OpenMMLab. All rights reserved.
import os
import shutil
import tempfile

from datasets import concatenate_datasets, load_dataset, load_from_disk
from mmengine import print_log
from torch import distributed as dist
from tqdm import tqdm

from .utils import InternLMPacker


def process(dataset_folder,
            max_length=2048,
            split='train',
            shuffle_before_pack=True,
            pack_to_max_length=False,
            map_num_proc=32,
            tmpdir=None):
    if tmpdir is not None:
        try:
            return load_from_disk(tmpdir)
        except FileNotFoundError:
            pass

    assert dataset_folder is not None
    ds = []
    for root, dirs, files in os.walk(dataset_folder, followlinks=True):
        for fn in tqdm(sorted(files), total=len(files), leave=False):
            if fn.endswith('.bin'):
                fp = os.path.join(root, fn)
                ds.append(load_dataset('json', data_files=fp)[split])
    dataset = concatenate_datasets(ds)
    print_log(f'Find {len(dataset)} samples.', 'current')
    dataset = dataset.rename_column('tokens', 'input_ids')

    # pack to max length
    if pack_to_max_length:
        if shuffle_before_pack:
            dataset = dataset.shuffle()
            dataset = dataset.flatten_indices()
        dataset = dataset.map(
            InternLMPacker(max_length), batched=True, num_proc=map_num_proc)
        print_log(
            f'After packing to {max_length}, '
            f'the length of dataset is {len(dataset)}.', 'current')

    if tmpdir is not None:
        dataset.save_to_disk(tmpdir)

    return dataset


def process_internlm_dataset(*args, **kwargs):
    if not (dist.is_available() and dist.is_initialized()):
        return process(*args, **kwargs)

    if dist.get_rank() == 0:
        tmpdir = tempfile.TemporaryDirectory(dir='./')
        dataset = process(*args, tmpdir=tmpdir.name, **kwargs)
        objects = [tmpdir.name]
    else:
        objects = [None]
    dist.broadcast_object_list(objects, src=0)

    if dist.get_rank() != 0:
        # load processed dataset from `cached_folder`
        dataset = process(*args, tmpdir=objects[0], **kwargs)

    dist.barrier()
    if dist.get_rank() == 0:
        shutil.rmtree(tmpdir.name)

    return dataset
