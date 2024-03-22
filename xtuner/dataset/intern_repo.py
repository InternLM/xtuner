# Copyright (c) OpenMMLab. All rights reserved.
import itertools as it
import json
import mmap
import operator
import os
import threading
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, load_dataset, load_from_disk
from mmengine import print_log
from torch import distributed as dist
from torch.utils.data import ConcatDataset

from xtuner.dataset.map_fns import openai_map_fn
from xtuner.registry import BUILDER
from .huggingface import process


class JsonlDataset(torch.utils.data.Dataset):
    """

    JSONL format is expected to roughly follow that of The Pile.
    One-line-per-document of the form:
    ```
    {
        "input_ids": List[int],
        "labels": List[int]
    }
    ```

    """

    def __init__(self, path: str, min_length=50):
        self.path = path
        self.threadlocal = threading.local()
        resolved_path = Path(path).resolve()
        self.resolved_path = resolved_path
        self.meta = Path(f'{resolved_path}.meta')

        # only build the cache in on the primary worker to prevent
        # overloading nfs
        assert os.path.exists(
            self.meta
        ), f'The cache file:{self.meta} is not found for file:{self.path}'
        try:
            with open(self.meta, 'rb') as f:
                meta = np.load(f)
        except Exception as e:
            print(f'Cannot load file {self.meta}...')
            raise e
        self.offsets = meta[:, 0]
        self.length = meta[:, -1]

        if min_length > 0:
            mask = self.length >= min_length
            self.offsets = self.offsets[mask]
            self.length = self.length[mask]

    def __getitem__(self, idx):
        f = self._get_mmap()
        position = self.offsets[idx]
        f.seek(position)
        item = f.readline().decode('utf-8')
        try:
            item = json.loads(item)
            item['input_ids'] = item['tokens']
            del item['tokens']
            labels = [x if x > 0 else -100 for x in item['input_ids']]
            item['input_ids'] = [abs(x) for x in item['input_ids']]
            item['labels'] = labels
            item['length'] = len(item['input_ids'])  # add a length info
        except Exception as err:
            raise json.decoder.JSONDecodeError(
                doc=self.path,
                pos=position,
                msg=(f'Error while loading JSONL line in file {self.path} '
                     f'at byte {position}. Contents of line:\n{item}\n{err}'),
            )
        return item

    def get_dataset_name(self):
        return str(self.resolved_path)

    def _get_mmap(self):
        if not hasattr(self.threadlocal, 'handles'):
            with open(self.path, 'rb') as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                self.threadlocal.handles = [f, mm]
                if self.path.endswith('.gz') or self.path.endswith(
                        '.bz') or self.path.endswith('.bz2'):
                    raise NotImplementedError(
                        'Compressed files are not supported because .seek() '
                        'would require rereading the entire file, making '
                        'performance too slow.')
        return self.threadlocal.handles[-1]

    def __setstate__(self, state):
        self.__dict__ = state
        self.threadlocal = threading.local()

    def __getstate__(self):
        d = {}
        for i, v in self.__dict__.items():
            if i != 'threadlocal':
                d[i] = v
        return d

    def __del__(self):
        if hasattr(self.threadlocal, 'handles'):
            # cleanup files we opened on initialization
            while self.threadlocal.handles:
                self.threadlocal.handles.pop().close()

    @staticmethod
    def exists(path):
        return os.path.exists(path)

    def __len__(self):
        # Virtual length of the dataset depends on the epoch number
        # if the number of documents is not perfectly divisible by the
        # data_subshard_count
        return len(self.offsets)


class PackedDataset(torch.utils.data.Dataset):
    """The class PackedDataset takes in a dataset and aggregates samples of
    different lengths together based on the packed_length.

    Args:
        dataset: The original dataset to pack.
        packed_length: The length of each packed sample. Default is 8192.
    """

    def __init__(self, dataset, packed_length: int = 8192, seed: int = 1024):
        self.dataset = dataset
        self.packed_length = packed_length
        if isinstance(dataset, JsonlDataset):
            self.length = dataset.length
        elif isinstance(dataset, Dataset):
            if hasattr(dataset, 'length'):
                length = dataset.length
            else:
                length = [len(i['input_ids']) for i in dataset]
            self.length = length
        else:
            raise NotImplementedError
        self.seed = seed

        rng = np.random.RandomState(self.seed)
        shuffled_indices = np.arange(len(self.length))
        rng.shuffle(shuffled_indices)
        self.shuffled_indices = shuffled_indices.tolist()
        self.shuffled_samples_len = list(
            map(self.length.__getitem__, shuffled_indices))
        self.shuffled_accumulated_samples_len = list(
            it.accumulate(self.shuffled_samples_len, operator.add))
        self.num_tokens = sum(self.length)

    def __len__(self):
        return self.num_tokens // self.packed_length

    def search_sample_index(self, pack_idx: int = 0):
        assert pack_idx >= 0
        length_train = (pack_idx + 1) * self.packed_length
        sample_index = np.searchsorted(
            self.shuffled_accumulated_samples_len, length_train, side='left')
        return sample_index

    def mapping(self, pack_idx: int = 0):
        begin_sample_idx, begin_token_id = 0, 0
        if pack_idx > 0:
            begin_sample_idx = self.search_sample_index(pack_idx - 1)
            # The position where the previous packed data ends
            begin_token_id = self.shuffled_samples_len[begin_sample_idx] - (
                self.shuffled_accumulated_samples_len[begin_sample_idx]
                -  # noqa: W504,W503
                (pack_idx) * self.packed_length)
            if begin_token_id == self.shuffled_samples_len[begin_sample_idx]:
                begin_sample_idx += 1
                begin_token_id = 0

        end_sample_idx = self.search_sample_index(pack_idx)
        end_token_id = self.shuffled_samples_len[end_sample_idx] - (
            self.shuffled_accumulated_samples_len[end_sample_idx]
            -  # noqa: W504,W503
            (pack_idx + 1) * self.packed_length)
        return begin_sample_idx, begin_token_id, end_sample_idx, end_token_id

    def build_pack(self, begin_sample_idx: int, begin_token_id: int,
                   end_sample_idx: int, end_token_id: int):
        pack, cumulative_len, position_ids, labels = [], [0], [], []

        while begin_sample_idx < end_sample_idx:
            sample_idx = self.shuffled_indices[begin_sample_idx]
            sample = self.dataset[sample_idx]
            chunk = sample['input_ids'][begin_token_id:]
            pack.extend(chunk)
            _labels = sample['labels'][begin_token_id:]
            assert len(_labels) == len(chunk), (_labels, chunk)
            labels.extend(_labels)
            cumulative_len.append(cumulative_len[-1] + len(chunk))
            position_ids.extend(list(range(len(chunk))))
            begin_sample_idx = begin_sample_idx + 1
            begin_token_id = 0

        sample_idx = self.shuffled_indices[end_sample_idx]
        sample = self.dataset[sample_idx]
        chunk = sample['input_ids'][begin_token_id:
                                    end_token_id]  # fragment of a sample
        _labels = sample['labels'][begin_token_id:end_token_id]
        pack.extend(chunk)
        assert len(_labels) == len(chunk), (_labels, chunk)
        labels.extend(_labels)
        cumulative_len.append(cumulative_len[-1] + len(chunk))
        position_ids.extend(list(range(len(chunk))))

        out = {
            'input_ids': pack,
            'cumulative_len': cumulative_len,
            'position_ids': position_ids,
            'labels': labels
        }
        return out

    def __getitem__(self, item: int):
        pos_before, token_id_before, pos_after, token_id_after = self.mapping(
            item)
        return self.build_pack(pos_before, token_id_before, pos_after,
                               token_id_after)


def load_intern_repo_tokenized_dataset(folder,
                                       min_length=0,
                                       data_order_path=None,
                                       file_type='.bin'):
    assert os.path.exists(folder), f'{folder} does not exist.'
    datasets = []

    if data_order_path is not None:
        data_order = load_dataset(
            'text', data_files=data_order_path, split='train')['text']
        for i, fp in enumerate(data_order):
            data_order[i] = os.path.join(folder, fp)
    else:
        triples = list(os.walk(folder, followlinks=True))
        data_order = []
        for root, dirs, files in triples:
            dirs.sort()
            for fn in sorted(files):
                if fn.endswith(file_type):
                    fp = os.path.join(root, fn)
                    data_order.append(fp)

    for fp in data_order:
        print_log(f'Reading {fp}...', logger='current')
        ds = JsonlDataset(fp, min_length=min_length)

        if len(ds) == 0:
            continue
        datasets.append(ds)

    return datasets


def load_intern_repo_untokenized_dataset(processed_dataset_dict_path=None,
                                         folder=None,
                                         tokenizer=None,
                                         max_length=None,
                                         template_map_fn=None,
                                         data_order_path=None,
                                         file_type='.json'):

    assert processed_dataset_dict_path or (folder and tokenizer and max_length)

    if processed_dataset_dict_path is not None:
        ds = load_from_disk(processed_dataset_dict_path)
        datasets = []
        for key, data in ds.items():
            datasets.append((key, data))
        datasets = sorted(datasets, key=lambda x: int(x[0]))
        datasets = [x[1] for x in datasets]
        return datasets

    assert os.path.exists(folder), f'{folder} does not exist.'
    datasets = []

    if data_order_path is not None:
        data_order = load_dataset(
            'text', data_files=data_order_path, split='train')['text']
        for i, fp in enumerate(data_order):
            data_order[i] = os.path.join(folder, fp)
    else:
        triples = list(os.walk(folder, followlinks=True))
        data_order = []
        for root, dirs, files in triples:
            dirs.sort()
            for fn in sorted(files):
                if fn.endswith(file_type):
                    fp = os.path.join(root, fn)
                    data_order.append(fp)

    for fp in data_order:
        print_log(f'Reading {fp}...', logger='current')
        dataset = []
        with open(fp) as file:
            lines = file.readlines()
            for line in lines:
                line = json.loads(line)
                dataset.append({'messages': line})
        dataset = Dataset.from_list(dataset)
        dataset = process(
            dataset,
            tokenizer=tokenizer,
            max_length=max_length,
            dataset_map_fn=openai_map_fn,
            template_map_fn=template_map_fn,
            remove_unused_columns=True,
            pack_to_max_length=False,
            map_num_proc=32)

        if len(dataset) == 0:
            continue

        datasets.append(dataset)

    return datasets


def build_packed_dataset_rank0(dataset_cfg, packed_length=8192, seed=1024):
    if isinstance(dataset_cfg, dict):
        datasets = BUILDER.build(dataset_cfg)
    else:
        datasets = dataset_cfg

    if not isinstance(datasets, list):
        datasets = [datasets]

    packed_datasets = []

    for dataset in datasets:
        ds = PackedDataset(dataset, packed_length, seed=seed)
        packed_datasets.append(ds)

    dataset = ConcatDataset(datasets=packed_datasets)

    return dataset


def build_packed_dataset(*args, **kwargs):
    if not (dist.is_available() and dist.is_initialized()):
        return build_packed_dataset_rank0(*args, **kwargs)

    if dist.get_rank() == 0:
        dataset = build_packed_dataset_rank0(*args, **kwargs)
        objects = [dataset]
    else:
        objects = [None]
    dist.broadcast_object_list(objects, src=0)
    return objects[0]
