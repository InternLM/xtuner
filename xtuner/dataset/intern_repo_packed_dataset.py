# Copyright (c) OpenMMLab. All rights reserved.
import os

from mmengine import print_log
from torch import distributed as dist
from tqdm import tqdm
import threading
import json
import mmap
import torch
from pathlib import Path
import numpy as np
import itertools as it
import operator
from copy import deepcopy
from torch.utils.data import ConcatDataset
from mmengine import print_log
from datasets import Dataset


class JsonlDataset(torch.utils.data.Dataset):
    """

    JSONL format is expected to roughly follow that of The Pile.
    One-line-per-document of the form:
    ```
    {
        "tokens": List[int],
    }
    ```

    Note that only the "tokens" key is used.
    """

    def __init__(self, path: str, min_length=50):
        self.path = path
        self.threadlocal = threading.local()
        resolved_path = Path(path).resolve()
        self.resolved_path = resolved_path
        self.meta = Path(f"{resolved_path}.meta")

        # only build the cache in on the primary worker to prevent overloading nfs
        assert os.path.exists(self.meta), f"The cache file:{self.meta} is not found for file:{self.path}"
        try:
            with open(self.meta, "rb") as f:
                meta = np.load(f)
        except Exception as e:
            print(f"Cannot load file {self.meta}...")
            raise e
        self.offsets = meta[:, 0]
        self.lengths = meta[:, -1]

        if min_length > 0:
            mask = self.lengths >= min_length
            self.old_lengths = self.lengths.copy()
            self.old_length = len(self.offsets)
            self.offsets = self.offsets[mask]
            self.lengths = self.lengths[mask]

    def __getitem__(self, idx):
        f = self._get_mmap()
        position = self.offsets[idx]
        f.seek(position)
        item = f.readline().decode("utf-8")
        try:
            item = json.loads(item)
            item["length"] = len(item["tokens"])  # add a length info
        except Exception as err:
            raise json.decoder.JSONDecodeError(
                doc=self.path,
                pos=position,
                msg=(
                    f"Error while loading JSONL line in file {self.path} at byte "
                    f"{position}. Contents of line:\n{item}\n{err}"
                ),
            )
        return item

    def get_dataset_name(self):
        return str(self.resolved_path)

    def _get_mmap(self):
        if not hasattr(self.threadlocal, "handles"):
            with open(self.path, "rb") as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                self.threadlocal.handles = [f, mm]
                if self.path.endswith(".gz") or self.path.endswith(".bz") or self.path.endswith(".bz2"):
                    raise NotImplementedError(
                        "Compressed files are not supported because .seek() would require "
                        "rereading the entire file, making performance too slow."
                    )
        return self.threadlocal.handles[-1]

    def __setstate__(self, state):
        self.__dict__ = state
        self.threadlocal = threading.local()

    def __getstate__(self):
        d = {}
        for i, v in self.__dict__.items():
            if i != "threadlocal":
                d[i] = v
        return d

    def __del__(self):
        if hasattr(self.threadlocal, "handles"):
            # cleanup files we opened on initialization
            while self.threadlocal.handles:
                self.threadlocal.handles.pop().close()

    @staticmethod
    def exists(path):
        return os.path.exists(path)

    def __len__(self):
        # Virtual length of the dataset depends on the epoch number if the number of documents
        # is not perfectly divisible by the data_subshard_count
        return len(self.offsets)


class PackedDataset(torch.utils.data.Dataset):
    """
    The class PackedDataset takes in a dataset and aggregates samples of different
    lengths together based on the packed_length.

    Args:
        dataset: The original dataset to pack.
        packed_length: The length of each packed sample. Default is 8192.
    """

    def __init__(
        self,
        dataset,
        packed_length: int = 8192,
        seed: int = 1024
    ):
        self.dataset = dataset
        self.packed_length = packed_length
        if isinstance(dataset, JsonlDataset):
            self.lengths = dataset.lengths
        elif isinstance(dataset, Dataset):
            assert 'lengths' in dataset.column_names
            self.lengths = dataset['lengths']
        else:
            raise NotImplementedError
        self.seed = seed

        rng = np.random.RandomState(self.seed)
        shuffled_indices = np.arange(len(self.lengths))
        rng.shuffle(shuffled_indices)
        self.shuffled_indices = shuffled_indices
        self.shuffled_samples_len = list(map(self.lengths.__getitem__, shuffled_indices))
        self.shuffled_accumulated_samples_len = list(it.accumulate(self.shuffled_samples_len, operator.add))
        self.num_tokens = sum(self.lengths)
    
    def __len__(self):
        return self.num_tokens // self.packed_length
    
    def search_sample_index(self, pack_idx: int = 0):
        assert pack_idx >= 0
        length_train = (pack_idx + 1) * self.packed_length
        sample_index = np.searchsorted(self.shuffled_accumulated_samples_len, length_train, side="left")
        return sample_index
    
    def mapping(self, pack_idx: int = 0):
        begin_sample_idx, begin_token_id = 0, 0
        if pack_idx > 0:
            begin_sample_idx = self.search_sample_index(pack_idx - 1)
            begin_token_id = self.shuffled_samples_len[begin_sample_idx] - (
                self.shuffled_accumulated_samples_len[begin_sample_idx] - (pack_idx) * self.packed_length
            )  # 前一条packed数据结束的位置是那条数据的第几个token
            if begin_token_id == self.shuffled_samples_len[begin_sample_idx]:
                begin_sample_idx += 1
                begin_token_id = 0
        
        end_sample_idx = self.search_sample_index(pack_idx)
        end_token_id = self.shuffled_samples_len[end_sample_idx] - (self.shuffled_accumulated_samples_len[end_sample_idx] - (pack_idx + 1) * self.packed_length)
        return begin_sample_idx, begin_token_id, end_sample_idx, end_token_id
    
    def build_pack(self, begin_sample_idx: int, begin_token_id: int, end_sample_idx: int, end_token_id: int):
        pack, cumulative_len, indexes, labels = [], [0], [], []

        while begin_sample_idx < end_sample_idx:
            sample_idx = self.shuffled_indices[begin_sample_idx]
            sample = self.dataset[sample_idx]
            chunk = sample["tokens"][begin_token_id:]
            pack.extend(chunk)
            _labels = deepcopy(chunk)
            assert len(_labels) == len(chunk), (_labels, chunk)
            labels.extend(_labels)
            cumulative_len.append(cumulative_len[-1] + len(chunk))
            indexes.extend(list(range(len(chunk))))
            begin_sample_idx = begin_sample_idx + 1
            begin_token_id = 0

        sample_idx = self.shuffled_indices[end_sample_idx]
        sample = self.dataset[sample_idx]
        chunk = sample["tokens"][begin_token_id:end_token_id]  # fragement of a sample
        pack.extend(chunk)
        _labels = deepcopy(chunk)
        assert len(_labels) == len(chunk), (_labels, chunk)
        labels.extend(_labels)
        cumulative_len.append(cumulative_len[-1] + len(chunk))
        indexes.extend(list(range(len(chunk))))

        out = {"input_ids": pack, "cumulative_len": cumulative_len, "indexes": indexes, "labels": labels}
        return out
    
    def __getitem__(self, item: int):
        pos_before, token_id_before, pos_after, token_id_after = self.mapping(item)
        return self.build_pack(pos_before, token_id_before, pos_after, token_id_after)


def build_packed_dataset(folder, packed_length=8192, min_length=0):

    assert os.path.exists(folder), f"{folder} does not exist."
    datasets = []
    if dist.get_rank() == 0:
        triples = [list(os.walk(folder, followlinks=True))]
    else:
        triples = [None]
    dist.broadcast_object_list(triples, src=0)
    triples = triples[0]

    for root, dirs, files in triples:
        dirs.sort()  # Let the folder need to be returned in a fixed order
        if dist.get_rank() == 0:
            print_log(f'Reading {root}...', logger='current')
        num_token_in_folder = 0

        for fn in tqdm(sorted(files), total=len(files), leave=False, disable=dist.get_rank() != 0):
            if fn.endswith(".bin"):
                fp = os.path.join(root, fn)
                ds = JsonlDataset(fp, min_length=min_length)

                if len(ds) == 0:
                    continue

                ds = PackedDataset(ds, packed_length)

                num_token_in_folder += len(ds) * packed_length
                datasets.append(ds)

    dataset = ConcatDataset(datasets=datasets)

    return dataset


from torch.utils.data import Sampler
from typing import Iterator, Optional, Sized
from mmengine.dist import get_dist_info, sync_random_seed
class DefaultSampler(Sampler):
    def __init__(self,
                 dataset: Sized,
                 shuffle: bool = True,
                 seed: Optional[int] = None) -> None:
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        self.shuffle = shuffle
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.epoch = 0

        self.num_samples = len(self.dataset) // world_size
        self.total_size = self.num_samples * world_size

    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""
        # deterministically shuffle based on epoch and seed
        if self.shuffle:
            rng = np.random.RandomState(self.seed + self.epoch)
            indices = np.arange(len(self.dataset))
            rng.shuffle(indices)
            indices = indices.tolist()
        else:
            indices = np.arange(len(self.dataset)).tolist()

        self.indices = indices[:self.total_size]

        # subsample
        indices = indices[self.rank:self.total_size:self.world_size]
        self.subsample_indices = indices

        return iter(indices)

    def __len__(self) -> int:
        """The number of samples in this rank."""
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas use a different
        random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


def packed_collate_fn(batch, packed_length, accumulative_counts):

    xs, ys, cumulative_len, indexes = [], [], [], []
    for b in batch:
        assert (
            len(b["input_ids"]) == packed_length
        ), f"length of a sample should be equal to packed_length, but got {len(b['input_ids'])} and {packed_length})"
        assert (
            len(b["labels"]) == packed_length
        ), f"length of a sample should be equal to packed_length, but got {len(b['labels'])} and {packed_length})"

        input_ids = [abs(w) for w in b["input_ids"]]
        labels = [w if w > 0 else -100 for w in b["labels"]]

        xs.append(torch.LongTensor(input_ids))
        ys.append(torch.LongTensor(labels))
        cumulative_len.append(torch.IntTensor(b["cumulative_len"]))
        indexes.append(torch.LongTensor(b["indexes"]))

    xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
    ys = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=-100)
    indexes = torch.stack(indexes, dim=0)

    assert xs.shape[1] == packed_length, (xs.shape[1], packed_length)
    assert len(batch) == accumulative_counts
    max_seqlen = [(cumulative_len[i][1:] - cumulative_len[i][:-1]).max().item() for i in range(accumulative_counts)]
    data_dict = {"input_ids": xs, "cumulative_len": cumulative_len, "indexes": indexes, "labels": ys, "max_seqlen": max_seqlen}

    return {'data': data_dict, 'data_samples': None}
