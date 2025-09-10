# Copyright (c) OpenMMLab. All rights reserved.
import bisect
import itertools
import multiprocessing
import random
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import shared_memory

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from xtuner.v1.utils import get_logger

from .jsonl import JsonlDataset


logger = get_logger()


class _LegacySoftPackDataset(torch.utils.data.Dataset):
    def __init__(self, datasets, pack_max_length=2048, global_pack=False, seed: int | None = None):
        self.random = random.Random()
        if seed is not None:
            self.random = random.Random(seed)

        if global_pack:
            num_tokens = [np.concatenate([dset.num_tokens for dset in datasets])]
            datasets = [ConcatDataset(datasets)]
        else:
            num_tokens = [dset.num_tokens for dset in datasets]

        self.datasets = datasets
        self.seed = seed
        self.global_pack = global_pack
        self.pack_max_length = pack_max_length

        pack_infos = []
        for i, dataset in enumerate(self.datasets):
            _infos = self.get_pack_infos(dataset, i, num_tokens[i])
            pack_infos.append(_infos)
        self.pack_infos = concatenate_datasets(pack_infos)

    @property
    def longest(self):
        return self.pack_infos["longest"]

    def get_pack_infos(self, dataset, dataset_id, num_tokens):
        inds = list(range(len(dataset)))
        self.random.shuffle(inds)

        item_buffer = []
        length_buffer = []
        longest = 0

        pack_infos = []
        for shfl_i in inds:
            if num_tokens[shfl_i] + sum(length_buffer) <= self.pack_max_length:
                item_buffer.append(shfl_i)
                length_buffer.append(num_tokens[shfl_i])
                longest = max(longest, num_tokens[shfl_i])
            else:
                if len(item_buffer) > 0:
                    info = {
                        "dataset_id": dataset_id,
                        "indices": item_buffer,
                        "longest": int(longest),
                    }
                    pack_infos.append(info)

                item_buffer = [shfl_i]
                length_buffer = [num_tokens[shfl_i]]
                longest = num_tokens[shfl_i]

        if len(item_buffer) > 0:
            info = {
                "dataset_id": dataset_id,
                "indices": item_buffer,
                "longest": int(longest),
            }

            pack_infos.append(info)

        pack_infos = Dataset.from_list(pack_infos)

        return pack_infos

    def __len__(self):
        return len(self.pack_infos)

    def __getitem__(self, item):
        indices = self.pack_infos[item]["indices"]
        dataset_id = self.pack_infos[item]["dataset_id"]
        return [self.datasets[dataset_id][i] for i in indices]

    def load_state_dict(self, state_dict):
        if self.seed != state_dict["seed"]:
            raise ValueError(
                f"Cannot load state dict with different seed . Origin: {state_dict['seed']}, New: {self.seed}"
            )

        if self.pack_max_length != state_dict["pack_max_length"]:
            raise ValueError(
                "Cannot load state dict with different pack_max_length "
                f". Origin: {state_dict['pack_max_length']}, New: {self.pack_max_length}"
            )

        if self.global_pack != state_dict["global_pack"]:
            raise ValueError(
                "Cannot load state dict with different global_pack "
                f". Origin: {state_dict['global_pack']}, New: {self.global_pack}"
            )

    def get_state_dict(self):
        return {
            "pack_max_length": self.pack_max_length,
            "seed": self.seed,
            "global_pack": self.global_pack,
        }


def closest_sum_indices(buffer, value):
    buffer = np.array(buffer)
    sorted_indices = np.argsort(buffer)
    closest_sum = 0
    closest_indices = []

    for idx in sorted_indices:
        closest_sum += buffer[idx]
        if closest_sum <= value:
            closest_indices.append(int(idx))
        if closest_sum >= value:
            break

    return closest_indices


def get_pack_chunk_infos(
    inds,
    dataset_id,
    target,
    flash_attn_block_size,
    pack_len_type,
    pack_extra_buffer_size,
    num_tokens=None,
    shm_name=None,
    shape=None,
    dtype=None,
):
    if num_tokens is None:
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        num_tokens = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)

    item_buffer = []
    length_buffer = []
    longest = 0
    num_patch = 0

    pack_infos = []

    while len(inds) > 0:
        shfl_i = inds.pop()

        if num_tokens[shfl_i] + sum(length_buffer) <= target:
            item_buffer.append(shfl_i)
            length_buffer.append(num_tokens[shfl_i])
            num_patch += (round(num_tokens[shfl_i] / flash_attn_block_size)) ** 2
            longest = max(longest, num_tokens[shfl_i])
        else:
            if len(item_buffer) > 0:
                if sum(length_buffer) == target:
                    info = {
                        "dataset_id": dataset_id,
                        "indices": item_buffer,
                    }
                    if pack_len_type == "total_block":
                        info["longest"] = int(num_patch)
                    elif pack_len_type == "max_block":
                        info["longest"] = int(longest)
                    pack_infos.append(info)
                else:
                    if pack_extra_buffer_size > 0:
                        # Try to find the most suitable.
                        buffer_index = inds[-pack_extra_buffer_size:]
                        buffer = num_tokens[buffer_index]
                        closest_indices = closest_sum_indices(buffer, target - sum(length_buffer))
                        indices_to_remove = []
                        for closest_inds in closest_indices:
                            indices_to_remove.append(closest_inds + len(inds) - len(buffer_index))
                            item_buffer.append(buffer_index[closest_inds])
                            length_buffer.append(num_tokens[buffer_index[closest_inds]])
                            num_patch += (round(num_tokens[buffer_index[closest_inds]] / flash_attn_block_size)) ** 2
                            longest = max(longest, num_tokens[buffer_index[closest_inds]])

                        indices_to_remove = sorted(indices_to_remove, reverse=True)
                        for index in indices_to_remove:
                            inds.pop(index)

                    info = {
                        "dataset_id": dataset_id,
                        "indices": item_buffer,
                    }
                    if pack_len_type == "total_block":
                        info["longest"] = int(num_patch)
                    elif pack_len_type == "max_block":
                        info["longest"] = int(longest)

                    pack_infos.append(info)

            item_buffer = [shfl_i]
            length_buffer = [num_tokens[shfl_i]]
            longest = num_tokens[shfl_i]
            num_patch = (round(num_tokens[shfl_i] / flash_attn_block_size)) ** 2

    if len(item_buffer) > 0:
        info = {
            "dataset_id": dataset_id,
            "indices": item_buffer,
        }
        if pack_len_type == "total_block":
            info["longest"] = int(num_patch)
        elif pack_len_type == "max_block":
            info["longest"] = int(longest)
        pack_infos.append(info)
    return pack_infos


class ExpandSoftPackDataset(_LegacySoftPackDataset):
    def __init__(
        self,
        datasets: list[JsonlDataset],
        pack_max_length: int = 2048,
        global_pack: bool = False,
        pack_len_type="total_block",
        flash_attn_block_size: int = 128,
        pack_extra_buffer_size: int = 1000,
        pack_chunk_size: int = 10000,
        pack_workers: int = 8,
        seed: int | None = None,
    ):
        self.pack_len_type = pack_len_type
        assert self.pack_len_type in ["total_block", "max_block"], f"Invalid pack_len_type: {self.pack_len_type}"
        self.flash_attn_block_size = flash_attn_block_size
        self.pack_extra_buffer_size = pack_extra_buffer_size
        self.pack_workers = pack_workers
        self.torch_random_generator = torch.Generator()
        self.pack_chunk_size = pack_chunk_size
        if seed is not None:
            self.torch_random_generator.manual_seed(seed)
        logger.info(f"Using {self.pack_workers} pack workers for packing datasets.")

        super().__init__(
            datasets=datasets,
            pack_max_length=pack_max_length,
            global_pack=global_pack,
            seed=seed,
        )

    def get_pack_infos(self, dataset, dataset_id, num_tokens):
        inds = torch.randperm(len(dataset), generator=self.torch_random_generator).tolist()
        if self.pack_workers <= 1:
            pack_infos = []
            for i in range(0, len(inds), self.pack_chunk_size):
                chunk_inds = inds[i : i + self.pack_chunk_size]
                chunk_pack_infos = get_pack_chunk_infos(
                    chunk_inds,
                    dataset_id,
                    self.pack_max_length,
                    self.flash_attn_block_size,
                    self.pack_len_type,
                    self.pack_extra_buffer_size,
                    num_tokens,
                )
                pack_infos.extend(chunk_pack_infos)
        else:
            chunks_inds = [inds[i : i + self.pack_chunk_size] for i in range(0, len(inds), self.pack_chunk_size)]

            shm = shared_memory.SharedMemory(create=True, size=num_tokens.nbytes)
            shm_array = np.ndarray(num_tokens.shape, dtype=num_tokens.dtype, buffer=shm.buf)
            np.copyto(shm_array, num_tokens)

            mp_context = multiprocessing.get_context("fork")
            process_chunk_with_args = partial(
                get_pack_chunk_infos,
                dataset_id=dataset_id,
                target=self.pack_max_length,
                flash_attn_block_size=self.flash_attn_block_size,
                pack_len_type=self.pack_len_type,
                pack_extra_buffer_size=self.pack_extra_buffer_size,
                shm_name=shm.name,
                shape=num_tokens.shape,
                dtype=num_tokens.dtype,
            )
            with ProcessPoolExecutor(max_workers=self.pack_workers, mp_context=mp_context) as executor:
                results = list(tqdm(executor.map(process_chunk_with_args, chunks_inds)))

            pack_infos = []
            for result in results:
                pack_infos.extend(result)

            shm.close()
            shm.unlink()

        total_index = []
        for infos in pack_infos:
            total_index.extend(infos["indices"])
        assert len(dataset) == len(total_index) == len(set(total_index))

        pack_infos = Dataset.from_list(pack_infos)
        return pack_infos


class HardPackDataset(torch.utils.data.Dataset):
    def __init__(self, datasets, target=2048, global_pack=True):
        if global_pack:
            num_tokens = [np.concatenate([dset.num_tokens for dset in datasets])]
            datasets = [ConcatDataset(datasets)]
        else:
            num_tokens = [dset.num_tokens for dset in datasets]
        self.datasets = datasets
        self.target = target

        pack_infos = []
        for i, dataset in enumerate(self.datasets):
            _info = self.get_pack_info(dataset, i, num_tokens[i])
            pack_infos.append(_info)

        _ranges_left = []
        _ranges_right = []
        _num_packed_samples = []
        _indices = []
        _max_length_per_pack = []
        _dataset_id = []
        for info in pack_infos:
            _ranges_left.extend(info["begin_indices"])
            _ranges_right.extend(info["end_indices"])
            _num_packed_samples.append(info["num_packed_samples"])
            _indices.extend(info["indices"])
            _max_length_per_pack.extend(info["longest_per_pack"])
            _dataset_id.extend(info["dataset_id"])

        self.pack_infos = {
            "begin_indices": _ranges_left,
            "end_indices": _ranges_right,
            "num_packed_samples": _num_packed_samples,
            "indices": _indices,
            "longest_per_pack": _max_length_per_pack,
            "dataset_id": _dataset_id,
        }

    @classmethod
    def _cal_max_length(cls, begin, end, shfl_item_begin_indices, shfl_item_end_indices):
        left = bisect.bisect_right(shfl_item_begin_indices, begin)
        right = bisect.bisect_left(shfl_item_end_indices, end)
        max_length = 0
        for i in range(left, right):
            item_begin = shfl_item_begin_indices[i]
            item_end = shfl_item_end_indices[i]
            inner_l = max(begin, item_begin) - item_begin
            inner_r = min(end, item_end) - item_begin
            trunc_size = inner_r - inner_l
            max_length = max(max_length, trunc_size)
        return max_length

    def get_pack_info(self, dataset, dataset_id, num_tokens):
        # The number of data items after packing
        num_packed_samples = int(num_tokens.sum() / self.target)

        # Shuffle the order of the original dataset
        # The packing will proceed according to the order after shuffle.
        # Assume the following conditions hold:
        #   (1) shfl_inds = [3, 1, 2, 0]
        #   (2) self._ori_lens[3] + self._ori_lens[1] = max_length
        #   (3) self._ori_lens[2] + self._ori_lens[0] = max_length
        # Ultimately, dataset[3] and dataset[1] will be combined into a new
        # data, and dataset[2] and dataset[0] will be combined into a new data.
        inds = [i for i in range(len(dataset))]
        # if seed is not None:
        #     random.seed(seed)
        random.shuffle(inds)
        shfl_inds = inds

        # shuffled cumulative lengths
        shfl_lens = [num_tokens[i] for i in shfl_inds]
        shfl_cu_lens = list(itertools.accumulate(shfl_lens))

        shfl_item_begin_indices = [0] + shfl_cu_lens[:-1]
        shfl_item_end_indices = shfl_cu_lens

        max_length_per_pack = []
        belong_dataset_ids = []
        for i in range(num_packed_samples):
            begin = i * self.target
            end = (i + 1) * self.target
            max_length_per_pack.append(
                self._cal_max_length(begin, end, shfl_item_begin_indices, shfl_item_end_indices)
            )
            belong_dataset_ids.append(dataset_id)

        pack_infos = {
            "begin_indices": shfl_item_begin_indices,
            "end_indices": shfl_item_end_indices,
            "num_packed_samples": num_packed_samples,
            "indices": shfl_inds,
            "dataset_id": belong_dataset_ids,
            "longest_per_pack": max_length_per_pack,
        }

        # pack_infos = Dataset.from_list(pack_infos)

        return pack_infos

    def _pack_ids_and_labels_in_range(self, begin: int, end: int):
        """Packs ids and labels in a given range using bisection method.

        Args:
            begin: Index indicating the beginning of the range.
            end: Index indicating the end of the range.

        Returns:
            A tuple containing packed ids, labels, and cumulative lengths.
        """

        # Use binary search to find dataset positions that fall within begin
        # and end range
        left = bisect.bisect_right(self.pack_infos["begin_indices"], begin)
        right = bisect.bisect_left(self.pack_infos["end_indices"], end)

        trunc_input_ids = []
        trunc_labels = []
        trunc_sizes = []

        for i in range(left, right):
            # Determine the real range we will cut in current original item
            item_begin = self.pack_infos["begin_indices"][i]
            item_end = self.pack_infos["end_indices"][i]

            # Calculate exact positions within current dataset item
            inner_l = max(begin, item_begin) - item_begin
            inner_r = min(end, item_end) - item_begin

            # Get original data and labels
            ori_idx = self.pack_infos["indices"][i]
            ori_dataset_id = self.pack_infos["dataset_id"][i]
            ori_input_ids = self.datasets[ori_dataset_id][ori_idx]["input_ids"]
            ori_labels = self.datasets[ori_dataset_id][ori_idx]["labels"]

            # Add original data and labels from calculated positions
            # to trunc_ids and trunc_labels
            trunc_input_ids.extend(ori_input_ids[inner_l:inner_r])
            trunc_labels.extend(ori_labels[inner_l:inner_r])
            trunc_sizes.append(inner_r - inner_l)

        # return populated lists of truncated ids, labels and their cumulative
        # lengths
        return trunc_input_ids, trunc_labels, trunc_sizes

    def __len__(self):
        return len(self.pack_infos["indices"])

    def __getitem__(self, item):
        """Returns a dict containing packed data in the given item.

        Args:
            item: An index to retrieve packed data.

        Returns:
            A dict including packed input_ids, labels, and cumulative_len.
        """
        # The cumulative length from the start position of this data
        begin = item * self.target
        # The cumulative length from the end position of this data
        end = (item + 1) * self.target

        # Extract data within the range from the shuffled original dataset.
        _res = self._pack_ids_and_labels_in_range(begin, end)
        packed_input_ids, packed_labels, num_tokens = _res
        assert self.target == len(packed_input_ids) == len(packed_labels)

        packed = {
            "input_ids": packed_input_ids,
            "labels": packed_labels,
            "num_tokens": num_tokens,
        }

        return packed
