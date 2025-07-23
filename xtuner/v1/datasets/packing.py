# Copyright (c) OpenMMLab. All rights reserved.
import multiprocessing
import os
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


logger = get_logger()


class SoftPackDataset(torch.utils.data.Dataset):
    def __init__(self, datasets, target=2048, blend=False, seed: int | None = None):
        self.random = random.Random()
        if seed is not None:
            self.random = random.Random(seed)

        if blend:
            num_tokens = [np.concatenate([dset.num_tokens for dset in datasets])]
            datasets = [ConcatDataset(datasets)]
        else:
            num_tokens = [dset.num_tokens for dset in datasets]
        self.datasets = datasets
        self.target = target

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
            if num_tokens[shfl_i] + sum(length_buffer) <= self.target:
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


class ExpandSoftPackDataset(SoftPackDataset):
    def __init__(
        self,
        *args,
        pack_len_type="total_block",
        flash_attn_block_size=128,
        pack_extra_buffer_size=1000,
        seed: int | None = None,
        **kwargs,
    ):
        self.pack_len_type = pack_len_type
        assert self.pack_len_type in ["total_block", "max_block"], f"Invalid pack_len_type: {self.pack_len_type}"
        self.flash_attn_block_size = flash_attn_block_size
        self.pack_extra_buffer_size = pack_extra_buffer_size
        self.pack_workers = int(os.environ.get("XTUNER_PACK_WORKERS", 1))
        self.torch_random_generator = torch.Generator()
        if seed is not None:
            self.torch_random_generator.manual_seed(seed)
        logger.info(f"Using {self.pack_workers} pack workers for packing datasets.")
        super().__init__(*args, **kwargs)

    def get_pack_infos(self, dataset, dataset_id, num_tokens):
        inds = torch.randperm(len(dataset), generator=self.torch_random_generator).tolist()
        if self.pack_workers <= 1:
            pack_infos = get_pack_chunk_infos(
                inds,
                dataset_id,
                self.target,
                self.flash_attn_block_size,
                self.pack_len_type,
                self.pack_extra_buffer_size,
                num_tokens,
            )
        else:
            chunk_size = (len(inds) + self.pack_workers - 1) // self.pack_workers
            chunks_inds = [inds[i : i + chunk_size] for i in range(0, len(inds), chunk_size)]

            shm = shared_memory.SharedMemory(create=True, size=num_tokens.nbytes)
            shm_array = np.ndarray(num_tokens.shape, dtype=num_tokens.dtype, buffer=shm.buf)
            np.copyto(shm_array, num_tokens)

            mp_context = multiprocessing.get_context("fork")
            process_chunk_with_args = partial(
                get_pack_chunk_infos,
                dataset_id=dataset_id,
                target=self.target,
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
