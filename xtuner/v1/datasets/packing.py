# Copyright (c) OpenMMLab. All rights reserved.
import multiprocessing
import random
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import shared_memory
from typing import Sized

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


def _hard_pack_chunk_core(
    i_chunk: list[int],
    *,
    dataset_id: int,
    pack_max_length: int,
    cu: np.ndarray,
    inds_arr: np.ndarray,
) -> list[dict]:
    lengths = cu[1:] - cu[:-1]
    out: list[dict] = []
    for i in i_chunk:
        begin = i * pack_max_length
        end = (i + 1) * pack_max_length

        s_idx = int(np.searchsorted(cu, begin, side="right") - 1)
        e_idx = int(np.searchsorted(cu, end - 1, side="right") - 1)

        s_off = int(begin - cu[s_idx])
        e_off = int(end - cu[e_idx])

        if s_idx == e_idx:
            longest = int(e_off - s_off)
        else:
            len_first = int(lengths[s_idx] - s_off)
            len_last = int(e_off)
            mid_max = int(lengths[s_idx + 1 : e_idx].max()) if e_idx - s_idx > 1 else 0
            longest = max(len_first, len_last, mid_max)

        out.append(
            {
                "dataset_id": dataset_id,
                "indices": inds_arr[s_idx : e_idx + 1].tolist(),
                "start_offset": s_off,
                "end_offset": e_off,
                "longest": longest,
            }
        )
    return out


def _hard_pack_chunk(
    i_chunk: list[int],
    *,
    dataset_id: int,
    pack_max_length: int,
    cu_shm_name: str,
    cu_shape,
    cu_dtype,
    inds_shm_name: str,
    inds_shape,
    inds_dtype,
):
    existing_cu = shared_memory.SharedMemory(name=cu_shm_name)
    cu: np.ndarray = np.ndarray(cu_shape, dtype=cu_dtype, buffer=existing_cu.buf)

    existing_inds = shared_memory.SharedMemory(name=inds_shm_name)
    inds_arr: np.ndarray = np.ndarray(inds_shape, dtype=inds_dtype, buffer=existing_inds.buf)

    out = _hard_pack_chunk_core(
        i_chunk,
        dataset_id=dataset_id,
        pack_max_length=pack_max_length,
        cu=cu,
        inds_arr=inds_arr,
    )

    existing_cu.close()
    existing_inds.close()
    return out


class HardPackDataset(_LegacySoftPackDataset):
    def __init__(
        self, datasets, pack_max_length=2048, global_pack=False, seed: int | None = None, pack_workers: int = 1
    ):
        self.pack_workers = pack_workers
        super().__init__(
            datasets=datasets,
            pack_max_length=pack_max_length,
            global_pack=global_pack,
            seed=seed,
        )

    def _get_single_pack_info(self, begin: int, end: int, cu_seq_lens: np.ndarray, inds: list[int]) -> dict:
        # cu_seq_lens: e.g. [0, 1241, 2141, 3141], strictly increasing
        cu = np.asarray(cu_seq_lens, dtype=np.int64).reshape(-1)
        if cu.ndim != 1 or cu.size < 2:
            raise ValueError(
                "cu_seq_lens must be a 1-D increasing sequence with at least two elements (including 0 and total)."
            )

        total = int(cu[-1])
        if not (0 <= begin < end <= total):
            raise ValueError(f"Invalid range: begin={begin}, end={end}, total={total}")

        # Find the sample index containing 'begin' (interval [cu[i], cu[i+1]))
        start_sample_idx = int(np.searchsorted(cu, begin, side="right") - 1)
        # Find the sample index containing 'end - 1'
        end_sample_idx = int(np.searchsorted(cu, end - 1, side="right") - 1)

        # Offsets inside the start/end samples
        start_offset = int(begin - cu[start_sample_idx])
        end_offset = int(end - cu[end_sample_idx])

        # Max original sample length among [start_sample_idx, end_sample_idx]
        lengths = cu[1:] - cu[:-1]
        if start_sample_idx == end_sample_idx:
            # If the interval falls entirely within a single sample, the maximum length equals that sample’s
            # sliced segment length.
            longest = int(end_offset - start_offset)
        else:
            # The effective length of the first sample is its original length minus the start offset.
            len_first = int(lengths[start_sample_idx] - start_offset)
            # The effective length of the last sample is the end_offset.
            len_last = int(end_offset)
            # For middle samples, take the maximum of their original lengths (0 if there are no middle samples).
            mid_max = (
                int(lengths[start_sample_idx + 1 : end_sample_idx].max())
                if end_sample_idx - start_sample_idx > 1
                else 0
            )
            longest = max(len_first, len_last, mid_max)

        return {
            "indices": inds[start_sample_idx : end_sample_idx + 1],
            "start_offset": start_offset,
            "end_offset": end_offset,
            "longest": longest,
        }

    def get_pack_infos(self, dataset: Sized, dataset_id: int, num_tokens: np.ndarray):
        # number of packed samples
        num_packed_samples = int(num_tokens.sum() / self.pack_max_length)

        # shuffled indices
        inds = list(range(len(dataset)))
        self.random.shuffle(inds)
        shfl_inds = inds

        # shuffled cumulative lengths with leading 0
        shfl_lens: np.ndarray = np.take(num_tokens, shfl_inds)
        shfl_cu_lens = np.cumsum(shfl_lens, dtype=np.int64)
        shfl_cu_lens = np.insert(shfl_cu_lens, 0, 0).astype(np.int64, copy=False)

        # shared memory for cu and inds
        cu_arr = np.asarray(shfl_cu_lens, dtype=np.int64).reshape(-1)
        inds_arr = np.asarray(shfl_inds, dtype=np.int64).reshape(-1)

        # chunk tasks
        chunk_size = 10000
        i_all = list(range(num_packed_samples))
        chunks = [i_all[i : i + chunk_size] for i in range(0, len(i_all), chunk_size)]

        pack_infos_list = []

        if self.pack_workers > 1:
            # Use fork to inherit read-only arrays; no extra shared memory copy needed
            mp_context = multiprocessing.get_context("fork")
            fn = partial(
                _hard_pack_chunk_core,
                dataset_id=dataset_id,
                pack_max_length=self.pack_max_length,
                cu=cu_arr,
                inds_arr=inds_arr,
            )
            with ProcessPoolExecutor(max_workers=self.pack_workers, mp_context=mp_context) as ex:
                for res in tqdm(ex.map(fn, chunks), total=len(chunks)):
                    pack_infos_list.extend(res)
        else:
            # single-process path, reuse the same core
            for i_chunk in tqdm(chunks, total=len(chunks)):
                pack_infos_list.extend(
                    _hard_pack_chunk_core(
                        i_chunk,
                        dataset_id=dataset_id,
                        pack_max_length=self.pack_max_length,
                        cu=cu_arr,
                        inds_arr=inds_arr,
                    )
                )

        pack_infos = Dataset.from_list(pack_infos_list)
        return pack_infos

    def __len__(self):
        return len(self.pack_infos)

    def __getitem__(self, item):
        info = self.pack_infos[item]
        dataset_id = info["dataset_id"]
        ds = self.datasets[dataset_id]

        indices = info["indices"]
        s_off = info["start_offset"]
        e_off = info["end_offset"]

        packed_list: list[dict] = []

        for i in range(len(indices)):
            idx = indices[i]
            sample = ds[idx]
            ids = sample["input_ids"]
            labs = sample.get("labels", None)

            st = 0 if i != 0 else s_off
            ed = len(ids) if i != len(indices) - 1 else e_off

            packed_list.append(
                {
                    "input_ids": ids[st:ed],
                    "labels": labs[st:ed] if labs is not None else None,
                    "num_tokens": ed - st,
                }
            )
        assert (total_num_tokens := sum(i["num_tokens"] for i in packed_list)) == self.pack_max_length, (
            f"Internal Error! Found size: {total_num_tokens} mismatch after hard packing."
        )
        return packed_list

    def get_state_dict(self):
        return {}

    def load_state_dict(self, state_dict): ...
