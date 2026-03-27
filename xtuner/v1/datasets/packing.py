# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import multiprocessing
import os
import random
import tempfile
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import cached_property, partial
from multiprocessing import shared_memory
from pathlib import Path
from typing import Sized, cast

import numpy as np
import torch
import xxhash
from datasets import Dataset, concatenate_datasets
from torch import distributed as dist
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

from xtuner.v1.utils import get_logger, is_local_rank0
from xtuner.v1.utils.executor import SharedPoolExecutor

from .jsonl import JsonlDataset
from .utils import _get_mmap_dir, ndarray_to_mmap
from .vlm_jsonl import VLMJsonlDataset


logger = get_logger()


def get_pack_infos_by_soft_split(inds: list[int], dataset_id: int, num_tokens: np.ndarray, pack_max_length: int):
    item_buffer: list[int] = []
    length_buffer: list[int] = []
    longest = 0

    pack_infos = []
    for shfl_i in inds:
        if num_tokens[shfl_i] + sum(length_buffer) <= pack_max_length:
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
    return pack_infos


class _LegacySoftPackDataset(torch.utils.data.Dataset):
    def __init__(self, datasets, pack_max_length=2048, global_pack=False, seed: int | None = None):
        self.random = random.Random()
        if seed is not None:
            self.random = random.Random(seed)

        if global_pack:
            num_tokens = [np.concatenate([dset.num_tokens for dset in datasets])]
            proxy_attn_flops = [np.concatenate([dset.proxy_attn_flops for dset in datasets])]
            assert len(num_tokens[0]) == len(proxy_attn_flops[0]), (
                f"num_tokens and proxy_attn_flops should have the same length after concatenation. but got {len(num_tokens[0])} and {len(proxy_attn_flops[0])}"
            )
            datasets = [ConcatDataset(datasets)]
        else:
            num_tokens = [dset.num_tokens for dset in datasets]
            proxy_attn_flops = [dset.proxy_attn_flops for dset in datasets]

        self.datasets = datasets
        self.seed = seed
        self.global_pack = global_pack
        self.pack_max_length = pack_max_length

        pack_infos = []
        for i, dataset in enumerate(self.datasets):
            _infos = self.get_pack_infos(dataset, i, num_tokens[i], proxy_attn_flops[i])
            pack_infos.append(_infos)
        self.pack_infos = concatenate_datasets(pack_infos)

    @property
    def longest(self):
        return self.pack_infos["longest"]

    def get_pack_infos(
        self, dataset: Sized, dataset_id: int, num_tokens: np.ndarray, proxy_attn_flops: np.ndarray | None = None
    ):
        inds = list(range(len(dataset)))
        self.random.shuffle(inds)

        pack_infos = get_pack_infos_by_soft_split(inds, dataset_id, num_tokens, self.pack_max_length)

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
    pack_extra_buffer_size,
    num_tokens=None,
    proxy_attn_flops=None,
    shm_name=None,
    shape=None,
    dtype=None,
    proxy_attn_flops_shm_name=None,
    proxy_attn_flops_dtype=None,
):
    if num_tokens is None:
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        num_tokens = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)

        existing_attn_shm = shared_memory.SharedMemory(name=proxy_attn_flops_shm_name)
        proxy_attn_flops = np.ndarray(shape, dtype=proxy_attn_flops_dtype, buffer=existing_attn_shm.buf)

    item_buffer = []
    length_buffer = []
    longest = 0
    pack_proxy_attn_flops = 0

    pack_infos = []

    while len(inds) > 0:
        shfl_i = inds.pop()

        if num_tokens[shfl_i] + sum(length_buffer) <= target:
            item_buffer.append(shfl_i)
            length_buffer.append(num_tokens[shfl_i])
            pack_proxy_attn_flops += proxy_attn_flops[shfl_i]
            longest = max(longest, num_tokens[shfl_i])
        else:
            if len(item_buffer) > 0:
                if sum(length_buffer) == target:
                    info = {
                        "dataset_id": dataset_id,
                        "indices": item_buffer,
                        "longest": int(pack_proxy_attn_flops),
                    }
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
                            pack_proxy_attn_flops += proxy_attn_flops[buffer_index[closest_inds]]
                            longest = max(longest, num_tokens[buffer_index[closest_inds]])

                        indices_to_remove = sorted(indices_to_remove, reverse=True)
                        for index in indices_to_remove:
                            inds.pop(index)

                    info = {
                        "dataset_id": dataset_id,
                        "indices": item_buffer,
                        "longest": int(pack_proxy_attn_flops),
                    }
                    pack_infos.append(info)

            item_buffer = [shfl_i]
            length_buffer = [num_tokens[shfl_i]]
            longest = num_tokens[shfl_i]
            pack_proxy_attn_flops = proxy_attn_flops[shfl_i]

    if len(item_buffer) > 0:
        info = {
            "dataset_id": dataset_id,
            "indices": item_buffer,
            "longest": int(pack_proxy_attn_flops),
        }
        pack_infos.append(info)
    return pack_infos


def get_pack_infos_by_expand_soft_split(
    inds: list[int],
    dataset_id: int,
    num_tokens: np.ndarray,
    proxy_attn_flops: np.ndarray,
    pack_max_length: int,
    pack_workers: int = 8,
    pack_chunk_size: int = 10000,
    pack_extra_buffer_size: int = 1000,
):
    assert len(num_tokens) == len(proxy_attn_flops), (
        "num_tokens and proxy_attn_flops should have the same length for shared memory."
    )
    if pack_workers <= 1:
        pack_infos = []
        for i in range(0, len(inds), pack_chunk_size):
            chunk_inds = inds[i : i + pack_chunk_size]
            chunk_pack_infos = get_pack_chunk_infos(
                chunk_inds,
                dataset_id,
                pack_max_length,
                pack_extra_buffer_size,
                num_tokens,
                proxy_attn_flops,
            )
            pack_infos.extend(chunk_pack_infos)
    else:
        chunks_inds = [inds[i : i + pack_chunk_size] for i in range(0, len(inds), pack_chunk_size)]
        shm = shared_memory.SharedMemory(create=True, size=num_tokens.nbytes)
        shm_array = np.ndarray(num_tokens.shape, dtype=num_tokens.dtype, buffer=shm.buf)
        np.copyto(shm_array, num_tokens)

        proxy_attn_flops_shm = shared_memory.SharedMemory(create=True, size=proxy_attn_flops.nbytes)
        proxy_attn_flops_shm_array = np.ndarray(
            num_tokens.shape, dtype=proxy_attn_flops.dtype, buffer=proxy_attn_flops_shm.buf
        )
        np.copyto(proxy_attn_flops_shm_array, proxy_attn_flops)

        mp_context = multiprocessing.get_context("fork")
        process_chunk_with_args = partial(
            get_pack_chunk_infos,
            dataset_id=dataset_id,
            target=pack_max_length,
            pack_extra_buffer_size=pack_extra_buffer_size,
            shm_name=shm.name,
            shape=num_tokens.shape,
            dtype=num_tokens.dtype,
            proxy_attn_flops_shm_name=proxy_attn_flops_shm.name,
            proxy_attn_flops_dtype=proxy_attn_flops.dtype,
        )
        with ProcessPoolExecutor(max_workers=pack_workers, mp_context=mp_context) as executor:
            results = list(tqdm(executor.map(process_chunk_with_args, chunks_inds)))

        pack_infos = []
        for result in results:
            pack_infos.extend(result)

        shm.close()
        shm.unlink()
        proxy_attn_flops_shm.close()
        proxy_attn_flops_shm.unlink()
    return pack_infos


class ExpandSoftPackDataset(_LegacySoftPackDataset):
    def __init__(
        self,
        datasets: Sequence[JsonlDataset],
        pack_max_length: int = 2048,
        global_pack: bool = False,
        pack_extra_buffer_size: int = 1000,
        pack_chunk_size: int = 10000,
        pack_workers: int = 8,
        seed: int | None = None,
    ):
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

    def get_pack_infos(
        self, dataset: Sized, dataset_id: int, num_tokens: np.ndarray, proxy_attn_flops: np.ndarray | None = None
    ):
        inds = torch.randperm(len(dataset), generator=self.torch_random_generator).tolist()
        pack_infos = get_pack_infos_by_expand_soft_split(
            inds,
            dataset_id,
            num_tokens,
            proxy_attn_flops,
            pack_max_length=self.pack_max_length,
            pack_workers=self.pack_workers,
            pack_chunk_size=self.pack_chunk_size,
            pack_extra_buffer_size=self.pack_extra_buffer_size,
        )
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
    lengths: np.ndarray,
) -> dict[str, np.ndarray]:
    n = len(i_chunk)
    out_dataset_id = np.full(n, dataset_id, dtype=np.int64)
    out_longest = np.empty(n, dtype=np.int64)
    indices_parts: list[np.ndarray] = []

    i_arr = np.asarray(i_chunk, dtype=np.int64)
    begins = i_arr * pack_max_length
    ends = begins + pack_max_length
    s_idxs = np.searchsorted(cu, begins, side="right").astype(np.int64) - 1
    e_idxs = np.searchsorted(cu, ends - 1, side="right").astype(np.int64) - 1
    out_start_offset = (begins - cu[s_idxs]).astype(np.int64)
    out_end_offset = (ends - cu[e_idxs]).astype(np.int64)

    for j in range(n):
        s_idx, e_idx = int(s_idxs[j]), int(e_idxs[j])
        s_off, e_off = int(out_start_offset[j]), int(out_end_offset[j])

        if s_idx == e_idx:
            out_longest[j] = e_off - s_off
        else:
            len_first = int(lengths[s_idx] - s_off)
            len_last = e_off
            mid_max = int(lengths[s_idx + 1 : e_idx].max()) if e_idx - s_idx > 1 else 0
            out_longest[j] = max(len_first, len_last, mid_max)

        indices_parts.append(inds_arr[s_idx : e_idx + 1])

    indices_flat = np.concatenate(indices_parts) if indices_parts else np.empty(0, dtype=np.int64)
    indices_lens = e_idxs - s_idxs + 1
    return {
        "dataset_id": out_dataset_id,
        "indices": indices_flat,
        "indices_cu_len": np.cumsum(indices_lens, dtype=np.int64),
        "start_offset": out_start_offset,
        "end_offset": out_end_offset,
        "longest": out_longest,
    }


def _merge_pack_infos(infos: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    cu_parts: list[np.ndarray] = []
    offset = np.int64(0)
    for r in infos:
        cu_parts.append(r["indices_cu_len"] + offset)
        offset += r["indices_cu_len"][-1]
    return {
        "dataset_id": np.concatenate([r["dataset_id"] for r in infos]),
        "indices": np.concatenate([r["indices"] for r in infos]),
        "indices_cu_len": np.concatenate(cu_parts),
        "start_offset": np.concatenate([r["start_offset"] for r in infos]),
        "end_offset": np.concatenate([r["end_offset"] for r in infos]),
        "longest": np.concatenate([r["longest"] for r in infos]),
    }


def get_pack_infos_by_hard_split(
    inds: np.ndarray, dataset_id: int, num_tokens: np.ndarray, pack_max_length: int, pack_workers: int = 1
):
    # number of packed samples
    shfl_inds = inds
    num_packed_samples = int(num_tokens.sum() / pack_max_length)

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

    lengths_arr = (cu_arr[1:] - cu_arr[:-1]).astype(np.int64)
    all_results: list[dict[str, np.ndarray]] = []

    if pack_workers > 1:
        # cu_arr, inds_arr, and lengths_arr are passed as partial_kwargs so
        # SharedPoolExecutor places them in POSIX shared memory.  All worker
        # processes attach to the same physical pages — no per-worker copy is
        # made.  For large datasets these arrays can be hundreds of MB, so
        # without shared memory the total footprint would scale linearly with
        # the number of workers.
        with SharedPoolExecutor(
            fn=_hard_pack_chunk_core,
            partial_kwargs={
                "dataset_id": dataset_id,
                "pack_max_length": pack_max_length,
                "cu": cu_arr,
                "inds_arr": inds_arr,
                "lengths": lengths_arr,
            },
            max_workers=pack_workers,
            mp_context="fork",
        ) as pool:
            all_results = list(tqdm(pool.map(chunks), total=len(chunks)))
    else:
        all_results = [
            _hard_pack_chunk_core(
                i_chunk,
                dataset_id=dataset_id,
                pack_max_length=pack_max_length,
                cu=cu_arr,
                inds_arr=inds_arr,
                lengths=lengths_arr,
            )
            for i_chunk in tqdm(chunks, total=len(chunks))
        ]
    return _merge_pack_infos(all_results)


class HardPackDataset(_LegacySoftPackDataset):
    # TODO: The shared-memory / mmap optimisation below applies only to
    # HardPackDataset.  SoftPackDataset stores pack_infos as a HuggingFace
    # Dataset whose "indices" column is a Python list shuffled every epoch,
    # so the result is not deterministic and cannot be content-addressed.
    # Extending this to SoftPack would require fixing the shuffle order into
    # the cache key and switching the indices storage to a flat ndarray.
    _PACK_INFO_FIELDS = ("dataset_id", "indices", "start_offset", "end_offset", "longest", "indices_cu_len")

    def __init__(
        self, datasets, pack_max_length=2048, global_pack=False, seed: int | None = None, pack_workers: int = 1
    ):
        self.pack_workers = pack_workers
        self.random = np.random.RandomState(seed) if seed is not None else np.random.RandomState()  # type: ignore

        # Create a single dedicated process group with a generous timeout for
        # all collective operations during dataset initialisation (ndarray_to_mmap
        # and _build_pack_infos).  This avoids triggering the default NCCL
        # watchdog when pack computation or large-array writes are slow.
        pack_pg = dist.new_group(timeout=datetime.timedelta(seconds=7200)) if dist.is_initialized() else None

        if global_pack:
            # Concatenate all datasets into a single virtual dataset so that
            # the packer can form packs that span sample boundaries across
            # datasets.  ndarray_to_mmap converts the concatenated array to a
            # file-backed mmap so that all ranks on the same node share a
            # single copy of the (potentially very large) num_tokens array
            # rather than each rank holding its own in-process copy.
            num_tokens = [ndarray_to_mmap(np.concatenate([dset.num_tokens for dset in datasets]), group=pack_pg)]
            datasets = [ConcatDataset(datasets)]
        else:
            # Per-dataset packing: keep each dataset's num_tokens array as-is.
            # No mmap conversion needed here since the arrays are already owned
            # by the individual JsonlDataset instances.
            num_tokens = [dset.num_tokens for dset in datasets]

        self.datasets = datasets
        self.seed = seed
        self.global_pack = global_pack
        self.pack_max_length = pack_max_length

        self.pack_infos = self._build_pack_infos(num_tokens, pack_pg=pack_pg)

    def _build_pack_infos(
        self, num_tokens: list[np.ndarray], pack_pg: "dist.ProcessGroup | None" = None
    ) -> dict[str, np.ndarray]:
        # Derive a content-addressed cache key from all inputs that determine
        # the final pack_infos (seed, pack_max_length, global_pack, and the
        # content of every num_tokens array).
        h = xxhash.xxh128()
        h.update(f"{self.seed}_{self.pack_max_length}_{self.global_pack}".encode())
        for nt in num_tokens:
            h.update(np.ascontiguousarray(nt).data.tobytes())
        pack_dir = _get_mmap_dir() / "pack_infos" / h.hexdigest()

        if is_local_rank0() and not pack_dir.exists():
            # Only rank 0 (per node) runs the full pack computation.  The
            # results are written to a content-addressed directory on shared
            # storage so that every other rank can load them without
            # recomputing.  Writing once and memory-mapping from disk lets all
            # ranks on a node share the same physical pages (see
            # _load_pack_infos_from_mmap), keeping per-node memory overhead
            # proportional to the data size rather than the rank count.
            all_infos = [
                self._compute_pack_infos(dataset, i, num_tokens[i]) for i, dataset in enumerate(self.datasets)
            ]
            self._save_pack_infos(_merge_pack_infos(all_infos), pack_dir)

        if dist.is_initialized():
            dist.barrier(group=pack_pg)
            if pack_pg is not None:
                dist.destroy_process_group(pack_pg)

        return self._load_pack_infos_from_mmap(pack_dir)

    def _save_pack_infos(self, pack_infos: dict[str, np.ndarray], pack_dir: Path) -> None:
        # Write to a temp directory then atomically rename so that other ranks
        # never observe a partially-written pack_dir if this rank is killed mid-write.
        pack_dir.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(tempfile.mkdtemp(dir=pack_dir.parent))
        try:
            for field in self._PACK_INFO_FIELDS:
                arr = pack_infos.get(field)
                if arr is not None:
                    np.save(str(tmp_dir / f"{field}.npy"), arr)
            os.rename(str(tmp_dir), str(pack_dir))
        except Exception:
            import shutil

            shutil.rmtree(str(tmp_dir), ignore_errors=True)
            raise

    def _load_pack_infos_from_mmap(self, pack_dir: Path) -> dict[str, np.ndarray]:
        # Load with mmap_mode="r" so that every rank on the same node maps the
        # same on-disk file into its virtual address space read-only.  The OS
        # automatically backs all of those mappings with a single set of
        # physical pages, meaning a node running N ranks holds only one copy of
        # the pack_infos arrays in RAM regardless of N.  This is critical at
        # large sample counts where the arrays can be hundreds of MB per dataset.
        return {
            field: np.load(str(pack_dir / f"{field}.npy"), mmap_mode="r")
            for field in self._PACK_INFO_FIELDS
            if (pack_dir / f"{field}.npy").exists()
        }

    def _compute_pack_infos(self, dataset: Sized, dataset_id: int, num_tokens: np.ndarray) -> dict[str, np.ndarray]:
        inds = np.arange(len(dataset), dtype=np.int64)
        self.random.shuffle(inds)  # type: ignore[arg-type]
        return get_pack_infos_by_hard_split(
            inds, dataset_id, num_tokens, pack_max_length=self.pack_max_length, pack_workers=self.pack_workers
        )

    def __getitem__(self, item: int):
        assert self.pack_infos is not None
        dataset_id = int(self.pack_infos["dataset_id"][item])
        ds = self.datasets[dataset_id]

        if item == 0:
            indices_start = 0
        else:
            indices_start = int(self.pack_infos["indices_cu_len"][item - 1])
        indices_end = int(self.pack_infos["indices_cu_len"][item])
        indices = self.pack_infos["indices"][indices_start:indices_end].tolist()
        s_off = int(self.pack_infos["start_offset"][item])
        e_off = int(self.pack_infos["end_offset"][item])

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

    def __len__(self):
        assert self.pack_infos is not None
        return self.pack_infos["dataset_id"].shape[0]

    @cached_property  # type: ignore[override]
    def longest(self):
        assert self.pack_infos is not None
        return self.pack_infos["longest"].tolist()

    def get_state_dict(self):
        return {}

    def load_state_dict(self, state_dict): ...


class MLLMPretrainHybridPackDataset(TorchDataset):
    def __init__(
        self,
        datasets: list[JsonlDataset],
        pack_max_length: int = 2048,
        global_pack: bool = False,
        pack_workers: int = 8,
        seed: int | None = None,
        pack_extra_buffer_size: int = 1000,  # for ExpandSoftPackDataset
        pack_chunk_size: int = 10000,  # for ExpandSoftPackDataset
    ):
        self.seed = seed
        self.pack_max_length = pack_max_length
        self.global_pack = global_pack
        self.pack_workers = pack_workers
        self.pack_extra_buffer_size = pack_extra_buffer_size
        self.pack_chunk_size = pack_chunk_size

        hard_pack_groups = []
        soft_pack_groups = []
        for dset in datasets:
            if isinstance(dset, VLMJsonlDataset):
                soft_pack_groups.append(dset)
            elif isinstance(dset, JsonlDataset):
                hard_pack_groups.append(dset)

        dataset_list: list[HardPackDataset | ExpandSoftPackDataset] = []

        if hard_pack_groups:
            hard_pack_dataset = HardPackDataset(
                datasets=hard_pack_groups,
                pack_max_length=pack_max_length,
                global_pack=global_pack,
                seed=seed,
                pack_workers=pack_workers,
            )
            dataset_list.append(hard_pack_dataset)

        if soft_pack_groups:
            soft_pack_dataset = ExpandSoftPackDataset(
                datasets=soft_pack_groups,
                pack_max_length=pack_max_length,
                global_pack=global_pack,
                pack_extra_buffer_size=pack_extra_buffer_size,
                pack_chunk_size=pack_chunk_size,
                pack_workers=pack_workers,
                seed=seed,
            )
            dataset_list.append(soft_pack_dataset)

        assert dataset_list, "No datasets provided for packing."
        self.datasets: ConcatDataset[HardPackDataset | ExpandSoftPackDataset] = ConcatDataset(dataset_list)

    @cached_property
    def longest(self):
        longest_list = []
        for dataset in self.datasets.datasets:
            longest_list.extend(cast(HardPackDataset | ExpandSoftPackDataset, dataset).longest)
        return longest_list

    def __getitem__(self, item: int):
        return self.datasets[item]

    def __len__(self) -> int:
        return len(self.datasets)

    def get_state_dict(self):
        return {
            "pack_max_length": self.pack_max_length,
            "seed": self.seed,
            "global_pack": self.global_pack,
            "pack_extra_buffer_size": self.pack_extra_buffer_size,
            "pack_chunk_size": self.pack_chunk_size,
        }

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

        if self.pack_extra_buffer_size != state_dict["pack_extra_buffer_size"]:
            raise ValueError(
                "Cannot load state dict with different pack_extra_buffer_size "
                f". Origin: {state_dict['pack_extra_buffer_size']}, New: {self.pack_extra_buffer_size}"
            )

        if self.pack_chunk_size != state_dict["pack_chunk_size"]:
            raise ValueError(
                "Cannot load state dict with different pack_chunk_size "
                f". Origin: {state_dict['pack_chunk_size']}, New: {self.pack_chunk_size}"
            )
