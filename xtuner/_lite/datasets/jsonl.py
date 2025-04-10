# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import json
import math
import multiprocessing
import os
import random
import signal
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Any, Callable

import numpy as np
import torch
from filelock import SoftFileLock
from mmengine import mkdir_or_exist
from torch import distributed as dist
from tqdm import tqdm

from xtuner._lite import get_logger

from .cache import CachableTokenizeFunction, CacheObj, calculate_file_sha256

logger = get_logger()


class JsonlDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path,
        sample_ratio: float = 1.0,
        tokenize_fn: Callable[[Any], CacheObj] | None = None,
        cache_dir: str | None = None,
        max_length: int | None = None,
    ):
        super().__init__()

        self.tokenize_fn = tokenize_fn
        self.path = path
        self.tokenizer_workers = int(os.environ.get("XTUNER_TOKENIZE_WORKERS", 8))

        if cache_dir:
            if os.path.exists(cache_dir):
                assert os.path.isdir(cache_dir)
            else:
                mkdir_or_exist(cache_dir)

            file_hash = calculate_file_sha256(path)
            file_cache_dir = os.path.join(cache_dir, file_hash)

            if file_hash not in os.listdir(cache_dir):
                mkdir_or_exist(file_cache_dir)

            _file_lock = self._cache_acquire(file_cache_dir)

            _cached_file = os.path.join(file_cache_dir, "offsets.npy")
            if not dist.is_initialized() or dist.get_rank() == 0:
                if not os.path.exists(_cached_file):
                    offsets = self.count_offsets(file_cache_dir)

            if dist.is_initialized():
                dist.barrier()
            logger.info(f"Loading `offsets` from cache: {_cached_file}")
            offsets = np.load(_cached_file)

            if tokenize_fn and isinstance(tokenize_fn, CachableTokenizeFunction):
                tok_hash = tokenize_fn.hash()
                tok_cache_dir = os.path.join(file_cache_dir, tok_hash)
                if tok_hash not in os.listdir(file_cache_dir):
                    mkdir_or_exist(tok_cache_dir)

                if "num_tokens.npy" in os.listdir(tok_cache_dir):
                    _cached_file = os.path.join(tok_cache_dir, "num_tokens.npy")
                    logger.info(f"Loading `num_tokens` from cache: {_cached_file}")
                    num_tokens = np.load(_cached_file)
                else:
                    num_tokens = self.count_tokens(offsets, tok_cache_dir)
            elif tokenize_fn:
                logger.warning(
                    f"{tokenize_fn.__name__} is not an instance of "
                    "`CachableTokenizeFunction`, data will always "
                    "be re-tokenized during training!"
                )
                num_tokens = self.count_tokens(offsets)
            else:
                num_tokens = None

            offsets = offsets
            num_tokens = num_tokens
            self._cache_release(_file_lock)
        else:
            offsets = self.count_offsets()
            num_tokens = None
            if tokenize_fn:
                num_tokens = self.count_tokens(offsets)

        _sampled = [i for i in range(len(offsets))]

        if num_tokens is not None and max_length is not None:
            assert isinstance(max_length, int)
            _filtered = [
                x for i, x in enumerate(_sampled) if num_tokens[i] < max_length
            ]

            if len(_filtered) < len(_sampled):
                missed_num = len(_sampled) - len(_filtered)
                logger.warning(
                    f"{path} has {missed_num} prompt length>{max_length}, discard."
                )

            _sampled = _filtered

        _target_num_samples = int(len(_sampled) * sample_ratio)
        self.sampled = _sampled * int(sample_ratio)
        self.sampled.extend(
            random.sample(_sampled, _target_num_samples - len(self.sampled))
        )

        if num_tokens is not None:
            num_tokens = num_tokens[self.sampled]

        self.num_tokens = num_tokens
        self.offsets = offsets[self.sampled]

    def count_offsets(self, cache_dir=None):
        offsets = [0]
        with open(self.path) as f:
            lines = f.readlines()
            for line in lines[:-1]:
                offsets.append(offsets[-1] + len(line.encode()))

        offsets = np.array(offsets)

        if cache_dir and (not dist.is_initialized() or dist.get_rank() == 0):
            save_path = os.path.join(cache_dir, "offsets.npy")
            np.save(save_path, offsets)

        return offsets

    def _tokenize_by_offset(self, offset, only_num_tokens=False):
        with open(self.path) as f:
            f.seek(offset)
            data = json.loads(f.readline())
        tokenize = self.tokenize_fn(data)
        if only_num_tokens:
            tokenize = {"num_tokens": tokenize["num_tokens"]}
        return tokenize

    def count_tokens(self, offsets, cache_dir=None):
        num_samples = len(offsets)

        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0

        num_per_rank = math.ceil(num_samples / world_size)

        start = rank * num_per_rank
        end = (rank + 1) * num_per_rank
        offsets_shard = offsets[start:end]

        desc = f"[Rank {rank}] {self.path}"
        chunk_size = min(1024, max(1, len(offsets_shard) // self.tokenizer_workers))

        mp_context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=self.tokenizer_workers, mp_context=mp_context
        ) as executor:
            tokenized = list(
                tqdm(
                    executor.map(
                        partial(self._tokenize_by_offset, only_num_tokens=True),
                        offsets_shard,
                        chunksize=chunk_size,
                    ),
                    desc=desc,
                    total=len(offsets_shard),
                )
            )

        _num_tokens = [data["num_tokens"] for data in tokenized]
        _num_tokens = np.array(_num_tokens)

        if dist.is_initialized():
            # TODO:
            # This is a workaround for `all_gather_object` would hang when
            # using `nccl` backend. Maybe we could find a better way to
            # synchronize the `num_tokens` since datasets are not always initialized
            # with the world size.
            if "gloo" in (backend := dist.get_backend()):
                tokenize_group = dist.new_group(backend=backend)
            else:
                tokenize_group = dist.new_group()
            num_tokens = [None] * world_size
            dist.all_gather_object(num_tokens, _num_tokens, group=tokenize_group)
            num_tokens = np.concatenate(num_tokens, axis=0)
        else:
            num_tokens = _num_tokens

        if rank == 0 and cache_dir:
            save_path = os.path.join(cache_dir, "num_tokens.npy")
            np.save(save_path, num_tokens)

        return num_tokens

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, item):
        """Returns a dict containing packed data in the given item.

        Args:
            item: An index to retrieve packed data.

        Returns:
            A dict including packed input_ids, labels, and cumulative_len.
        """
        with open(self.path) as f:
            f.seek(self.offsets[item])
            line = f.readline()

        raw_data = json.loads(line)

        if self.tokenize_fn:
            tokenized_data = self.tokenize_fn(raw_data)
            return tokenized_data
        else:
            return raw_data

    def _cache_acquire(self, lock_dir: str):
        """Acquires a cache lock to prevent simultaneous read/write operations
        on the cache directory.

        This function is designed to handle scenarios where multiple independent processes are performing
        data caching, such as when multiple training tasks are started simultaneously. These training tasks
        are independent of each other, and simultaneous read/write operations on the cache can lead to bugs.

        To avoid such issues, `_cache_acquire` ensures that only one training task is processing data at any
        given time. Within the same training task, distributed initialization with predefined ranks is used
        to avoid simultaneous read/write operations.
        """  # noqa: E501
        if dist.is_initialized():
            if "gloo" in (backend := dist.get_backend()):
                _cache_lock_group = dist.new_group(
                    backend=backend, timeout=datetime.timedelta(seconds=1800)
                )
            else:
                _cache_lock_group = dist.new_group(
                    timeout=datetime.timedelta(seconds=1800)
                )
            rank = dist.get_rank()
        else:
            rank = 0

        lock_dir = os.path.join(lock_dir, "__lock")
        mkdir_or_exist(lock_dir)

        if rank == 0:
            logger.info(
                f"Acquiring lock for lock_dir, if it hangs over 30m, please remove {lock_dir} it manually."
            )

        _filelock = SoftFileLock(os.path.join(lock_dir, f"{rank}.lock"))

        # Register a signal handler for SIGINT (Ctrl+C) and SIGTERM
        def release_lock_on_signal(signum, frame):
            if _filelock.is_locked:
                _filelock.release()
            sys.exit(0)

        # TODO: Overwrite the default signal handler for SIGINT and SIGTERM is
        # not recommended. We should find a better way to handle this.
        signal.signal(signal.SIGINT, release_lock_on_signal)
        signal.signal(signal.SIGTERM, release_lock_on_signal)

        # Make sure only one training task will acquire the lock
        if rank == 0:
            try:
                _filelock.acquire(timeout=1500)
            except TimeoutError:
                logger.warning(
                    f"Failed to acquire the lock for {lock_dir} within 25 minutes. This may indicate "
                    "that multiple processes are attempting to cache the same JSONL data simultaneously, "
                    "which can cause unexpected errors. If you encounter this message or any unexpected errors, "
                    f"please remove the cache: {os.path.join(lock_dir, '..')}."
                )
                _filelock.release()

        if dist.is_initialized():
            dist.barrier(group=_cache_lock_group)  # type: ignore

        if rank != 0:
            _filelock.acquire()
        return _filelock

    def _cache_release(self, filelock: SoftFileLock):
        filelock.release()
