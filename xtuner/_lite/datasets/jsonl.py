# Copyright (c) OpenMMLab. All rights reserved.
import hashlib
import json
import math
import multiprocessing
import os
import random
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, TypedDict

import numpy as np
import torch
from mmengine import mkdir_or_exist
from torch import distributed as dist
from tqdm import tqdm

from xtuner._lite import get_logger

logger = get_logger()


def calculate_jsonl_sha256(path):
    with open(path, "rb") as f:
        file_hash = hashlib.sha256()
        file_hash.update(f.read())
    return file_hash.hexdigest()


CacheObj = TypedDict("CachedObj", {"num_tokens": int}, total=False)


class CachableTokenizeFunction(ABC):
    @abstractmethod
    def __call__(self, item: Any) -> CacheObj:
        raise NotImplementedError

    @abstractmethod
    def hash(self) -> str:
        raise NotImplementedError


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

        if cache_dir and isinstance(tokenize_fn, CachableTokenizeFunction):
            if os.path.exists(cache_dir):
                assert os.path.isdir(cache_dir)
            else:
                mkdir_or_exist(cache_dir)

            file_hash = calculate_jsonl_sha256(path)
            file_cache_dir = os.path.join(cache_dir, file_hash)

            if file_hash not in os.listdir(cache_dir):
                mkdir_or_exist(file_cache_dir)

            if "offsets.npy" in os.listdir(file_cache_dir):
                _cached_file = os.path.join(file_cache_dir, "offsets.npy")
                offsets = np.load(_cached_file)
            else:
                offsets = self.count_offsets(file_cache_dir)

            if self.tokenize_fn:
                tok_hash = tokenize_fn.hash()
                tok_cache_dir = os.path.join(file_cache_dir, tok_hash)
                if tok_hash not in os.listdir(file_cache_dir):
                    mkdir_or_exist(tok_cache_dir)

                if "num_tokens.npy" in os.listdir(tok_cache_dir):
                    _cached_file = os.path.join(tok_cache_dir, "num_tokens.npy")
                    num_tokens = np.load(_cached_file)
                else:
                    num_tokens = self.count_tokens(offsets, tok_cache_dir)
            else:
                num_tokens = None

            offsets = offsets
            num_tokens = num_tokens

        else:
            offsets = self.count_offsets()
            num_tokens = None
            if max_length is not None:
                assert self.tokenize_fn
                num_tokens = self.count_tokens(offsets)

        _sampled = [i for i in range(len(offsets))]

        if max_length is not None:
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

        if dist.get_rank() == 0 and cache_dir:
            save_path = os.path.join(cache_dir, "offsets.npy")
            np.save(save_path, offsets)

        return offsets

    def _tokenize_by_offset(self, offset):
        with open(self.path) as f:
            f.seek(offset)
            data = json.loads(f.readline())
        return self.tokenize_fn(data)

    def count_tokens(self, offsets, cache_dir=None):
        num_samples = len(offsets)

        if dist.is_available():
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
                        self._tokenize_by_offset, offsets_shard, chunksize=chunk_size
                    ),
                    desc=desc,
                    total=len(offsets_shard),
                )
            )

        _num_tokens = [data["num_tokens"] for data in tokenized]
        _num_tokens = np.array(_num_tokens)

        if dist.is_available():
            num_tokens = [None] * world_size
            dist.all_gather_object(num_tokens, _num_tokens)
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
