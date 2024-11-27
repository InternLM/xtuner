import hashlib
import inspect
import json
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor
from mmengine import mkdir_or_exist
import numpy as np
import torch
from torch import distributed as dist
from tqdm import tqdm


def calculate_json_sha256(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()

    hash_object = hashlib.sha256(data)
    hash_hex = hash_object.hexdigest()
    return hash_hex


def calculate_tokenize_fn_sha256(tokenize_fn):
    """Calculate SHA-256 hash for an instance method's source code."""
    # Get the source code of the method
    fn_source = inspect.getsource(tokenize_fn.__call__)
    return hashlib.sha256(fn_source.encode('utf-8')).hexdigest()


class JsonDataset(torch.utils.data.Dataset):

    def __init__(self,
                 path,
                 sample_ratio=1.0,
                 tokenize_fn=None,
                 cache_dir=None):
        super().__init__()

        assert sample_ratio <= 1
        self.tokenize_fn = tokenize_fn
        self.path = path
        self.tokenizer_workers = int(os.environ.get('XTUNER_TOKENIZE_WORKERS', 8))

        if cache_dir:
            if os.path.exists(cache_dir):
                assert os.path.isdir(cache_dir)
            else:
                mkdir_or_exist(cache_dir)

            file_hash = calculate_json_sha256(path)
            file_cache_dir = os.path.join(cache_dir, file_hash)

            if file_hash not in os.listdir(cache_dir):
                mkdir_or_exist(file_cache_dir)

            if self.tokenize_fn:
                tok_hash = calculate_tokenize_fn_sha256(tokenize_fn)
                tok_cache_dir = os.path.join(file_cache_dir, tok_hash)
                if tok_hash not in os.listdir(file_cache_dir):
                    mkdir_or_exist(tok_cache_dir)

                if 'num_tokens.npy' in os.listdir(tok_cache_dir):
                    _cached_file = os.path.join(tok_cache_dir,
                                                'num_tokens.npy')
                    num_tokens = np.load(_cached_file)
                else:
                    num_tokens = self.count_tokens(tok_cache_dir)
            else:
                num_tokens = None

        else:
            num_tokens = None

        with open(self.path) as f:
            dataset = json.load(f)

        num_samples = int(len(dataset) * sample_ratio)
        sampled = random.sample([i for i in range(len(dataset))], num_samples)
        self.sampled = sampled

        if num_tokens is not None:
            num_tokens = num_tokens[sampled]

        self.num_tokens = num_tokens

        self.dataset = None

    def count_tokens(self, cache_dir=None):

        dataset = []

        with open(self.path) as f:
            dataset = json.load(f)

        num_samples = len(dataset)

        if dist.is_available():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0

        num_per_rank = math.ceil(num_samples / world_size)

        start = rank * num_per_rank
        end = (rank + 1) * num_per_rank
        dataset_shard = dataset[start:end]

        desc = f'[Rank {rank}] {self.path}'
        with ProcessPoolExecutor(max_workers=self.tokenizer_workers) as executor:
            tokenized = list(
                tqdm(
                    executor.map(self.tokenize_fn, dataset_shard,
                                 chunksize=max(1, len(dataset_shard) // self.tokenizer_workers)),
                    desc=desc,
                    total=len(dataset_shard)))

        _num_tokens = [data['num_tokens'] for data in tokenized]
        _num_tokens = np.array(_num_tokens)

        if dist.is_available():
            num_tokens = [None] * world_size
            dist.all_gather_object(num_tokens, _num_tokens)
            num_tokens = np.concatenate(num_tokens, axis=0)
        else:
            num_tokens = _num_tokens

        if rank == 0 and cache_dir:
            save_path = os.path.join(cache_dir, 'num_tokens.npy')
            np.save(save_path, num_tokens)

        return num_tokens

    def __len__(self):
        return len(self.sampled)

    def __getitem__(self, item):
        """Returns a dict containing packed data in the given item.

        Args:
            item: An index to retrieve packed data.

        Returns:
            A dict including packed input_ids, labels, and cumulative_len.
        """
        if self.dataset is None:
            with open(self.path) as f:
                self.dataset = json.load(f)

        raw_data = self.dataset[self.sampled[item]]

        if self.tokenize_fn:
            tokenized_data = self.tokenize_fn(raw_data)
            return tokenized_data
        else:
            return raw_data
