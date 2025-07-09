# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import json
import math
import multiprocessing
import os
import random
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from io import BytesIO
from multiprocessing import Process, Queue
from threading import Lock
from typing import Callable, cast

import numpy as np
import torch
from mmengine import mkdir_or_exist
from mmengine.dist import barrier, get_rank
from torch import distributed as dist
from tqdm import tqdm

from xtuner.v1.utils import SharedMemory, get_logger

from .utils import CachableTokenizeFunction, CacheObj, calculate_xxhash


logger = get_logger()
_lock = Lock()


CACHE_META = ".xpuyu-cache-meta.json"
XTUNER_FILE_OPEN_CONCURRENCY = int(os.environ.get("XTUNER_FILE_OPEN_CONCURRENCY", "8"))


# TODO: (yehaochen) chunk size and tokenize workers should be parameters of the dataset.
XTUNER_TOKENIZE_CHUNK_SIZE = int(os.environ.get("XTUNER_TOKENIZE_CHUNK_SIZE", "10"))


def _streaming_parallel_open_inplace(path: str, buf, executor: ThreadPoolExecutor):
    file_size = os.path.getsize(path)
    max_workers = min(executor._max_workers, file_size)
    chunk_size = math.ceil(file_size / max_workers)

    def read_chunk(path: str, start: int, size: int) -> None:
        with open(path, "rb") as f:
            f.seek(start)
            f.readinto(buf[start : start + size])

    futures: list[Future[None]] = []

    for i in range(max_workers):
        start = i * chunk_size
        read_size = min(chunk_size, file_size - start)
        futures.append(executor.submit(read_chunk, path, start, read_size))

    for future in futures:
        future.result()


# TODO: We need to refactor this multiprocessing code for general use.
def tokenize_worker(
    tokenize_fun: Callable,
    shm_name: str,
    data_queue: Queue,
    out_queue: Queue,
    cpu_ids: list[int],
):
    os.sched_setaffinity(os.getpid(), cpu_ids)
    shared_memory = SharedMemory(name=shm_name, create=False)
    # f = open("/cpfs01/user/yehaochen/codebase/xpuyu-image/tmp1/part-6853cd7cb0ee-000074.jsonl" , "rb")
    while True:
        data_chunk = data_queue.get()

        if data_chunk is None:
            out_queue.put(None)
            break
        chunk_results = []
        for idx, (start, end) in data_chunk:
            databytes = shared_memory.buf[start:end].tobytes()
            # f.seek(start)
            # databytes = f.read(end - start)
            chunk_results.append([idx, tokenize_fun(databytes)])
        out_queue.put(chunk_results)

    while not out_queue.empty():
        time.sleep(0.01)
    shared_memory.close()


def chunk_data_to_queue(
    data_queue: Queue,
    data: list[tuple[int, tuple[int, int]]],
    chunk_size: int,
    nproc: int,
):
    data_iter = iter(data)
    chunk_data = []
    while True:
        try:
            item = next(data_iter)
        except StopIteration:
            break
        chunk_data.append(item)
        if len(chunk_data) == chunk_size:
            data_queue.put(chunk_data)
            chunk_data = []
    if chunk_data:
        data_queue.put(chunk_data)

    for _ in range(nproc):
        data_queue.put(None)


def _get_local_concurrency():
    """Get the local concurrency level based on the environment variable."""
    if dist.is_initialized():
        local_rank_concurrency = os.getenv("LOCAL_WORLD_SIZE", "1")
    else:
        local_rank_concurrency = 1
    return int(local_rank_concurrency)


# NOTE: The `map` or `submit` function of `concurrent.futures.ProcessPoolExecutor` will cause frequent serialization
# and deserialization of the tokenizer, processing 1000 samples will serialize and deserialize 1000 times, thus
# affecting performance. Here we redefine `parallel_execute` to bind processes with `tokenize_fn`, so the tokenizer
# will only be serialized and deserialized `nproc` times
def parallel_execute(
    tokenize_fn: Callable[[bytes], dict],
    offsets,
    shm_name: str,
    nproc: int,
    chunksize: int,
    rank: int,
):
    cpu_ids = list(os.sched_getaffinity(0))
    local_rank_concurrency = _get_local_concurrency()
    local_cpu_ids = cpu_ids[rank::local_rank_concurrency]

    processes: list[Process] = []
    data_queue: Queue[tuple[int, tuple[int, int]]] = Queue()  # idx, (start, end)
    output_queue: Queue[dict] = Queue()  # {"num_tokens": int}
    # task_id = bar.add_task(total=task_num, description=description)
    chunk_data_to_queue(data_queue, offsets, chunksize, nproc)

    cpus_per_process = min(len(local_cpu_ids) // nproc, 1)
    ctx = multiprocessing.get_context("fork")
    for idx in range(nproc):
        bind_cpu_ids = local_cpu_ids[idx * cpus_per_process : (idx + 1) * cpus_per_process]
        process = ctx.Process(
            target=tokenize_worker,
            args=(tokenize_fn, shm_name, data_queue, output_queue, bind_cpu_ids),
        )
        process.start()
        # ForkedProcess instances should be acceptable for a list of type Process
        processes.append(process)  # type: ignore

    results: list[dict] = []
    finished_process = 0
    while finished_process < nproc:
        chunk_results = output_queue.get()
        if chunk_results is None:
            finished_process += 1
            continue
        results.extend(chunk_results)
    results = [x[1] for x in sorted(results, key=lambda x: x[0])]
    return results


class JsonlDataset(torch.utils.data.Dataset):
    _process_group: dist.ProcessGroup | None = None
    _thread_executor: ThreadPoolExecutor | None = None
    _shared_memory: SharedMemory | None = None

    def __init__(
        self,
        path,
        sample_ratio: float = 1.0,
        tokenize_fn: CachableTokenizeFunction | None = None,
        cache_dir: str | None = None,
        max_length: int | None = None,
        cache_tag: str | None = None,
    ):
        super().__init__()

        self.tokenize_fn = tokenize_fn
        self.path = path
        self.tokenizer_workers = int(os.environ.get("XTUNER_TOKENIZE_WORKERS", 8))
        self._shared_memory = self._init_shared_memory(path)
        self.meta_path = os.path.join(cache_dir, CACHE_META) if cache_dir else None

        logger.info(f"Start loading {self.path} with sample_ratio={sample_ratio}.")

        if cache_tag is not None and (cached := self._get_cached_tag(cache_tag, tokenize_fn)) is not None:
            offset_path, num_tokens_path = cached["offsets"], cached["num_tokens"]
            offsets = np.load(offset_path)
            num_tokens = np.load(num_tokens_path)
        elif cache_dir:
            assert self.meta_path is not None

            if get_rank() == 0:
                if os.path.exists(cache_dir):
                    assert os.path.isdir(cache_dir)
                else:
                    mkdir_or_exist(cache_dir)
            barrier()

            file_hash = calculate_xxhash(self._shared_memory.buf)
            # file_hash = calculate_bytes_sha256(self._shared_memory.buf)
            file_cache_dir = os.path.join(cache_dir, file_hash)

            if file_hash not in os.listdir(cache_dir):
                if get_rank() == 0:
                    mkdir_or_exist(file_cache_dir)
            barrier()

            _cached_file = os.path.join(file_cache_dir, "offsets.npy")
            if get_rank() == 0:
                if not os.path.exists(_cached_file):
                    offsets = self.count_offsets(file_cache_dir)

                if not os.path.exists(self.meta_path):
                    with open(self.meta_path, "w") as f:
                        f.write("{}")

                # The structure of meta file
                # Example:
                # {
                #     "<file hash>": {
                #         "offsets": [
                #             "<file cache path>"
                #         ],
                #         "num_tokens": {
                #             "<tokenize hash>": [
                #                 <tokenize cache path>
                #             ]
                #         }
                #     },
                #     "tags": {
                #         "<tag name>": {
                #             "<file path>": {
                #                 "<tokenize hash>": {
                #                     "num_tokens": "<tokenize cache path>",
                #                     "offsets": "<file cache path>",
                #                     "datetime": "2025-06-30 09:40:24"
                #                 }
                #             }
                #         }
                #     }
                # }

                with open(self.meta_path, "r+") as f:
                    origin_data = json.load(f)
                    if file_hash not in origin_data:
                        origin_data[file_hash] = {"offsets": [self.path]}
                    else:
                        if self.path not in origin_data[file_hash]["offsets"]:
                            origin_data[file_hash]["offsets"].append(self.path)
                    f.seek(0)
                    f.truncate(0)
                    f.write(json.dumps(origin_data, indent=4, ensure_ascii=False))

            barrier()

            offsets = np.load(_cached_file)

            if tokenize_fn and isinstance(tokenize_fn, CachableTokenizeFunction):
                tok_hash = tokenize_fn.hash()
                tok_cache_dir = os.path.join(file_cache_dir, tok_hash)
                if tok_hash not in os.listdir(file_cache_dir):
                    if get_rank() == 0:
                        mkdir_or_exist(tok_cache_dir)
                barrier()

                if "num_tokens.npy" in os.listdir(tok_cache_dir):
                    _cached_file = os.path.join(tok_cache_dir, "num_tokens.npy")
                    logger.info(f"Loading `num_tokens` from cache: {_cached_file}")
                    num_tokens = np.load(_cached_file)
                else:
                    num_tokens = self.count_tokens(offsets, tok_cache_dir)

                if get_rank() == 0:
                    with open(self.meta_path, "r+") as f:
                        origin_data = json.load(f)
                        data = origin_data[file_hash]
                        if "num_tokens" not in data:
                            data["num_tokens"] = {}
                        if tok_hash not in data["num_tokens"]:
                            data["num_tokens"][tok_hash] = [self.path]
                        else:
                            if self.path not in data["num_tokens"][tok_hash]:
                                data["num_tokens"][tok_hash].append(self.path)

                        if cache_tag is not None:
                            if "tags" not in origin_data:
                                origin_data["tags"] = {}

                            tag_data: dict = defaultdict(lambda: defaultdict(dict))

                            if cache_tag in origin_data["tags"]:
                                tag_data.update(origin_data["tags"][cache_tag])

                            origin_data["tags"][cache_tag] = tag_data

                            if tok_hash not in tag_data[self.path]:
                                tag_data[self.path][tok_hash] = {
                                    "num_tokens": os.path.join(tok_cache_dir, "num_tokens.npy"),
                                    "offsets": os.path.join(file_cache_dir, "offsets.npy"),
                                    "datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                }
                        f.seek(0)
                        f.truncate(0)
                        f.write(json.dumps(origin_data, indent=4, ensure_ascii=False))

                barrier()

            elif tokenize_fn:
                logger.warning(
                    f"{tokenize_fn.__class__.__name__} is not an instance of "
                    "`CachableTokenizeFunction`, data will always "
                    "be re-tokenized during training!"
                )
                num_tokens = self.count_tokens(offsets)
            else:
                num_tokens = None

                offsets = offsets
                num_tokens = num_tokens
        else:
            offsets = self.count_offsets()
            num_tokens = None
            if tokenize_fn is not None:
                tokenize_fn.set_state("cache")
                num_tokens = self.count_tokens(offsets)
                tokenize_fn.set_state("runtime")

        # offset starts from 0 and endwith `file_size`
        # The size of offsets is `num_samples + 1`
        _sampled = list(range(len(offsets) - 1))
        # Filter out samples with num_tokens=0, 0 means the sample is damaged
        if num_tokens is not None:
            orig_sample_num = len(num_tokens)
            _sampled = [i for i in _sampled if num_tokens[i] != 0]
            if len(_sampled) < orig_sample_num:
                logger.warning(f"{path} has {orig_sample_num - len(_sampled)} damaged samples, discard.")

        if num_tokens is not None and max_length is not None:
            assert isinstance(max_length, int)
            _filtered = [i for i in _sampled if num_tokens[i] <= max_length]

            if len(_filtered) < len(_sampled):
                missed_num = len(_sampled) - len(_filtered)
                logger.warning(f"{path} has {missed_num} prompt length>{max_length}, discard.")

            _sampled = _filtered

        _target_num_samples = int(len(_sampled) * sample_ratio)
        self.sampled = _sampled * int(sample_ratio)
        self.sampled.extend(random.sample(_sampled, _target_num_samples - len(self.sampled)))

        if num_tokens is not None:
            num_tokens = num_tokens[self.sampled]

        self.num_tokens = num_tokens
        self.offsets = offsets[self.sampled]
        self._release_shared_memory()

    def _init_shared_memory(self, path: str) -> SharedMemory:
        if dist.is_initialized():
            rank = dist.get_rank()
            output: list[None | str] = [None] * dist.get_world_size()
            local_concurrency = _get_local_concurrency()
            # Asumming that each node has the same rank
            # This allgather eunsure that each node rank share the same shared memory.
            # For example:
            # rank[0-7] use rank0 shared memory
            # rank[8-15] use rank8 shared memory
            # ...
            if rank % local_concurrency == 0:
                shared_memory = SharedMemory(
                    size=os.path.getsize(path),
                    create=True,
                )
                _streaming_parallel_open_inplace(path, shared_memory.buf, self.executor)
                name = shared_memory.name
                dist.all_gather_object(output, name, group=self.process_group)
                # dist.broadcast_object_list(name, src=0, group=self.process_group)
            else:
                dist.all_gather_object(output, [None], group=self.process_group)
                # dist.broadcast_object_list(name, src=0, group=self.process_group)
                local_master_rank = rank // local_concurrency * local_concurrency
                shm_name = output[local_master_rank]
                assert isinstance(shm_name, str)
                shared_memory = SharedMemory(
                    name=shm_name,
                    create=False,
                )
        else:
            shared_memory = SharedMemory(
                create=True,
                size=os.path.getsize(path),
            )
            _streaming_parallel_open_inplace(path, shared_memory.buf, self.executor)
        return shared_memory

    def count_offsets(self, cache_dir=None):
        offsets = [0]
        assert self._shared_memory is not None, "Shared memory is not initialized. Call `_init_shared_memory` first."

        with BytesIO(self._shared_memory.buf) as f:
            for line in f:
                offsets.append(offsets[-1] + len(line))

        offsets = np.array(offsets)

        if cache_dir and get_rank() == 0:
            save_path = os.path.join(cache_dir, "offsets.npy")
            np.save(save_path, offsets)

        return offsets

    @staticmethod
    def _tokenize_by_offset(
        data: bytes,
        tokenize_fn: Callable[[dict], CacheObj],
    ) -> dict:
        line = data.decode()
        tokenized = tokenize_fn(json.loads(line))
        return {"num_tokens": tokenized["num_tokens"]}

    def count_tokens(self, offsets, cache_dir=None):
        num_samples = len(offsets)

        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0

        num_per_rank = math.ceil(num_samples / world_size)

        starts = offsets[:-1]
        ends = offsets[1:]

        range_list = list(enumerate(zip(starts, ends)))
        start = rank * num_per_rank
        end = (rank + 1) * num_per_rank
        range_list_shard = range_list[start:end]

        desc = f"[Rank {rank}] {self.path}"

        assert self._shared_memory is not None and self._shared_memory.name is not None
        shm_name = self._shared_memory.name
        worker = partial(JsonlDataset._tokenize_by_offset, tokenize_fn=self.tokenize_fn)

        if self.tokenizer_workers > 1:
            chunked_size = XTUNER_TOKENIZE_CHUNK_SIZE
            # chunked_range = list(batched(range_list_shard, math.ceil(len(range_list_shard) / self.tokenizer_workers)))
            tokenized = parallel_execute(
                tokenize_fn=worker,
                offsets=range_list_shard,
                shm_name=shm_name,
                nproc=self.tokenizer_workers,
                chunksize=chunked_size,
                rank=rank % _get_local_concurrency(),
            )
        else:
            tokenized = []
            for start, end in tqdm(range_list_shard, desc=desc, smoothing=0.001):
                tokenized.append(worker(bytes(self._shared_memory.buf[start:end])))

        _num_tokens = [data["num_tokens"] for data in tokenized]
        _num_tokens = np.array(_num_tokens)

        if dist.is_initialized():
            # TODO:
            # This is a workaround for `all_gather_object` would hang when
            # using `nccl` backend. Maybe we could find a better way to
            # synchronize the `num_tokens` since datasets are not always initialized
            # with the world size.
            num_tokens = [None] * world_size
            dist.all_gather_object(num_tokens, _num_tokens, group=self.process_group)
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

    @property
    def process_group(self) -> dist.ProcessGroup:
        cls = self.__class__
        with _lock:
            if cls._process_group is None:
                if "gloo" in (backend := dist.get_backend()):
                    group = dist.new_group(backend=backend, timeout=datetime.timedelta(seconds=1800))
                else:
                    group = dist.new_group(timeout=datetime.timedelta(seconds=1800))
                cls._process_group = group
        return cast(dist.ProcessGroup, cls._process_group)

    @property
    def executor(self) -> ThreadPoolExecutor:
        cls = self.__class__
        with _lock:
            if cls._thread_executor is None:
                cls._thread_executor = ThreadPoolExecutor(max_workers=XTUNER_FILE_OPEN_CONCURRENCY)
        return cast(ThreadPoolExecutor, cls._thread_executor)

    def _release_shared_memory(self):
        """Release shared memory if it exists."""
        if dist.is_initialized():
            dist.barrier()

            if self._shared_memory is not None:
                self._shared_memory.close()
        else:
            if self._shared_memory is not None:
                self._shared_memory.close()

        if self._shared_memory is not None:
            local_rank_concurrency = _get_local_concurrency()
            if not dist.is_initialized() or dist.get_rank() % local_rank_concurrency == 0:
                self._shared_memory.unlink()
            self._shared_memory = None

    def _get_cached_tag(self, tag: str, tokenizer_fn: CachableTokenizeFunction | Callable | None) -> dict | None:
        """Check if the dataset is cached with the given tag."""
        if not isinstance(tokenizer_fn, CachableTokenizeFunction):
            return None

        if self.meta_path is None or not os.path.exists(self.meta_path):
            return None

        # TODO: (yehaochen) This is very suck. I do not know why sometimes the meta file could be
        # a broken json. The barrier has been put at the right place, but it still happens.
        # I believe this is cause by the bug of filesystem (maybe), hope someone could help to fix it.
        for _ in range(10):
            try:
                with open(self.meta_path) as f:
                    content = f.read()
                    meta = json.loads(content)
            except json.JSONDecodeError:
                time.sleep(0.01)
                continue
            else:
                break
        else:
            raise json.JSONDecodeError("Failed to decode JSON file after 10 attempts.", doc=content, pos=0)

        if dist.is_initialized():
            dist.barrier()

        tok_hash = tokenizer_fn.hash()

        return meta.get("tags", {}).get(tag, {}).get(self.path, {}).get(tok_hash)
