# Copyright (c) OpenMMLab. All rights reserved.
import atexit
import datetime
import hashlib
import itertools
import json
import math
import multiprocessing
import os
import random
import shutil
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from io import BytesIO
from multiprocessing import Process, Queue
from pathlib import Path
from threading import Lock
from typing import Callable, Dict, TypeVar, cast

import numpy as np
import torch
from mmengine import mkdir_or_exist
from mmengine.dist import barrier, get_rank
from torch import distributed as dist
from tqdm import tqdm

from xtuner.v1.datasets.data_item import CacheItem
from xtuner.v1.datasets.pt_tokenize_fn.long_text import LongTextPretrainTokenizeFunction
from xtuner.v1.datasets.rl_tokenize_fn.rl_tokenize_fn import RLTokenizeFn
from xtuner.v1.utils import SharedMemory, get_logger
from xtuner.v1.utils.dist_utils import get_local_process_group, get_local_world_size, is_local_rank0

from .utils import CachableTokenizeFunction, CacheDict, CacheObj, calculate_xxhash


T = TypeVar("T")
logger = get_logger()
_lock = Lock()

CACHE_META = ".xpuyu-cache-meta.json"
XTUNER_FILE_OPEN_CONCURRENCY = int(os.environ.get("XTUNER_FILE_OPEN_CONCURRENCY", "8"))

XTUNER_TOKENIZE_CHUNK_SIZE = int(os.environ.get("XTUNER_TOKENIZE_CHUNK_SIZE", "10"))


def _concat_values(values):
    if isinstance(values[0], np.ndarray):
        return np.concatenate(values, axis=0)
    return list(itertools.chain.from_iterable(values))


def save_dict_to_npy_dir(data: Dict[str, np.ndarray], dir_path: str) -> None:
    """将 dict 以每 key 一个 .npy 文件的形式保存到目录."""
    os.makedirs(dir_path, exist_ok=True)
    for k, v in data.items():
        if not isinstance(k, str):
            raise TypeError(f"key must be str, got {type(k)}")
        if not isinstance(v, np.ndarray):
            raise TypeError(f"value for key '{k}' must be np.ndarray, got {type(v)}")
        np.save(os.path.join(dir_path, f"{k}.npy"), v)


def load_dict_from_npy_dir(dir_path: str, mmap: bool = True) -> Dict[str, np.ndarray]:
    """从 npy 目录按 key 加载 _meta；object 数组完整加载，其余 mmap_mode='r'."""
    if not os.path.exists(dir_path):
        return {}
    result = {}
    for fname in os.listdir(dir_path):
        if not fname.endswith(".npy"):
            continue
        key = fname[:-4]
        fpath = os.path.join(dir_path, fname)
        arr = np.load(fpath, mmap_mode="r" if mmap else None)
        result[key] = arr
    return result


def _filter_sampled_indices(
    sampled: np.ndarray,
    num_tokens: np.ndarray | None,
    max_length: int | None,
) -> np.ndarray:
    # Filter out samples with num_tokens=0, 0 means the sample is damaged
    if num_tokens is not None:
        assert isinstance(num_tokens, np.ndarray)
        orig_sample_num = len(num_tokens)
        sampled = sampled[num_tokens[sampled] != 0]
        if len(sampled) < orig_sample_num:
            missed = orig_sample_num - len(sampled)
            logger.warning(f"filtered {missed} damaged samples (num_tokens==0).")

    if num_tokens is not None and max_length is not None:
        assert isinstance(max_length, int)
        before = len(sampled)
        sampled = sampled[num_tokens[sampled] <= max_length]
        if len(sampled) < before:
            logger.warning(f"filtered {before - len(sampled)} samples with length>{max_length}.")

    return sampled


def _apply_sample_ratio(
    sampled: np.ndarray,
    *,
    sample_ratio: float,
    enable_sequential_sampler: bool,
) -> np.ndarray:
    target = int(len(sampled) * sample_ratio)
    if target <= 0:
        return sampled[:0]

    base_repeats = int(sample_ratio)
    repeated = np.tile(sampled, base_repeats) if base_repeats > 0 else sampled[:0]
    remaining = target - len(repeated)
    if remaining <= 0:
        return repeated[:target]

    if enable_sequential_sampler:
        extra = sampled[:remaining]
    else:
        # Keep the same no-replacement behavior as `random.sample`,
        # but avoid converting the whole numpy array to Python list.
        choice_idxs = random.sample(range(len(sampled)), remaining)
        extra = sampled[np.asarray(choice_idxs, dtype=np.int64)]

    return np.concatenate([repeated, extra], axis=0)


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
    try:
        os.sched_setaffinity(os.getpid(), cpu_ids)
    except OSError as e:
        logger.debug(f"Failed to set CPU affinity: {e}")

    shared_memory = SharedMemory(name=shm_name, create=False)
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
    local_rank_concurrency = get_local_world_size()
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


class JsonlDataset(torch.utils.data.Dataset[T | CacheItem]):
    _process_group: dist.ProcessGroup | None = None
    _thread_executor: ThreadPoolExecutor | None = None
    # TODO: Using shared memory should be optional since the size of `/dev/shm` could be not enough for some devices
    _shared_memory: SharedMemory | None = None
    _meta: dict[str, np.ndarray]

    def __init__(
        self,
        anno_path,
        sample_ratio: float = 1.0,
        tokenize_fn: CachableTokenizeFunction[T] | None = None,
        name: str = "default",
        cache_dir: str | Path | None = None,
        max_length: int | None = None,  # TODO: Remove max_length in dataset
        cache_tag: str | None = None,
        enable_sequential_sampler: bool = False,
        enable_mmap_shared: bool = False,
        disable_filter: bool = False,
    ):
        super().__init__()

        self.disable_filter = disable_filter
        self.tokenize_fn = tokenize_fn
        self.path = str(anno_path)
        self.name = name
        self._shared_memory = None

        self.tokenizer_workers = int(os.environ.get("XTUNER_TOKENIZE_WORKERS", 8))
        self.meta_path = os.path.join(cache_dir, CACHE_META) if cache_dir else None

        logger.info(f"[Dataset] Start loading [{self.name}]{self.path} with sample_ratio={sample_ratio}.")

        self._has_chunk = isinstance(tokenize_fn, LongTextPretrainTokenizeFunction)

        tok_cache_dir: str | None = None  # set inside cache_dir branch when tokenize_fn is CachableTokenizeFunction
        if cache_tag is not None and (cached := self._get_cached_tag(cache_tag, tokenize_fn)) is not None:
            logger.info(f"[Dataset] Load cached [{self.name}]{self.path} of cache tgs {cache_tag}.")
            offset_path = cached["offsets"]
            meta_path = cached.get("jsonl_meta")
            offsets = np.load(offset_path, mmap_mode="r" if enable_mmap_shared else None)
            if meta_path:
                _meta = load_dict_from_npy_dir(meta_path, mmap=enable_mmap_shared)
        elif cache_dir:
            self._shared_memory = self._init_shared_memory(anno_path)
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
                #             "<original file path>"  # for mapping to <file hash>
                #         ],
                #         "jsonl_meta": {
                #             "<tokenize hash>": [
                #                 <original file path>
                #             ]
                #         }
                #     },
                #     "tags": {
                #         "<tag name>": {
                #             "<file path>": {
                #                 "<tokenize hash>": {
                #                     "jsonl_meta": "<tokenize cache path>/jsonl_meta/",
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

            offsets = np.load(_cached_file, mmap_mode="r" if enable_mmap_shared else None)

            if tokenize_fn and isinstance(tokenize_fn, CachableTokenizeFunction):
                tok_hash = tokenize_fn.hash()
                tok_cache_dir = os.path.join(file_cache_dir, tok_hash)
                if tok_hash not in os.listdir(file_cache_dir):
                    if get_rank() == 0:
                        mkdir_or_exist(tok_cache_dir)
                barrier()

                _meta_file = os.path.join(tok_cache_dir, "jsonl_meta")
                if os.path.exists(_meta_file):
                    logger.info(f"Loading tokenize meta from cache: {_meta_file}")
                    _meta = load_dict_from_npy_dir(_meta_file, mmap=enable_mmap_shared)
                else:
                    _meta = self.count_tokens(offsets, tok_cache_dir)

                if get_rank() == 0:
                    with open(self.meta_path, "r+") as f:
                        origin_data = json.load(f)
                        data = origin_data[file_hash]
                        if "jsonl_meta" not in data:
                            data["jsonl_meta"] = {}
                        if tok_hash not in data["jsonl_meta"]:
                            data["jsonl_meta"][tok_hash] = [self.path]
                        else:
                            if self.path not in data["jsonl_meta"][tok_hash]:
                                data["jsonl_meta"][tok_hash].append(self.path)

                        if cache_tag is not None:
                            if "tags" not in origin_data:
                                origin_data["tags"] = {}

                            tag_data: dict = defaultdict(lambda: defaultdict(dict))

                            if cache_tag in origin_data["tags"]:
                                tag_data.update(origin_data["tags"][cache_tag])

                            origin_data["tags"][cache_tag] = tag_data

                            if tok_hash not in tag_data[self.path]:
                                tag_data[self.path][tok_hash] = {
                                    "jsonl_meta": os.path.join(tok_cache_dir, "jsonl_meta"),
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
                _meta = self.count_tokens(offsets)
            else:
                offsets = offsets
                _meta = {}
        else:
            self._shared_memory = self._init_shared_memory(anno_path)
            offsets = self.count_offsets()
            _meta = {}
            if tokenize_fn is not None:
                _meta = self.count_tokens(offsets)

        _meta["offsets"] = offsets
        if _meta["num_tokens"] is None:
            _meta.pop("num_tokens")

        ################################## Post-processing of offsets, num_tokens and _meta #######################################

        tok_hash_str = ""
        if isinstance(
            tokenize_fn, RLTokenizeFn
        ):  # RLTokenizeFn is CachableTokenizeFunction, but it does not have a hash method
            tok_hash_str = "RLTokenizeFn"
        elif isinstance(tokenize_fn, CachableTokenizeFunction):
            tok_hash_str = tokenize_fn.hash()

        job_discriminator = os.environ.get("MASTER_PORT", "")
        tmp_dir = os.path.join(
            "/tmp",
            hashlib.md5(
                f"jsonl_mmap_{self.path}_{sample_ratio}_{max_length}_{tok_hash_str}_{job_discriminator}".encode()
            ).hexdigest(),
        )

        if enable_mmap_shared and dist.is_initialized() and get_local_world_size() > 1:
            _meta_need_update = {}
            if is_local_rank0():
                _meta_need_update = self._get_meta_need_update(
                    _meta,
                    sample_ratio=sample_ratio,
                    max_length=max_length,
                    enable_sequential_sampler=enable_sequential_sampler,
                )
                save_dict_to_npy_dir(_meta_need_update, tmp_dir)
                atexit.register(shutil.rmtree, tmp_dir, True)

            dist.barrier(group=get_local_process_group())
            _meta_need_update = load_dict_from_npy_dir(tmp_dir, mmap=True)
        else:
            _meta_need_update = self._get_meta_need_update(
                _meta,
                sample_ratio=sample_ratio,
                max_length=max_length,
                enable_sequential_sampler=enable_sequential_sampler,
            )

        _meta.update(_meta_need_update)
        self._meta = _meta

        if self._shared_memory is not None:
            self._release_shared_memory()

    def _get_meta_need_update(
        self,
        _meta: dict[str, np.ndarray],
        *,
        sample_ratio: float,
        max_length: int | None,
        enable_sequential_sampler: bool,
    ) -> dict[str, np.ndarray]:
        """对 _meta 做过滤、采样，返回 _meta 中需要更新的 key-value (即 _meta_need_update)。

        如果使用 LongTextPretrainTokenizeFunction (即 self._has_chunk=True)，还需要将offsets更新为 chunk 对齐后的 offsets。

        需要更新的 _meta_need_update, 在不使用 mmap (enable_mmap_shared=False) 时，后续在各rank会更新_meta。
        而在使用 mmap 时，local rank0会将 _meta_need_update 保存到 tmp 目录，然后所有rank将通过mmap共享物理页加载。

        在使用 mmap 时，具体有3种情况:
        1. disable_filter and sample_ratio == 1.0 and not self._has_chunk:
             所有rank直接使用_meta，因为_meta已经是mmap模式。这是高速通路，保持了meta数据懒加载
        2. disable_filter and sample_ratio == 1.0 and self._has_chunk:
            所有rank直接使用_meta，除了_meta["offsets"]。因为 offsets 在 LongTextPretrainTokenizeFunction 时会做扩增。
            Local rank0将offsets保存到tmp，然后所有rank将通过mmap共享物理页加载。
        3. 其他情况:
           只有local rank0过滤和采样样本，然后保存到tmp；所有rank将通过mmap共享物理页加载。
           这时 offsets.npy 和 jsonl_meta 虽然是懒加载，但是在过滤和采样时都会被加载到内存。是慢速通路。

        Returns:
            无过滤且 `sample_ratio==1` 时，
              1) _has_chunk=False 为 `{}`，
              2) _has_chunk=True 为 `{"offsets": ...}`，
            3) 否则为与 `_meta` 相同的完整字典。
        """
        _meta_need_update = {}
        if self._has_chunk:
            line_idxs = _meta.pop("line_idxs")
            _meta["offsets"] = _meta["offsets"][line_idxs]
            # After line_idxs indexing, offsets has exactly num_chunks elements
            # (no trailing sentinel), so use len(offsets) directly.
            base_len = len(_meta["offsets"])
            _meta_need_update["offsets"] = _meta["offsets"]
        else:
            # offsets has trailing sentinel (file_size), so samples are num_offsets - 1
            _meta["offsets"] = _meta["offsets"][:-1]
            # [:-1] 是 基本切片（basic slicing），得到的是 视图（view），与原来的 _meta["offsets"] 共享同一段底层缓冲区。
            # 若原数组是 np.load(..., mmap_mode="r") 得到的 memmap，这段缓冲区就是文件 mmap 出来的；
            # 切片后的数组只是换了 shape/偏移，仍然通过视图链指向同一块 mmap 内存。
            base_len = len(_meta["offsets"])

        if self.disable_filter and sample_ratio == 1.0:
            # self._meta = _meta
            return _meta_need_update

        dtype = np.int32 if base_len < np.iinfo(np.int32).max else np.int64
        _sampled = np.arange(base_len, dtype=dtype)

        if not self.disable_filter:
            _sampled = _filter_sampled_indices(_sampled, _meta.get("num_tokens"), max_length)

        if sample_ratio != 1.0:
            _sampled = _apply_sample_ratio(
                _sampled,
                sample_ratio=sample_ratio,
                enable_sequential_sampler=enable_sequential_sampler,
            )

        for _, v in _meta.items():
            assert base_len == len(v)
        _meta_need_update = {}
        for k, v in _meta.items():
            assert isinstance(v, np.ndarray)
            _meta_need_update[k] = v[_sampled]

        return _meta_need_update

    @property
    def offsets(self) -> np.ndarray:
        return self._meta["offsets"]

    @property
    def num_tokens(self) -> np.ndarray | None:
        return self._meta.get("num_tokens")

    @property
    def proxy_attn_flops(self) -> np.ndarray:
        return self._meta["proxy_attn_flops"]

    def _init_shared_memory(self, path: str) -> SharedMemory:
        if dist.is_initialized():
            rank = dist.get_rank()
            output: list[None | str] = [None] * dist.get_world_size()
            local_concurrency = get_local_world_size()
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
        tokenize_fn: Callable[[dict], CacheDict | CacheObj],
    ) -> dict:
        line = data.decode()
        tokenized = tokenize_fn(json.loads(line))
        if isinstance(tokenized, CacheObj):
            num_tokens = tokenized.num_tokens
        else:
            num_tokens = tokenized["num_tokens"]
        return {"num_tokens": num_tokens}

    def count_tokens(self, offsets, cache_dir=None):
        self.tokenize_fn.set_state("cache")

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
                rank=rank % get_local_world_size(),
            )
        else:
            tokenized = []
            for _, (start, end) in tqdm(range_list_shard, desc=desc, smoothing=0.001):
                tokenized.append(worker(bytes(self._shared_memory.buf[start:end])))

        # serialize tokenized
        if self._has_chunk:
            num_tokens_of_chunks = []
            proxy_attn_flops_list = []
            chunks_of_chunks = []
            line_idxs_of_chunks = []
            for line_idx, data in enumerate(tokenized):
                num_tokens_of_chunks.extend(data["num_tokens"])
                proxy_attn_flops_list.extend(data["proxy_attn_flops"])
                chunks_of_chunks.extend(
                    [(c["char_start"], c["char_end"], c["token_start_offset"]) for c in data["chunks"]]
                )
                line_idxs_of_chunks.extend([line_idx] * len(data["num_tokens"]))
            serialized_tokenized = {
                "num_tokens": np.array(num_tokens_of_chunks),
                "proxy_attn_flops": np.array(proxy_attn_flops_list),
                "chunks": np.array(chunks_of_chunks),
                "line_idxs": np.array(line_idxs_of_chunks),
            }
        else:
            serialized_tokenized = {
                "num_tokens": np.array([data["num_tokens"] for data in tokenized]),
                "proxy_attn_flops": np.array([data["proxy_attn_flops"] for data in tokenized]),
            }

        if dist.is_initialized():
            # TODO:
            # This is a workaround for `all_gather_object` would hang when
            # using `nccl` backend. Maybe we could find a better way to
            # synchronize the `num_tokens` since datasets are not always initialized
            # with the world size.
            all_tokenized = [None] * world_size
            dist.all_gather_object(all_tokenized, serialized_tokenized, group=self.process_group)
            serialized_tokenized_global = {
                k: _concat_values([data[k] for data in all_tokenized]) for k in serialized_tokenized.keys()
            }
        else:
            serialized_tokenized_global = serialized_tokenized

        if rank == 0 and cache_dir:
            save_dict_to_npy_dir(serialized_tokenized_global, os.path.join(cache_dir, "jsonl_meta"))

        self.tokenize_fn.set_state("runtime")
        return serialized_tokenized_global

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, item) -> T | CacheItem:
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

        if self.tokenize_fn is not None:
            if self._has_chunk:
                assert "chunks" in self._meta, "chunks must be in _meta"
                cs = int(self._meta["chunks"][item][0])
                ce = int(self._meta["chunks"][item][1])
                token_start_offset = int(self._meta["chunks"][item][2])
                return self.tokenize_fn(raw_data, char_start=cs, char_end=ce, token_start_offset=token_start_offset)
            return self.tokenize_fn(raw_data)
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
        """Release shared memory."""
        if dist.is_initialized():
            dist.barrier()
            self._shared_memory.close()
        else:
            self._shared_memory.close()

        local_rank_concurrency = get_local_world_size()
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

    # JsonlDataset does not need to save or load state dict for resuming.
    def load_state_dict(self, state_dict: dict): ...

    def get_state_dict(self):
        return {}
