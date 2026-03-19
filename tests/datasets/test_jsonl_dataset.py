import os
import gc
import psutil
import time
import random

import tracemalloc
import parametrize
import numpy as np
import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import DistributedTestBase

from transformers import AutoTokenizer

from xtuner.v1.datasets import FTDPTokenizeFnConfig
from xtuner.v1.datasets.config import DatasetConfig
from xtuner.v1.utils.device import get_device
from xtuner.v1.datasets.jsonl import _apply_sample_ratio, _filter_sampled_indices


DEVICE = get_device()


def _get_rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def _get_pss_mb() -> float:
    proc = psutil.Process(os.getpid())
    return proc.memory_full_info().pss / 1024 / 1024


class TestJsonlDatasetSmokeTest(DistributedTestBase):
    def create_pg(self, device):
        ret = super().create_pg(device)
        os.environ["LOCAL_RANK"] = str(dist.get_rank())
        os.environ["LOCAL_WORLD_SIZE"] = str(self.world_size)
        if device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"):
            torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
        return ret

    @property
    def world_size(self) -> int:
        # 默认按八卡跑；本地可用环境变量临时改小做快速验证
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "8"))

    @parametrize.parametrize("enable_mmap_shared", [(False,), (True,)])
    def test_jsonl_dataset_smoke_test(self, enable_mmap_shared: bool):
        alpaca_path = os.path.join(os.environ["ALPACA_PATH"], "202404121913-shard-1-of-3.jsonl")
        tokenizer_path = os.environ["QWEN3_MOE_PATH"]

        self.create_pg(DEVICE)
        rank = dist.get_rank()

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        dataset_cfg = DatasetConfig(name="alpaca", anno_path=alpaca_path, sample_ratio=1000.0, enable_mmap_shared=enable_mmap_shared)
        tokenize_fn_cfg = FTDPTokenizeFnConfig(max_length=16386)
        tokenize_fn = tokenize_fn_cfg.build(tokenizer)

        # Get memory used during dataset build
        gc.collect()
        dist.barrier()

        start_time = time.time()
        rss_before, pss_before = _get_rss_mb(), _get_pss_mb()
        if rank == 0:
            print(f"[Rank {rank}] Before dataset build: RSS={rss_before:.2f} MB, PSS={pss_before:.2f} MB")

        dataset = dataset_cfg.build(tokenize_fn)
        time_cost = time.time() - start_time

        gc.collect()
        dist.barrier()
        rss_after, pss_after = _get_rss_mb(), _get_pss_mb()
        # Check: same length in all ranks
        length = len(dataset)
        length_list = [length]
        dist.broadcast_object_list(length_list, src=0)
        self.assertEqual(length, length_list[0])
        self.assertGreater(length, 0)

        if rank == 0:
            print(f"[Rank {rank}] dataset length: {length}")
            print(f"[Rank {rank}] After dataset build: RSS={rss_after:.2f} MB, PSS={pss_after:.2f} MB")
            print(f"[Rank {rank}] After dataset build: RSS delta={rss_after - rss_before:.2f} MB, PSS delta={pss_after - pss_before:.2f} MB")

            print(f"[Rank {rank}] Build Time cost: {time_cost:.2f} s")
        dist.barrier()

        # Random read 10000 samples, and test time cost
        start_time = time.time()
        for i in range(10000):
            idx = random.randint(0, length - 1)
            _ = dataset[idx]
        time_cost = time.time() - start_time
        if rank == 0:
            print(f"[Rank {rank}] Random read 10000 samples Time cost: {time_cost:.2f} s")
        dist.barrier()

    def test_mmap_shared_pss_lower_than_baseline(self):
        """PSS per rank with enable_mmap_shared should be lower than without, because physical pages are shared."""
        alpaca_path = os.path.join(os.environ["ALPACA_PATH"], "202404121913-shard-1-of-3.jsonl")
        tokenizer_path = os.environ["QWEN3_MOE_PATH"]

        self.create_pg(DEVICE)
        rank = dist.get_rank()

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        tokenize_fn_cfg = FTDPTokenizeFnConfig(max_length=16386)
        tokenize_fn = tokenize_fn_cfg.build(tokenizer)

        def _build_and_measure(enable_mmap_shared: bool):
            gc.collect()
            dist.barrier()
            tracemalloc.start()
            pss_before, rss_before = _get_pss_mb(), _get_rss_mb()
            cfg = DatasetConfig(
                name="alpaca", anno_path=alpaca_path, sample_ratio=0.8,
                enable_mmap_shared=enable_mmap_shared,
                enable_sequential_sampler=True,
            )
            ds = cfg.build(tokenize_fn)
            gc.collect()
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            for stat in top_stats[:10]:
                if rank < 2:
                    print(f"[Rank {rank}][mmap={enable_mmap_shared}] {stat}")
            dist.barrier()
            pss_delta = _get_pss_mb() - pss_before
            rss_delta = _get_rss_mb() - rss_before
            return pss_delta, rss_delta, ds

        pss_delta_base, rss_delta_base, dataset_base = _build_and_measure(enable_mmap_shared=False)
        if rank < 2:
            print(f"[Rank {rank}] No mmap: PSS delta: {pss_delta_base:.2f} MB, RSS delta: {rss_delta_base:.2f} MB")

        # Release baseline dataset before mmap test
        item_base = dataset_base[3]
        del dataset_base

        gc.collect()
        dist.barrier()

        pss_delta_mmap, rss_delta_mmap, dataset_mmap = _build_and_measure(enable_mmap_shared=True)
        item_mmap = dataset_mmap[3]
        if rank < 2:
            print(f"[Rank {rank}] mmap: PSS delta: {pss_delta_mmap:.2f} MB, RSS delta: {rss_delta_mmap:.2f} MB")

        self.assertLess(pss_delta_mmap, pss_delta_base,
                        msg=f"mmap PSS delta ({pss_delta_mmap:.1f} MB) should be less than baseline ({pss_delta_base:.1f} MB)")
        
        self.assertSequenceEqual(item_base["input_ids"], item_mmap["input_ids"], msg=f"input_ids should be equal, but got {item_base['input_ids']} and {item_mmap['input_ids']}")


def test_build_sampled_indices_uses_numpy_array_and_matches_sampling_semantics():
    num_offsets = 5  # non-chunk: base indices 0..3
    num_tokens = np.asarray([1, 0, 2, 3], dtype=np.int64)
    base_len = num_offsets - 1
    dtype = np.int32 if base_len < np.iinfo(np.int32).max else np.int64
    sampled = np.arange(base_len, dtype=dtype)
    sampled = _filter_sampled_indices(sampled, num_tokens, max_length=2)
    sampled = _apply_sample_ratio(sampled, sample_ratio=1.5, enable_sequential_sampler=True)
    assert isinstance(sampled, np.ndarray)
    assert sampled.dtype.kind in ("i", "u")
    # After filtering: indices [0,2] (idx1 damaged, idx3 > max_length)
    # sample_ratio=1.5 => target=3, base_repeats=1 => [0,2] + one extra sampled from [0,2]
    assert sampled.tolist() == [0, 2, 0]
