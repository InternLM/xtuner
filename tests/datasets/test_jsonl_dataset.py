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
from xtuner.v1.datasets.jsonl import (
    JsonlDataset,
    _apply_sample_ratio,
    _filter_sampled_indices,
    load_dict_from_npy_dir,
    save_dict_to_npy_dir,
)


DEVICE = get_device()


def _get_rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def _get_pss_mb() -> float:
    proc = psutil.Process(os.getpid())
    return proc.memory_full_info().pss / 1024 / 1024


class TestJsonlDatasetDist(DistributedTestBase):
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
        dataset_cfg = DatasetConfig(name="alpaca", anno_path=alpaca_path, sample_ratio=100.0, enable_mmap_shared=enable_mmap_shared)
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
        for i in range(100):
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

    def _run_cache_consistency(self, use_cache_tag: bool, enable_mmap_shared: bool = False):
        """
        Helper: first build creates npy cache; second build loads from cache without re-tokenizing.
        use_cache_tag=False → second build goes through cache_dir branch.
        use_cache_tag=True  → second build goes through cache_tag fast-path branch.
        enable_mmap_shared  → when True, also verifies offsets are memmap-backed after cache load.
        """
        import shutil
        from pathlib import Path

        alpaca_path = os.path.join(os.environ["ALPACA_PATH"], "202404121913-shard-1-of-3.jsonl")
        tokenizer_path = os.environ["QWEN3_MOE_PATH"]

        self.create_pg(DEVICE)
        rank = dist.get_rank()

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        tokenize_fn_cfg = FTDPTokenizeFnConfig(max_length=16386)
        tokenize_fn = tokenize_fn_cfg.build(tokenizer)

        tag = "npy_test" if use_cache_tag else None
        suffix = "with_tag" if use_cache_tag else "no_tag"
        if enable_mmap_shared:
            suffix += "_mmap"
        cache_dir = f"/tmp/xtuner_test_cache_npy_{suffix}"
        if rank == 0:
            shutil.rmtree(cache_dir, ignore_errors=True)
            os.makedirs(cache_dir, exist_ok=True)
        dist.barrier()

        try:
            cfg = DatasetConfig(
                name="alpaca", anno_path=alpaca_path, sample_ratio=1.0,
                cache_dir=cache_dir, cache_tag=tag,
                enable_mmap_shared=enable_mmap_shared,
            )

            # First build: no cache, tokenizes from scratch
            ds_ref = cfg.build(tokenize_fn)
            dist.barrier()

            # Verify npy cache directory was created with .npy files
            if rank == 0:
                meta_dirs = list(Path(cache_dir).rglob("jsonl_meta"))
                self.assertTrue(len(meta_dirs) > 0, "jsonl_meta/ dir should exist after first build")
                self.assertTrue(len(list(meta_dirs[0].glob("*.npy"))) > 0,
                                "jsonl_meta/ should contain .npy files")
            dist.barrier()

            # Second build: count_tokens must NOT be called (cache was loaded)
            was_called = [False]
            _orig = JsonlDataset.count_tokens

            def _patched(self_, *args, **kwargs):
                was_called[0] = True
                return _orig(self_, *args, **kwargs)

            JsonlDataset.count_tokens = _patched
            try:
                ds_res = cfg.build(tokenize_fn)
            finally:
                JsonlDataset.count_tokens = _orig

            self.assertFalse(was_called[0], "count_tokens should not be called when loading from npy cache")

            # When enable_mmap_shared is True, the final self.offsets must be memmap-backed
            # (written to tmp_dir then mmap-loaded by all ranks, including local rank 0).
            if enable_mmap_shared:
                self.assertIsInstance(
                    ds_res.offsets, np.memmap,
                    msg="offsets should be memmap-backed when enable_mmap_shared=True",
                )
                # verify _meta dict values are memmap-backed
                for k, v in ds_res._meta.items():
                    self.assertIsInstance(v, np.memmap,
                                          msg=f"{k} should be memmap-backed when enable_mmap_shared=True")

            # Results must be identical to first build
            self.assertEqual(len(ds_ref), len(ds_res))
            for i in [0, len(ds_ref) // 2, len(ds_ref) - 1]:
                self.assertSequenceEqual(ds_ref[i]["input_ids"], ds_res[i]["input_ids"],
                                         msg=f"input_ids mismatch at index {i}")
            dist.barrier()
        finally:
            if rank == 0:
                shutil.rmtree(cache_dir, ignore_errors=True)

    def test_cache_dir_npy_format_consistent(self):
        """cache_dir path: second build loads npy cache without re-tokenizing."""
        self._run_cache_consistency(use_cache_tag=False)

    def test_cache_tag_npy_format_consistent(self):
        """cache_tag fast-path: second build resolves stored paths and loads npy without re-tokenizing."""
        self._run_cache_consistency(use_cache_tag=True)

    def test_cache_dir_npy_format_consistent_with_mmap(self):
        """cache_dir + enable_mmap_shared: offsets are memmap-backed and results are correct."""
        self._run_cache_consistency(use_cache_tag=False, enable_mmap_shared=True)

    def test_cache_tag_npy_format_consistent_with_mmap(self):
        """cache_tag fast-path + enable_mmap_shared: offsets are memmap-backed and results are correct."""
        self._run_cache_consistency(use_cache_tag=True, enable_mmap_shared=True)


def test_npy_dir_meta_save_and_mmap_reload():
    """save_dict_to_npy_dir + load_dict_from_npy_dir: values match and large arrays are memmap-backed."""
    import shutil
    import tempfile

    data = {
        "proxy_attn_flops": np.array([10, 20, 30], dtype=np.int64),
        "chunks": np.array([[0, 5, 0], [5, 10, 3]], dtype=np.int64),
    }
    tmp_dir = tempfile.mkdtemp(prefix="xtuner_npy_meta_test_")
    try:
        save_dict_to_npy_dir(data, tmp_dir)
        loaded = load_dict_from_npy_dir(tmp_dir, mmap=True)

        assert set(loaded.keys()) == set(data.keys())
        for k in data:
            np.testing.assert_array_equal(loaded[k], data[k])
            assert isinstance(loaded[k], np.memmap), f"{k} should be memmap-backed"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


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
