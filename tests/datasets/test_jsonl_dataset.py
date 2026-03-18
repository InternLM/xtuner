import os
import gc
import psutil
import time
import random
import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import DistributedTestBase

from transformers import AutoTokenizer

from xtuner.v1.datasets import FTDPTokenizeFnConfig
from xtuner.v1.datasets.config import DatasetConfig
from xtuner.v1.utils.device import get_device


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
        if device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"):
            torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
        return ret

    @property
    def world_size(self) -> int:
        # 默认按八卡跑；本地可用环境变量临时改小做快速验证
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "8"))

    def test_build_jsonl_dataset_distributed(self):
        alpaca_path = os.path.join(os.environ["ALPACA_PATH"], "202404121913-shard-1-of-3.jsonl")
        tokenizer_path = os.environ["QWEN3_MOE_PATH"]

        self.create_pg(DEVICE)
        rank = dist.get_rank()

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        dataset_cfg = DatasetConfig(name="alpaca", anno_path=alpaca_path, sample_ratio=1000.0)
        tokenize_fn_cfg = FTDPTokenizeFnConfig(max_length=16386)
        tokenize_fn = tokenize_fn_cfg.build(tokenizer)

        # Get memory used during dataset build
        gc.collect()
        dist.barrier()
        start_time = time.time()
        rss_before, pss_before = _get_rss_mb(), _get_pss_mb()
        if rank == 0:
            print(f"[Rank {rank}] RSS before dataset build: {rss_before:.2f} MB")
            print(f"[Rank {rank}] PSS before dataset build: {pss_before:.2f} MB")

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
            print(f"[Rank {rank}] RSS after  dataset build: {rss_after:.2f} MB")
            print(f"[Rank {rank}] RSS delta: {rss_after - rss_before:.2f} MB")
            print(f"[Rank {rank}] PSS after  dataset build: {pss_after:.2f} MB")
            print(f"[Rank {rank}] PSS delta: {pss_after - pss_before:.2f} MB")

        print(f"[Rank {rank}] Build Time cost: {time_cost:.2f} s")
        dist.barrier()

        # Random read 10000 samples, and test time cost
        start_time = time.time()
        for i in range(10000):
            idx = random.randint(0, length - 1)
            _ = dataset[idx]
        time_cost = time.time() - start_time
        print(f"[Rank {rank}] Random read 10000 samples Time cost: {time_cost:.2f} s")
        dist.barrier()
