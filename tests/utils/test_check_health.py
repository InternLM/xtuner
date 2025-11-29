import os
import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import DistributedTestBase
from unittest.mock import patch, Mock

import xtuner.v1.utils.check_health as check_health

from xtuner.v1.utils.device import get_device


DEVICE = get_device()


def fake_health_job(dtype, loop=10):
    if dist.get_rank() == 1:
        print(f"rank {dist.get_rank()} world size {dist.get_world_size()} return 0.0")
        return torch.tensor(0.0, dtype=dtype, device=DEVICE)
    else:
        print(f"rank {dist.get_rank()} world size {dist.get_world_size()} return 1.0")
        return torch.tensor(1.0, dtype=dtype, device=DEVICE)


class TestCheckHealth(DistributedTestBase):
    def create_pg(self, device):
        ret = super().create_pg(device)
        os.environ["LOCAL_RANK"] = str(dist.get_rank())
        torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
        return ret

    def test_check_health_normal(self):
        self.create_pg(DEVICE)

        self.assertTrue(check_health.check_health())

    def test_check_health_failed(self):
        self.create_pg(DEVICE)

        with patch("xtuner.v1.utils.check_health.health_job", fake_health_job):
            self.assertFalse(check_health.check_health())
