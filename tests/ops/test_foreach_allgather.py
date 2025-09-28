from threading import local
from torch.testing._internal.common_distributed import DistributedTestBase
import torch
import torch.distributed as dist
import os

import parametrize
from xtuner.v1.ops.comm.foreach_allgather import foreach_all_gather
import time


class TestMoETorchAll2AllDispatcher(DistributedTestBase):
    @parametrize.parametrize("device", [("cuda",)])
    def test_foreach_all_gather_acc(self, device):
        self.create_pg(device)

        # Create dummy parameters for testing
        local_rank = dist.get_rank()
        local_data = [torch.tensor(local_rank + i, device=device) for i in range(4)]
        # rank0: [torch.tensor(0), torch.tensor(1), torch.tensor(2), torch.tensor(3)]
        # rank1: [torch.tensor(1), torch.tensor(2), torch.tensor(3), torch.tensor(4)]

        expected = [
            [torch.tensor(0, device=device), torch.tensor(1, device=device)],
            [torch.tensor(1, device=device), torch.tensor(2, device=device)],
            [torch.tensor(2, device=device), torch.tensor(3, device=device)],
            [torch.tensor(3, device=device), torch.tensor(4, device=device)],
        ]

        # Gather parameters using foreach_all_gather
        gathered_params = foreach_all_gather(local_data, dist.group.WORLD)

        # Check that the gathered parameters have the correct shape
        for a, b in zip(gathered_params, expected):
            for x, y in zip(a, b):
                self.assertTrue(torch.equal(x, y))

    @parametrize.parametrize(
        "device,shape",
        [
            ("cuda", (1,)),
            ("cuda", (1, 2)),
            ("cuda", (1, 2, 3)),
        ]
    )
    def test_foreach_all_gather_efficiency(self, device, shape):
        self.create_pg(device)

        # Create dummy parameters for testing
        local_data_list = [torch.randn(shape, device=device) for _ in range(1000)]

        pos1 = time.time()
        expceted_results = []
        for data in local_data_list:
            # Ensure each tensor is unique
            output_list = [torch.zeros_like(data, device=device) for i in range(dist.get_world_size())]
            dist.all_gather(output_list, data, group=dist.group.WORLD)
            expceted_results.extend(output_list)

        pos2 = time.time()
        results = sum(foreach_all_gather(local_data_list, dist.group.WORLD), [])
        pos3 = time.time()

        self.assertTrue(pos3 - pos2 < pos2 - pos1, "foreach_all_gather should be more efficient than manual all_gather")

        for res, exp in zip(results, expceted_results):
            self.assertTrue(torch.equal(res, exp), "The gathered tensors should match the expected tensors")

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "2"))
