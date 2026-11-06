"""LogProb loss context sequence-parallel regression tests.

Covered behavior:
- TestLogProbSP
    - test_build_with_sp_mesh_splits_labels: ``LogProbConfig.build(data, sp_mesh)``
      must split ``shifted_labels`` for every SP rank (regression: LogProbKwargs
      lacked ``sp_split`` and crashed with AttributeError under SP > 1); odd
      lengths are padded with ignore_idx so padding never enters logprob reads.
"""

import os
import unittest

import parametrize
import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import DistributedTestBase

from xtuner.v1.loss.rl_loss import LogProbConfig
from xtuner.v1.utils.test_utils import init_data_mesh


IGNORE_IDX = -100


class TestLogProbSP(DistributedTestBase):
    def test_build_with_sp_mesh_splits_labels(self) -> None:
        device = "cuda"
        pg = self.create_pg(device)
        sp = self.world_size
        if sp < 2:
            raise unittest.SkipTest("SP parity tests need sp >= 2")
        mesh = init_data_mesh(device, sp)["sp"]
        rank = dist.get_rank()

        seq_len = 2 * sp + 1  # force one padding column on the last rank
        labels = torch.randint(0, 32, (2, seq_len), device=device)
        dist.broadcast(labels, src=0)

        ctx = LogProbConfig().build({"shifted_labels": labels}, sp_mesh=mesh, device=device)
        assert ctx is not None
        shard = ctx.loss_kwargs.shifted_labels
        padded = (seq_len + sp - 1) // sp * sp
        assert shard.shape == (2, padded // sp)
        expected = torch.nn.functional.pad(labels, (0, padded - seq_len), value=IGNORE_IDX)
        assert torch.equal(shard, expected[:, rank * shard.shape[1] : (rank + 1) * shard.shape[1]])
        if rank == sp - 1:
            assert shard[:, -1].eq(IGNORE_IDX).all()
        dist.destroy_process_group(pg)

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "2"))
