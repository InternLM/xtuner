"""Regression tests for the InterleavedShard (tpep) DCP planners.

These validate that ``InterleavedShardSavePlanner`` / ``InterleavedShardLoadPlanner`` round-trip
per-expert column-parallel fused MoE weights through DCP. DCP's default planner models each
DTensor as a single contiguous chunk and silently mis-maps an ``InterleavedShard`` local tensor
(which is several interleaved runs), so these params used to be dropped from DCP checkpoints.

The layout is built with ``distribute_tensor`` on a 2D ``(ep, tp)`` mesh — the same
``(Shard, InterleavedShard)`` placement ``GroupedLinear`` produces — without ``fully_shard`` so the
test does not depend on FSDP2's support for the strided placement.

Topology: world_size = ep * tp = 2 * 2 = 4.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import parametrize
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.patch import InterleavedShardLoadPlanner, InterleavedShardSavePlanner
from xtuner.v1.utils.interleaved_shard import InterleavedShard, compute_runs, reconstruct_full_tensor


NUM_EXPERTS = 4
NUM_FUSED_PROJECTIONS = 2  # fused_w1w3 packs gate_proj + up_proj per expert.
PER_PROJ_OUT = 8
IN_FEATURES = 6
GLOBAL_ROWS = NUM_EXPERTS * NUM_FUSED_PROJECTIONS * PER_PROJ_OUT


def _global_source() -> torch.Tensor:
    # Deterministic, rank-independent global tensor so every rank agrees on ground truth.
    return torch.arange(GLOBAL_ROWS * IN_FEATURES, device="cuda", dtype=torch.float32).reshape(
        GLOBAL_ROWS, IN_FEATURES
    )


def _build_interleaved_dtensor(ep_size: int, tp_size: int) -> DTensor:
    # Build with ``from_local`` (not ``distribute_tensor``) to mirror ``GroupedLinear``: the latter
    # goes through ``redistribute``, which crashes on ``(Shard, InterleavedShard)`` on torch >= 2.9.
    mesh = init_device_mesh("cuda", (ep_size, tp_size), mesh_dim_names=("ep", "tp"))
    local_experts = NUM_EXPERTS // ep_size
    num_local_stripes = local_experts * NUM_FUSED_PROJECTIONS
    placements = (Shard(0), InterleavedShard(0, num_local_stripes=num_local_stripes))
    g = _global_source()
    # Scatter the deterministic global source into this rank's interleaved runs so every rank holds
    # real, distinct data (from_local does not scatter — the caller must supply the local shard).
    local = torch.empty(GLOBAL_ROWS // (ep_size * tp_size), IN_FEATURES, device="cuda")
    dt = DTensor.from_local(local, mesh, placements, run_check=False)
    for run in compute_runs(dt):
        start = run.global_offset[0]
        local[run.local_start : run.local_start + run.local_size] = g[start : start + run.local_size]
    return dt


class TestDCPInterleavedPlanner(DeterministicDDPTestCase):
    @parametrize.parametrize("device", [("cuda",)])
    def test_interleaved_round_trip(self, device: str) -> None:
        """Save then load an InterleavedShard DTensor under the same topology; local + global match."""
        pg = self.create_pg(device)

        src = _build_interleaved_dtensor(ep_size=2, tp_size=2)
        # ``distribute_tensor`` scatters the deterministic global source, giving each rank real data.
        assert len(compute_runs(src)) > 1, "expected multiple interleaved runs per rank"
        local_before = src._local_tensor.clone()
        full_before = reconstruct_full_tensor(src).clone()

        tmp = [tempfile.mkdtemp()] if dist.get_rank() == 0 else [None]
        dist.broadcast_object_list(tmp, src=0)
        ckpt = Path(tmp[0])

        dcp.save({"w": src}, checkpoint_id=ckpt, planner=InterleavedShardSavePlanner())
        dist.barrier()

        dst = _build_interleaved_dtensor(ep_size=2, tp_size=2)
        dst._local_tensor.zero_()
        dcp.load({"w": dst}, checkpoint_id=ckpt, planner=InterleavedShardLoadPlanner())
        dist.barrier()

        self.assertTrue(torch.equal(local_before, dst._local_tensor), "local shard mismatch after DCP round-trip")
        self.assertTrue(torch.equal(full_before, reconstruct_full_tensor(dst)), "global tensor mismatch")

        dist.barrier()
        torch.cuda.empty_cache()
        try:
            dist.destroy_process_group(pg)
        except Exception:
            pass

    @parametrize.parametrize("device", [("cuda",)])
    def test_interleaved_reshard_across_topology(self, device: str) -> None:
        """Checkpoint saved at (ep=2, tp=2) reloads correctly at (ep=1, tp=4) — global-coordinate storage."""
        pg = self.create_pg(device)

        src = _build_interleaved_dtensor(ep_size=2, tp_size=2)
        full_before = reconstruct_full_tensor(src).clone()

        tmp = [tempfile.mkdtemp()] if dist.get_rank() == 0 else [None]
        dist.broadcast_object_list(tmp, src=0)
        ckpt = Path(tmp[0])

        dcp.save({"w": src}, checkpoint_id=ckpt, planner=InterleavedShardSavePlanner())
        dist.barrier()

        dst = _build_interleaved_dtensor(ep_size=1, tp_size=4)
        dst._local_tensor.zero_()
        dcp.load({"w": dst}, checkpoint_id=ckpt, planner=InterleavedShardLoadPlanner())
        dist.barrier()

        self.assertTrue(
            torch.equal(full_before, reconstruct_full_tensor(dst)),
            "global tensor mismatch after resharding (ep=2,tp=2) -> (ep=1,tp=4)",
        )

        dist.barrier()
        torch.cuda.empty_cache()
        try:
            dist.destroy_process_group(pg)
        except Exception:
            pass

    @property
    def world_size(self) -> int:
        # (ep, tp) = (2, 2) → 4 GPUs.
        return 4

    @property
    def destroy_pg_upon_exit(self) -> bool:
        return False
