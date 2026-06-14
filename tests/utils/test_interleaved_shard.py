"""Unit tests for ``xtuner.v1.utils.interleaved_shard``.

These tests cover the InterleavedShard placement and the ``reconstruct_full_tensor`` helper
across the layouts that XTuner actually uses:

  * Plain ``(Shard, InterleavedShard)`` on a 2D (ep, tp) mesh — the layout produced by
    ``GroupedLinear`` when TP is enabled.
  * The post-``fully_shard`` 3D layout with FSDP prepended on top — what HF save sees in
    practice.

Run with::

    /mnt/shared-storage-user/yehaochen/miniconda3/envs/py312-pt210/bin/torchrun \
        --nproc-per-node=8 tests/utils/test_interleaved_shard.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _REPO_ROOT)
# Import the module directly to avoid pulling in xtuner package's heavy deps (loguru etc.) that
# aren't required for this unit test.
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "interleaved_shard",
    os.path.join(_REPO_ROOT, "xtuner", "v1", "utils", "interleaved_shard.py"),
)
assert _spec is not None and _spec.loader is not None
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
InterleavedShard = _mod.InterleavedShard
has_interleaved_placement = _mod.has_interleaved_placement
reconstruct_full_tensor = _mod.reconstruct_full_tensor


NUM_EXPERTS = 4
OUT_PER_EXPERT = 4
IN_FEATURES = 8
GLOBAL_ROWS = NUM_EXPERTS * OUT_PER_EXPERT  # 16


def _build_expected_local(
    g: torch.Tensor,
    ep_rank: int,
    tp_rank: int,
    ep_size: int,
    tp_size: int,
) -> torch.Tensor:
    """Hand-computed per-expert column parallel slice."""
    experts_per_ep = NUM_EXPERTS // ep_size
    rows_per_expert = g.shape[0] // NUM_EXPERTS
    rows_per_tp_per_expert = rows_per_expert // tp_size
    chunks = []
    for local_expert in range(experts_per_ep):
        global_expert = ep_rank * experts_per_ep + local_expert
        expert_start = global_expert * rows_per_expert
        row_start = expert_start + tp_rank * rows_per_tp_per_expert
        chunks.append(g[row_start : row_start + rows_per_tp_per_expert])
    return torch.cat(chunks, dim=0)


def test_2d_layout_and_reconstruct():
    """Build a DTensor on (ep, tp) with (Shard, InterleavedShard) and reconstruct."""
    mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("ep", "tp"))
    ep_rank = mesh.get_local_rank("ep")
    tp_rank = mesh.get_local_rank("tp")

    g = torch.arange(GLOBAL_ROWS * IN_FEATURES, device="cuda", dtype=torch.float32).reshape(
        GLOBAL_ROWS, IN_FEATURES
    )
    dist.broadcast(g, src=0)

    placements = (Shard(0), InterleavedShard(0, num_local_stripes=NUM_EXPERTS // 2))
    dt = distribute_tensor(g, mesh, placements)

    # Layout correctness: per-rank local matches hand-computed per-expert column parallel.
    expected = _build_expected_local(g, ep_rank, tp_rank, 2, 2)
    assert torch.allclose(dt.to_local(), expected), (
        f"rank {dist.get_rank()} local mismatch"
    )

    # Detection helper works on this placement. ``shard_order`` only exists on torch>=2.10;
    # the implementation guards with ``getattr`` so the test must too.
    assert has_interleaved_placement(dt), "shard_order should be None for this placement"
    assert getattr(dt._spec, "shard_order", None) is None

    # Reconstruct gives back the global tensor.
    full = reconstruct_full_tensor(dt)
    assert torch.allclose(full, g), (
        f"reconstruct mismatch on 2D layout: max_diff={(full - g).abs().max().item()}"
    )


class _ToyGroupedLinear(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = nn.Parameter(weight)

    def forward(self, x):
        w = self.weight.to_local() if isinstance(self.weight, DTensor) else self.weight
        return torch.nn.functional.linear(x, w)


def test_post_fully_shard_reconstruct():
    """Layout after FSDP wraps the (ep, tp) DTensor — the case HF save actually sees."""
    mesh = init_device_mesh("cuda", (2, 2, 2), mesh_dim_names=("fsdp", "ep", "tp"))
    ep_tp = mesh["ep", "tp"]
    fsdp_mesh = mesh["fsdp"]

    g = torch.arange(GLOBAL_ROWS * IN_FEATURES, device="cuda", dtype=torch.float32).reshape(
        GLOBAL_ROWS, IN_FEATURES
    )
    dist.broadcast(g, src=0)

    placements = (Shard(0), InterleavedShard(0, num_local_stripes=NUM_EXPERTS // 2))
    dt = distribute_tensor(g, ep_tp, placements)

    model = _ToyGroupedLinear(dt).cuda()
    fully_shard(
        model,
        mesh=fsdp_mesh,
        mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
        reshard_after_forward=True,
    )

    # Sanity: a forward pass through the wrapped model still produces the right output.
    x = torch.randn(6, IN_FEATURES, device="cuda", dtype=torch.bfloat16)
    dist.broadcast(x, src=0)
    y = model(x)
    ep_rank = mesh.get_local_rank("ep")
    tp_rank = mesh.get_local_rank("tp")
    expected_local = _build_expected_local(g, ep_rank, tp_rank, 2, 2).to(torch.bfloat16)
    expected_y = torch.nn.functional.linear(x, expected_local)
    assert torch.allclose(y.detach(), expected_y, atol=1e-2, rtol=1e-2)
    y.sum().backward()

    # Detection helper still recognizes the wrapped DTensor.
    assert has_interleaved_placement(model.weight)

    # Reconstruct from the post-FSDP local matches the original global.
    full = reconstruct_full_tensor(model.weight)
    assert torch.allclose(full, g), (
        f"reconstruct mismatch on post-FSDP layout: max_diff={(full - g).abs().max().item()}"
    )


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    world = dist.get_world_size()

    rank = dist.get_rank()
    if world == 4:
        test_2d_layout_and_reconstruct()
        if rank == 0:
            print("[2d_layout_and_reconstruct] PASSED", flush=True)
    elif world == 8:
        test_post_fully_shard_reconstruct()
        if rank == 0:
            print("[post_fully_shard_reconstruct] PASSED", flush=True)
    else:
        if rank == 0:
            print(
                f"World size {world} not handled (expected 4 or 8). Skipping.", flush=True
            )
        dist.destroy_process_group()
        sys.exit(0)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
