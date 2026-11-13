"""Real-process norm regressions; also runnable with torchrun for NCCL checks."""

from __future__ import annotations

import argparse
import os
import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
from torch.distributed.tensor.placement_types import Placement, _StridedShard

from xtuner.v1.utils.dtensor import cal_total_norm
from xtuner.v1.utils.grad_norm import cal_grad_norm


class TestCalTotalNormDistributed:
    def test_real_gloo_norm_clipping_and_adam(self, tmp_path: Path) -> None:
        """Compare distributed clipping and Adam against a dense reference.

        Args:
            tmp_path (Path): Directory for the temporary Gloo rendezvous.
        """
        context = mp.spawn(_worker, args=(4, f"file://{tmp_path / 'store'}", "cpu", 2), nprocs=4, join=False)
        deadline = time.monotonic() + 180
        try:
            while time.monotonic() < deadline:
                if context.join(timeout=1):
                    return
            raise TimeoutError("Distributed norm regression exceeded 180 seconds")
        finally:
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
            for process in context.processes:
                process.join(timeout=10)


def _wrap(full: torch.Tensor, local: torch.Tensor, mesh: DeviceMesh, placements: tuple[Placement, ...]) -> DTensor:
    return DTensor.from_local(local.clone(), mesh, placements, run_check=False, shape=full.shape, stride=full.stride())


def _check_norms(device_type: str, ep_size: int) -> None:
    # Strided FSDP sharding of an existing EP shard reverses the ownership order:
    # local = full.chunk(EP)[ep_rank].chunk(FSDP)[fsdp_rank].
    mesh = init_device_mesh(device_type, (2, ep_size), mesh_dim_names=("fsdp", "ep"))
    fsdp_rank, ep_rank = mesh.get_coordinate()
    device = torch.device(device_type, int(os.environ.get("LOCAL_RANK", "0"))) if device_type == "cuda" else "cpu"
    full = (torch.arange(2 * ep_size * 12, dtype=torch.float32, device=device).reshape(-1, 3) - 11) / 17
    strided = (_StridedShard(0, split_factor=ep_size), Shard(0))
    cases = [
        (strided, full.chunk(ep_size)[ep_rank].chunk(2)[fsdp_rank]),
        ((Shard(0), Shard(0)), full.chunk(2)[fsdp_rank].chunk(ep_size)[ep_rank]),
        ((Shard(0), Replicate()), full.chunk(2)[fsdp_rank]),
        ((Replicate(), Shard(0)), full.chunk(ep_size)[ep_rank]),
        ((Replicate(), Replicate()), full),
    ]
    expected = torch.linalg.vector_norm(torch.cat((full.flatten(), (full * 0.3).flatten())))
    mixed = []
    for placements, local in cases:
        grads = [_wrap(full, local, mesh, placements), _wrap(full * 0.3, local * 0.3, mesh, placements)]
        # Three dispatch paths on the regression layout; avoid a redundant full product.
        for foreach in (False, True, None) if placements == strided else (True,):
            torch.testing.assert_close(cal_total_norm(grads, foreach=foreach), expected)
        mixed.extend(grads)
    actual, groups = cal_grad_norm(mixed)
    assert len(groups) == len(cases)
    torch.testing.assert_close(actual, expected * len(cases) ** 0.5)

    local = full.chunk(ep_size)[ep_rank].chunk(2)[fsdp_rank]
    grads = [_wrap(full, local, mesh, strided)]
    torch.testing.assert_close(grads[0].full_tensor(), full)
    torch.testing.assert_close(cal_total_norm(grads, dtype=torch.float64), full.double().norm())
    torch.testing.assert_close(
        cal_total_norm([_wrap(full, torch.zeros_like(local), mesh, strided)]), full.new_zeros(())
    )
    assert cal_total_norm([]).item() == 0

    partial = _wrap(full, full, mesh, (Partial(), Replicate()))
    with pytest.raises(ValueError, match="Unsupported placement type"):
        cal_total_norm([partial], foreach=True)

    # Ordinary Shard supports nonuniform and empty slices; do not synthesize
    # unsupported uneven strided layouts. All ranks still join the reduction.
    flat_mesh = init_device_mesh(device_type, (2 * ep_size,))
    rank = dist.get_rank()
    for rows in (2 * ep_size + 1, 1):
        reference = torch.arange(rows * 3, dtype=torch.float32, device=device).reshape(rows, 3) + 1
        width = (rows + flat_mesh.size() - 1) // flat_mesh.size()
        shard = reference[rank * width : (rank + 1) * width]
        grad = _wrap(reference, shard, flat_mesh, (Shard(0),))
        torch.testing.assert_close(cal_total_norm([grad], foreach=True), reference.norm())

    # Exercise production clipping, not a test-only reimplementation. Compare
    # Adam moments as well: first-step Adam updates can mask uniform scale errors.
    from xtuner.v1.engine.train_engine import TrainEngine

    for max_norm in (0.25, 1000.0):
        grad = _wrap(full, local, mesh, strided)
        before, grouped = cal_grad_norm([grad])
        torch.testing.assert_close(before, full.norm())
        TrainEngine._clip_gradients([grad], max_norm, torch.float32, (before, grouped))
        coefficient = torch.clamp(max_norm / (full.norm() + 1e-6), max=1)
        torch.testing.assert_close(grad.to_local(), local * coefficient)
        torch.testing.assert_close(cal_total_norm([grad]), full.norm() * coefficient)
        param = torch.nn.Parameter(torch.ones_like(local))
        dense_param = torch.nn.Parameter(torch.ones_like(full))
        param.grad = grad.to_local().clone()
        dense_param.grad = full * coefficient
        optimizer = torch.optim.AdamW([param], lr=0.01)
        dense_optimizer = torch.optim.AdamW([dense_param], lr=0.01)
        optimizer.step()
        dense_optimizer.step()
        torch.testing.assert_close(param, dense_param.chunk(ep_size)[ep_rank].chunk(2)[fsdp_rank])
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                optimizer.state[param][key],
                dense_optimizer.state[dense_param][key].chunk(ep_size)[ep_rank].chunk(2)[fsdp_rank],
            )
        assert optimizer.state[param]["step"].item() == 1
    dist.barrier()
    if dist.get_rank() == 0:
        print(
            f"PASS: {device_type} {2 * ep_size} ranks, split_factor={ep_size}, norms/clipping/Adam moments", flush=True
        )


def _worker(rank: int, world_size: int, init_method: str, device_type: str, ep_size: int) -> None:
    torch.set_num_threads(1)
    if device_type == "cuda":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(
        "nccl" if device_type == "cuda" else "gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=180),
    )
    try:
        _check_norms(device_type, ep_size)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--ep-size", type=int, default=2)
    args = parser.parse_args()
    _worker(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), "env://", args.device, args.ep_size)
