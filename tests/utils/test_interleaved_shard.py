"""Distributed behavior tests for ``xtuner.v1.utils.interleaved_shard``.

The 2D cases cover the ``(Shard, InterleavedShard)`` layout produced by
``GroupedLinear``. The 3D case prepends FSDP, matching the runtime layout seen
by HF save/load.
"""

from __future__ import annotations

import shutil
import tempfile

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.utils.dtensor import cal_total_norm
from xtuner.v1.utils.interleaved_shard import (
    InterleavedShard,
    RuntimeLayout,
    reconstruct_full_tensor,
)


NUM_EXPERTS = 4
OUT_PER_EXPERT = 4
IN_FEATURES = 8
GLOBAL_ROWS = NUM_EXPERTS * OUT_PER_EXPERT


def _build_expected_local(
    global_tensor: torch.Tensor,
    ep_rank: int,
    tp_rank: int,
    ep_size: int,
    tp_size: int,
) -> torch.Tensor:
    """Return the hand-computed per-expert column-parallel slice."""
    experts_per_ep = NUM_EXPERTS // ep_size
    rows_per_expert = global_tensor.shape[0] // NUM_EXPERTS
    rows_per_tp_per_expert = rows_per_expert // tp_size
    chunks = []
    for local_expert in range(experts_per_ep):
        global_expert = ep_rank * experts_per_ep + local_expert
        expert_start = global_expert * rows_per_expert
        row_start = expert_start + tp_rank * rows_per_tp_per_expert
        chunks.append(global_tensor[row_start : row_start + rows_per_tp_per_expert])
    return torch.cat(chunks, dim=0)


class TestInterleavedShard2D(DeterministicDDPTestCase):
    def test_layout_and_reconstruct(self) -> None:
        self.create_pg("cuda")
        mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("ep", "tp"))
        ep_rank = mesh.get_local_rank("ep")
        tp_rank = mesh.get_local_rank("tp")

        global_tensor = torch.arange(
            GLOBAL_ROWS * IN_FEATURES,
            device="cuda",
            dtype=torch.float32,
        ).reshape(GLOBAL_ROWS, IN_FEATURES)
        dist.broadcast(global_tensor, src=0)

        placements = (Shard(0), InterleavedShard(0, num_local_stripes=NUM_EXPERTS // 2))
        tensor = distribute_tensor(global_tensor, mesh, placements)

        expected = _build_expected_local(global_tensor, ep_rank, tp_rank, 2, 2)
        torch.testing.assert_close(tensor.to_local(), expected)
        assert RuntimeLayout.from_dtensor(tensor).is_interleaved
        torch.testing.assert_close(reconstruct_full_tensor(tensor), global_tensor)

    def test_hf_round_trip(self) -> None:
        """Exercise the public BaseModel HF save/load path."""
        from transformers import PretrainedConfig
        from xtuner.v1.model.base import BaseModel, XTunerBaseModelConfig

        class _ToyConfig(XTunerBaseModelConfig):
            @property
            def hf_config(self) -> PretrainedConfig:
                return PretrainedConfig()

        class _ToyModel(BaseModel):
            def __init__(self, weight: DTensor):
                super().__init__(_ToyConfig())
                self.weight = nn.Parameter(weight)
                self._init_load_spec()

            def to_hf_key_list(self, key: str) -> list[str]:
                return [key]

        self.create_pg("cuda")
        mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("ep", "tp"))
        placements = (Shard(0), InterleavedShard(0, num_local_stripes=NUM_EXPERTS // 2))
        global_weight = torch.arange(
            GLOBAL_ROWS * IN_FEATURES,
            device="cuda",
            dtype=torch.bfloat16,
        ).reshape(GLOBAL_ROWS, IN_FEATURES)
        dist.broadcast(global_weight, src=0)
        model = _ToyModel(distribute_tensor(global_weight, mesh, placements))

        checkpoint_dirs = [tempfile.mkdtemp() if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(checkpoint_dirs, src=0)
        checkpoint_dir = checkpoint_dirs[0]
        assert checkpoint_dir is not None

        try:
            model.save_hf(checkpoint_dir)
            restored_weight = distribute_tensor(torch.zeros_like(global_weight), mesh, placements)
            restored = _ToyModel(restored_weight)
            restored.from_hf(checkpoint_dir)
            torch.testing.assert_close(restored.weight.to_local(), model.weight.to_local(), rtol=0, atol=0)
        finally:
            dist.barrier()
            if dist.get_rank() == 0:
                shutil.rmtree(checkpoint_dir)

    @property
    def world_size(self) -> int:
        return 4


class _ToyGroupedLinear(nn.Module):
    def __init__(self, weight: DTensor):
        super().__init__()
        self.weight = nn.Parameter(weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to_local() if isinstance(self.weight, DTensor) else self.weight
        return torch.nn.functional.linear(inputs, weight)


class TestInterleavedShardPostFSDP(DeterministicDDPTestCase):
    def test_reconstruct_and_load(self) -> None:
        self.create_pg("cuda")
        mesh = init_device_mesh("cuda", (2, 2, 2), mesh_dim_names=("fsdp", "ep", "tp"))
        ep_tp_mesh = mesh["ep", "tp"]
        fsdp_mesh = mesh["fsdp"]

        global_tensor = torch.arange(
            GLOBAL_ROWS * IN_FEATURES,
            device="cuda",
            dtype=torch.float32,
        ).reshape(GLOBAL_ROWS, IN_FEATURES)
        dist.broadcast(global_tensor, src=0)

        placements = (Shard(0), InterleavedShard(0, num_local_stripes=NUM_EXPERTS // 2))
        tensor = distribute_tensor(global_tensor, ep_tp_mesh, placements)
        model = _ToyGroupedLinear(tensor).cuda()
        fully_shard(
            model,
            mesh=fsdp_mesh,
            mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
            reshard_after_forward=True,
        )

        inputs = torch.randn(6, IN_FEATURES, device="cuda", dtype=torch.bfloat16)
        dist.broadcast(inputs, src=0)
        output = model(inputs)
        expected_local = _build_expected_local(
            global_tensor,
            mesh.get_local_rank("ep"),
            mesh.get_local_rank("tp"),
            2,
            2,
        ).to(torch.bfloat16)
        expected_output = torch.nn.functional.linear(inputs, expected_local)
        torch.testing.assert_close(output.detach(), expected_output, atol=1e-2, rtol=1e-2)
        output.sum().backward()

        assert RuntimeLayout.from_dtensor(model.weight).is_interleaved
        torch.testing.assert_close(reconstruct_full_tensor(model.weight), global_tensor)

        # LoadSpec compiles the same post-FSDP layout into source-to-local copy
        # runs, so loading does not need model-specific Expert TP branches.
        from xtuner.v1.utils.load_spec import LoadSpec

        local = model.weight._local_tensor
        loaded_local = torch.empty_like(local, dtype=global_tensor.dtype)
        load_spec = LoadSpec.from_tensor(name="weight", hf_keys=["weight"], tensor=model.weight)
        load_spec.plan_hf_load().load_into(
            [global_tensor],
            loaded_local,
            lambda _, checkpoint_tensor: checkpoint_tensor,
        )
        torch.testing.assert_close(loaded_local, local.to(global_tensor.dtype))

    @property
    def world_size(self) -> int:
        return 8


class TestNestedShardGradNorm(DeterministicDDPTestCase):
    def test_cal_total_norm_for_fsdp2_ep4(self) -> None:
        """FSDP's prepended shard and EP's shard must both contribute."""
        self.create_pg("cuda")
        mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=("fsdp", "ep"))

        global_weight = torch.zeros(
            GLOBAL_ROWS,
            IN_FEATURES,
            device="cuda",
            dtype=torch.bfloat16,
        )
        tensor = distribute_tensor(global_weight, mesh["ep"], (Shard(0),))
        model = _ToyGroupedLinear(tensor).cuda()
        fully_shard(
            model,
            mesh=mesh["fsdp"],
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
            ),
            reshard_after_forward=True,
        )

        inputs = torch.ones(6, IN_FEATURES, device="cuda", dtype=torch.bfloat16)
        model(inputs).sum().backward()

        assert isinstance(model.weight.grad, DTensor)
        total_norm = cal_total_norm([model.weight.grad], foreach=True)
        expected = torch.tensor(
            6 * (GLOBAL_ROWS * IN_FEATURES) ** 0.5,
            device="cuda",
            dtype=torch.float32,
        )
        torch.testing.assert_close(total_norm, expected)

    @property
    def world_size(self) -> int:
        return 8
