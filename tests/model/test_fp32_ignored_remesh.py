"""Regression tests for FP32-ignored param remesh inside ``BaseModel._fully_shard``.

Contract: parameters matching ``fp32_keys_pattern`` must end as
``DTensor(world_mesh, all Replicate)`` so grad reduce / norm stay consistent
across FSDP ranks. Remesh must be idempotent under MoE's layer-then-root
``_fully_shard`` calls, and must refuse sharded layouts instead of gathering.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.config import FSDPConfig
from xtuner.v1.model.base import BaseModel, HFSaveCfg, XTunerBaseModelConfig


class _ToyConfig(XTunerBaseModelConfig):
    compile_cfg: bool = False
    hf_save_cfg: HFSaveCfg = HFSaveCfg(fp32_keys_pattern=[r"fp32_weight"])


class _ToyLayer(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.fp32_weight = nn.Parameter(weight)
        self.other_weight = nn.Parameter(torch.zeros_like(weight))


class _ToyModel(BaseModel):
    config: _ToyConfig

    def __init__(self, config: _ToyConfig, fp32_weight: torch.Tensor):
        super().__init__(config)
        self.layer = _ToyLayer(fp32_weight)
        self._init_load_spec()

    def to_hf_key_list(self, key: str) -> list[str]:
        return [key]


def _mp_policy() -> MixedPrecisionPolicy:
    return MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32)


class TestFP32IgnoredRemesh(DeterministicDDPTestCase):
    @property
    def world_size(self) -> int:
        return 4

    def _build_model(self, fp32_weight: torch.Tensor) -> _ToyModel:
        model = _ToyModel(_ToyConfig(), fp32_weight).cuda()
        world_mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("fsdp", "ep"))
        model._world_mesh = world_mesh
        model.fsdp_mesh = world_mesh["fsdp"]
        model.fsdp_config = FSDPConfig(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            torch_compile=False,
        )
        return model

    def _call_fully_shard(self, model: _ToyModel, module: nn.Module | None) -> None:
        model._fully_shard(
            mesh=model.fsdp_mesh,
            mp_policy=_mp_policy(),
            reshard_after_forward=True,
            offload_policy=None,
            module=module,
        )

    def test_remesh_ep_replicate_onto_world(self) -> None:
        """EP-only Replicate (MoE ``_replicate_other_params``) must become world Replicate."""
        self.create_pg("cuda")
        torch.manual_seed(0)
        weight = torch.randn(4, 4, device="cuda", dtype=torch.float32)
        torch.distributed.broadcast(weight, src=0)

        model = self._build_model(weight)
        ep_mesh = model.world_mesh["ep"]
        model.layer.fp32_weight = nn.Parameter(
            distribute_tensor(weight, ep_mesh, [Replicate()]),
            requires_grad=True,
        )
        # Refresh load_spec ids after replacing the parameter.
        model._init_load_spec()

        self._call_fully_shard(model, module=model.layer)

        remeshed = model.layer.fp32_weight
        assert isinstance(remeshed, DTensor)
        assert remeshed.device_mesh == model.world_mesh
        assert all(placement.is_replicate() for placement in remeshed.placements)
        torch.testing.assert_close(remeshed.to_local(), weight)

    def test_remesh_is_idempotent_across_layer_and_root(self) -> None:
        """Layer then root ``_fully_shard`` must not rebuild an already-correct replica."""
        self.create_pg("cuda")
        torch.manual_seed(1)
        weight = torch.randn(4, 4, device="cuda", dtype=torch.float32)
        torch.distributed.broadcast(weight, src=0)

        model = self._build_model(weight)
        ep_mesh = model.world_mesh["ep"]
        model.layer.fp32_weight = nn.Parameter(
            distribute_tensor(weight, ep_mesh, [Replicate()]),
            requires_grad=True,
        )
        model._init_load_spec()

        self._call_fully_shard(model, module=model.layer)
        after_layer = model.layer.fp32_weight
        assert isinstance(after_layer, DTensor)
        assert after_layer.device_mesh == model.world_mesh

        # MoE root pass traverses into the already-sharded layer again.
        self._call_fully_shard(model, module=None)
        assert model.layer.fp32_weight is after_layer

    def test_remesh_rejects_sharded_layout(self) -> None:
        """Sharded FP32-pattern matches must fail closed, not ``full_tensor()`` gather."""
        self.create_pg("cuda")
        torch.manual_seed(2)
        weight = torch.randn(8, 4, device="cuda", dtype=torch.float32)
        torch.distributed.broadcast(weight, src=0)

        model = self._build_model(weight)
        ep_mesh = model.world_mesh["ep"]
        model.layer.fp32_weight = nn.Parameter(
            distribute_tensor(weight, ep_mesh, [Shard(0)]),
            requires_grad=True,
        )
        model._init_load_spec()

        with self.assertRaisesRegex(ValueError, r"non-replicate|Shard|cannot rematerialize"):
            self._call_fully_shard(model, module=model.layer)
