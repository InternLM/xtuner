"""L0 placement tests for the MoE EP/FSDP mesh layouts.

These tests run in a single process on top of a fake ``ProcessGroup`` so that
8/16/64-rank layouts can be asserted without any multi-process launch. The
model stays on the ``meta`` device; only the mesh bookkeeping and the DTensor
placements produced by ``MoE.fully_shard`` are inspected.
"""

from collections.abc import Iterator
from contextlib import contextmanager

import pytest
import torch
import torch.distributed as dist
from pydantic import ValidationError
from torch.distributed.device_mesh import DeviceMesh, _mesh_resources
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.placement_types import _StridedShard
from torch.testing._internal.distributed.fake_pg import FakeStore

from xtuner.v1.config import FSDPConfig
from xtuner.v1.model.base import BaseModel
from xtuner.v1.model.moe.moe import MoE, MoEConfig
from xtuner.v1.module.attention import MHAConfig
from xtuner.v1.module.router import NoAuxRouterConfig


PREFIX = "l0mesh"
HIDDEN = 128
N_EXPERTS = 64
MOE_INTER = 128

DENSE_PARAMS = (
    "embed_tokens.weight",
    "layers.0.mlp.gate_proj.weight",
    "layers.1.self_attn.q_proj.weight",
    "layers.1.gate.weight",
    "layers.1.shared_experts.gate_proj.weight",
    "layers.1.input_layernorm.weight",
    "norm.weight",
    "lm_head.weight",
)
EXPERT_PARAMS = (
    "layers.1.experts.fused_w1w3.weight",
    "layers.1.experts.fused_w2.weight",
)


def _reset_mesh_resources() -> None:
    _mesh_resources.mesh_stack.clear()
    _mesh_resources.child_to_root_mapping.clear()
    _mesh_resources.root_to_flatten_mapping.clear()
    _mesh_resources.flatten_name_to_root_dims.clear()


@contextmanager
def _fake_world(world_size: int, rank: int) -> Iterator[None]:
    dist.init_process_group("fake", store=FakeStore(), rank=rank, world_size=world_size)
    try:
        yield
    finally:
        dist.destroy_process_group()
        _reset_mesh_resources()


# `MoE.__init__` unconditionally creates CUDA streams and dispatcher buffers, so a
# CUDA context is required even though the fake ProcessGroup never communicates.
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="MoE construction requires a CUDA device")


@pytest.fixture(autouse=True)
def _keep_model_on_meta(monkeypatch: pytest.MonkeyPatch) -> None:
    # Placements are fully determined before materialization; skipping
    # `_to_empty_meta` keeps the test free of device allocations.
    monkeypatch.setattr(BaseModel, "_to_empty_meta", lambda self: None)


def _build_model(ep_size: int) -> MoE:
    config = MoEConfig(
        vocab_size=1024,
        max_position_embeddings=256,
        pad_token_id=0,
        eos_token_id=0,
        num_hidden_layers=2,
        hidden_size=HIDDEN,
        intermediate_size=256,
        rms_norm_eps=1e-6,
        rope_theta=1e6,
        hidden_act="silu",
        attention=MHAConfig(num_attention_heads=4, num_key_value_heads=4, head_dim=32),
        tie_word_embeddings=False,
        n_routed_experts=N_EXPERTS,
        n_shared_experts=1,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        hidden_factor=1.0,
        moe_intermediate_size=MOE_INTER,
        router=NoAuxRouterConfig(
            scoring_func="sigmoid",
            router_scaling_factor=1.0,
            n_group=1,
            topk_group=1,
            norm_topk_prob=True,
        ),
        compile_cfg=False,
        ep_size=ep_size,
        dispatcher="all2all",
        mesh_prefix=PREFIX,
    )
    with torch.device("meta"):
        return MoE(config)


def _shard_model(ep_size: int, **fsdp_kwargs) -> MoE:
    model = _build_model(ep_size)
    fsdp_config = FSDPConfig(ep_size=ep_size, torch_compile=False, **fsdp_kwargs)
    return model.fully_shard(fsdp_config)


def _named_params(model: MoE) -> dict[str, DTensor]:
    params: dict[str, DTensor] = {}
    for name, param in model.named_parameters():
        params[name.replace("_checkpoint_wrapped_module.", "")] = param
    return params


def _ranks(mesh: DeviceMesh) -> list:
    return mesh.mesh.tolist()


def _legacy_ep_ranks(world_size: int, ep_size: int, rank: int) -> list[int]:
    start = (rank // ep_size) * ep_size
    return list(range(start, start + ep_size))


def _legacy_fsdp_ranks(world_size: int, ep_size: int, rank: int) -> list[int]:
    return [rank % ep_size + k * ep_size for k in range(world_size // ep_size)]


def _assert_placements(param: DTensor, mesh_dim_names: tuple[str, ...], placements: tuple) -> None:
    assert isinstance(param, DTensor)
    assert param.device_mesh.mesh_dim_names == mesh_dim_names
    assert len(param.placements) == len(placements)
    for actual, expected in zip(param.placements, placements):
        assert type(actual) is type(expected), (actual, expected)
        if isinstance(expected, _StridedShard):
            assert actual.dim == expected.dim and actual.split_factor == expected.split_factor
        elif isinstance(expected, Shard):
            assert actual.dim == expected.dim


class TestLegacyMoEMeshLayout:
    """Layout produced by the original EP-orthogonal-to-FSDP path.

    ``decouple_ep_fsdp`` is off by default, so these expectations are the
    regression contract for the legacy path: ``root = (fsdp = world / ep, ep)``
    with ``ep`` on the innermost (contiguous-rank) dimension.
    """

    @pytest.mark.parametrize(
        "world_size,ep_size,rank",
        [(8, 8, 0), (8, 8, 5), (8, 4, 6), (8, 2, 3), (16, 8, 9), (64, 8, 37)],
    )
    def test_mesh_ranks(self, world_size: int, ep_size: int, rank: int) -> None:
        with _fake_world(world_size, rank):
            model = _shard_model(ep_size)

            assert model.hsdp_mesh is None
            assert model._world_mesh is not None
            assert tuple(model._world_mesh.shape) == (world_size // ep_size, ep_size)
            assert model._world_mesh.mesh_dim_names == (f"{PREFIX}.fsdp", f"{PREFIX}.ep")

            assert model.fsdp_mesh is not None and model.ep_mesh is not None
            assert model.fsdp_mesh.mesh_dim_names == (f"{PREFIX}.fsdp",)
            assert _ranks(model.fsdp_mesh) == _legacy_fsdp_ranks(world_size, ep_size, rank)
            assert model.ep_mesh.mesh_dim_names == (f"{PREFIX}.ep",)
            assert _ranks(model.ep_mesh) == _legacy_ep_ranks(world_size, ep_size, rank)
            assert dist.get_process_group_ranks(model.ep_mesh.get_group()) == _legacy_ep_ranks(
                world_size, ep_size, rank
            )

    @pytest.mark.parametrize("world_size,ep_size,rank", [(8, 8, 0), (16, 8, 9), (64, 8, 37)])
    def test_param_placements(self, world_size: int, ep_size: int, rank: int) -> None:
        fsdp_size = world_size // ep_size
        with _fake_world(world_size, rank):
            model = _shard_model(ep_size)
            params = _named_params(model)
            mesh_dim_names = (f"{PREFIX}.fsdp", f"{PREFIX}.ep")

            for name in DENSE_PARAMS:
                param = params[name]
                # Dense params are replicated over ep and only sharded world/ep ways.
                _assert_placements(param, mesh_dim_names, (Shard(0), Replicate()))
                assert param.to_local().shape[0] == param.shape[0] // fsdp_size, name

            for name in EXPERT_PARAMS:
                param = params[name]
                _assert_placements(param, mesh_dim_names, (_StridedShard(0, split_factor=ep_size), Shard(0)))
                assert param.to_local().shape[0] == param.shape[0] // world_size, name

    def test_ep1_is_plain_fsdp(self) -> None:
        world_size, rank = 8, 3
        with _fake_world(world_size, rank):
            model = _shard_model(ep_size=1)
            params = _named_params(model)

            assert model.ep_mesh is not None and model.ep_mesh.size() == 1
            assert model.fsdp_mesh is not None and _ranks(model.fsdp_mesh) == list(range(world_size))
            for name in DENSE_PARAMS + EXPERT_PARAMS:
                param = params[name]
                _assert_placements(param, (f"{PREFIX}.fsdp",), (Shard(0),))
                assert param.to_local().shape[0] == param.shape[0] // world_size, name

    def test_hsdp_ep1_layout(self) -> None:
        world_size, shard_size, rank = 16, 8, 11
        with _fake_world(world_size, rank):
            model = _shard_model(ep_size=1, hsdp_sharding_size=shard_size)
            params = _named_params(model)

            assert model.hsdp_mesh is not None
            assert tuple(model.hsdp_mesh.shape) == (world_size // shard_size, shard_size)
            assert model.hsdp_mesh.mesh_dim_names == (f"{PREFIX}.hsdp_replicate", f"{PREFIX}.hsdp_shard")
            assert model.fsdp_mesh is not None
            assert _ranks(model.fsdp_mesh) == list(
                range((rank // shard_size) * shard_size, (rank // shard_size + 1) * shard_size)
            )
            for name in DENSE_PARAMS + EXPERT_PARAMS:
                param = params[name]
                _assert_placements(param, model.hsdp_mesh.mesh_dim_names, (Replicate(), Shard(0)))
                assert param.to_local().shape[0] == param.shape[0] // shard_size, name

    def test_hsdp_with_ep_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="HSDP requires expert parallel size to be 1"):
            FSDPConfig(ep_size=8, hsdp_sharding_size=8)


def _decoupled_ranks(world_size: int, ep_size: int, dp_shard: int, rank: int) -> dict[str, list]:
    replicate = world_size // dp_shard
    efsdp = dp_shard // ep_size
    block_start = (rank // dp_shard) * dp_shard
    ep_start = (rank // ep_size) * ep_size
    efsdp_in_block = [(rank % ep_size) + k * ep_size for k in range(efsdp)]
    return {
        "ep": list(range(ep_start, ep_start + ep_size)),
        "efsdp": [block_start + r for r in efsdp_in_block],
        "dp_shard": list(range(block_start, block_start + dp_shard)),
        "hsdp": [[r * dp_shard + j for j in range(dp_shard)] for r in range(replicate)],
        "expert_hsdp": [[r * dp_shard + j for j in efsdp_in_block] for r in range(replicate)],
    }


class TestDecoupledMoEMeshShapes:
    """Mesh bookkeeping of the decoupled (dp2ep) path: ``root = (replicate, efsdp, ep)``."""

    @pytest.mark.parametrize(
        "world_size,ep_size,rank",
        [(8, 8, 5), (8, 4, 6), (16, 8, 9), (64, 8, 37), (64, 32, 50)],
    )
    def test_mesh_without_hsdp(self, world_size: int, ep_size: int, rank: int) -> None:
        expected = _decoupled_ranks(world_size, ep_size, world_size, rank)
        with _fake_world(world_size, rank):
            model = _build_model(ep_size)
            model._init_device_mesh(FSDPConfig(ep_size=ep_size, torch_compile=False, decouple_ep_fsdp=True))

            root = model._world_mesh
            assert root is not None
            assert tuple(root.shape) == (1, world_size // ep_size, ep_size)
            assert root.mesh_dim_names == (f"{PREFIX}.replicate", f"{PREFIX}.efsdp", f"{PREFIX}.ep")
            assert _mesh_resources.get_root_mesh(model.ep_mesh) is root
            assert _mesh_resources.get_root_mesh(model.fsdp_mesh) is root
            assert _mesh_resources.get_root_mesh(model.expert_fsdp_mesh) is root

            assert model.hsdp_mesh is None
            assert model.ep_mesh.mesh_dim_names == (f"{PREFIX}.ep",)
            assert _ranks(model.ep_mesh) == expected["ep"]
            assert dist.get_process_group_ranks(model.ep_mesh.get_group()) == expected["ep"]
            assert model.expert_fsdp_mesh.mesh_dim_names == (f"{PREFIX}.efsdp",)
            assert _ranks(model.expert_fsdp_mesh) == expected["efsdp"]
            assert model.fsdp_mesh.mesh_dim_names == (f"{PREFIX}.dp_shard",)
            assert _ranks(model.fsdp_mesh) == expected["dp_shard"]
            assert dist.get_process_group_ranks(model.fsdp_mesh.get_group()) == expected["dp_shard"]

    @pytest.mark.parametrize(
        "world_size,ep_size,dp_shard,rank",
        [(16, 8, 8, 11), (64, 8, 32, 37), (64, 8, 8, 3)],
    )
    def test_mesh_with_hsdp(self, world_size: int, ep_size: int, dp_shard: int, rank: int) -> None:
        expected = _decoupled_ranks(world_size, ep_size, dp_shard, rank)
        replicate = world_size // dp_shard
        with _fake_world(world_size, rank):
            model = _build_model(ep_size)
            model._init_device_mesh(
                FSDPConfig(ep_size=ep_size, torch_compile=False, decouple_ep_fsdp=True, hsdp_sharding_size=dp_shard)
            )

            root = model._world_mesh
            assert root is not None
            assert tuple(root.shape) == (replicate, dp_shard // ep_size, ep_size)
            assert _ranks(model.ep_mesh) == expected["ep"]
            assert _ranks(model.fsdp_mesh) == expected["dp_shard"]

            assert model.hsdp_mesh is not None
            assert model.hsdp_mesh.mesh_dim_names == (f"{PREFIX}.replicate", f"{PREFIX}.dp_shard")
            assert _ranks(model.hsdp_mesh) == expected["hsdp"]
            assert model.expert_fsdp_mesh.mesh_dim_names == (f"{PREFIX}.replicate", f"{PREFIX}.efsdp")
            assert _ranks(model.expert_fsdp_mesh) == expected["expert_hsdp"]
            assert _mesh_resources.get_root_mesh(model.hsdp_mesh) is root
            assert _mesh_resources.get_root_mesh(model.expert_fsdp_mesh) is root

    def test_config_requires_ep_to_divide_shard_size(self) -> None:
        FSDPConfig(ep_size=8, hsdp_sharding_size=8, decouple_ep_fsdp=True)
        FSDPConfig(ep_size=4, hsdp_sharding_size=8, decouple_ep_fsdp=True)
        with pytest.raises(ValidationError, match="divisible by `ep_size`"):
            FSDPConfig(ep_size=8, hsdp_sharding_size=4, decouple_ep_fsdp=True)
        with pytest.raises(ValidationError, match="divisible by `ep_size`"):
            FSDPConfig(ep_size=3, hsdp_sharding_size=8, decouple_ep_fsdp=True)


class TestDecoupledMoEParamPlacements:
    """DTensor placements after the two-level ``fully_shard`` of the decoupled path."""

    @pytest.mark.parametrize("world_size,ep_size,rank", [(8, 8, 0), (16, 8, 9), (64, 8, 37)])
    def test_param_placements(self, world_size: int, ep_size: int, rank: int) -> None:
        efsdp = world_size // ep_size
        with _fake_world(world_size, rank):
            model = _shard_model(ep_size, decouple_ep_fsdp=True)
            params = _named_params(model)

            for name in DENSE_PARAMS:
                param = params[name]
                # Dense params: plain FSDP over the full dp_shard mesh, no EP replication.
                _assert_placements(param, (f"{PREFIX}.dp_shard",), (Shard(0),))
                assert param.to_local().shape[0] == param.shape[0] // world_size, name

            for name in EXPERT_PARAMS:
                param = params[name]
                _assert_placements(
                    param,
                    (f"{PREFIX}.efsdp", f"{PREFIX}.ep"),
                    (_StridedShard(0, split_factor=ep_size), Shard(0)),
                )
                assert param.device_mesh.size(0) == efsdp
                assert param.to_local().shape[0] == param.shape[0] // world_size, name

            from torch.distributed.fsdp import FSDPModule

            layer = model.layers["1"]
            assert isinstance(layer.experts, FSDPModule)
            assert isinstance(layer, FSDPModule)
            assert not isinstance(model.layers["0"].mlp, FSDPModule)

    @pytest.mark.parametrize("world_size,ep_size,dp_shard,rank", [(16, 8, 8, 11), (64, 8, 32, 37)])
    def test_param_placements_with_hsdp(self, world_size: int, ep_size: int, dp_shard: int, rank: int) -> None:
        efsdp = dp_shard // ep_size
        with _fake_world(world_size, rank):
            model = _shard_model(ep_size, decouple_ep_fsdp=True, hsdp_sharding_size=dp_shard)
            params = _named_params(model)

            for name in DENSE_PARAMS:
                param = params[name]
                _assert_placements(param, (f"{PREFIX}.replicate", f"{PREFIX}.dp_shard"), (Replicate(), Shard(0)))
                assert param.to_local().shape[0] == param.shape[0] // dp_shard, name

            for name in EXPERT_PARAMS:
                param = params[name]
                _assert_placements(
                    param,
                    (f"{PREFIX}.replicate", f"{PREFIX}.efsdp", f"{PREFIX}.ep"),
                    (Replicate(), _StridedShard(0, split_factor=ep_size), Shard(0)),
                )
                assert param.device_mesh.size(1) == efsdp
                assert param.to_local().shape[0] == param.shape[0] // dp_shard, name

    def test_legacy_layout_is_untouched_by_default(self) -> None:
        # Same construction as the decoupled tests, but with the flag left at its default.
        world_size, ep_size, rank = 16, 8, 9
        with _fake_world(world_size, rank):
            model = _shard_model(ep_size)
            params = _named_params(model)
            assert model.expert_fsdp_mesh is None
            _assert_placements(
                params["layers.1.self_attn.q_proj.weight"], (f"{PREFIX}.fsdp", f"{PREFIX}.ep"), (Shard(0), Replicate())
            )
