import os

import pytest
import torch
import torch.distributed as dist
from pydantic import ValidationError
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Shard as DTensorShard
from torch.distributed.tensor import distribute_tensor
from torch.distributed.tensor.placement_types import _StridedShard

from xtuner.v1.model.base import BaseModel, XTunerBaseModelConfig
from xtuner.v1.utils import load_spec as load_spec_module
from xtuner.v1.utils.load_spec import LoadSpec, ShardDescriptor, unshard_tensors_for_hf_save


@pytest.fixture(scope="module")
def single_rank_group() -> dist.ProcessGroup:
    # ShardDescriptor.group is typed as `dist.ProcessGroup`; Pydantic enforces
    # the isinstance check even with `arbitrary_types_allowed=True`, so schema
    # tests need a real (but minimal) process group. A single-rank gloo group
    # is sufficient and avoids any CUDA / multi-process plumbing.
    if not dist.is_initialized():
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29555")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    group = dist.group.WORLD
    assert group is not None
    return group


class TestLoadSpecSchema:
    """New-schema fields should describe layout without legacy dispatch state."""

    def test_same_unsharded_spec(self) -> None:
        spec = LoadSpec(
            name="layers.0.mlp.gate.weight",
            global_hf_keys=["model.layers.0.mlp.gate.weight"],
            global_shape=(128, 64),
        )

        assert spec.is_fused is False
        assert spec.is_sharded is False
        assert spec.fused_dim is None
        assert spec.shards == []
        assert spec.origin_shape is None
        assert spec.unpadded_global_shape == spec.global_shape

    def test_from_tensor_derives_plain_tensor_layout(self) -> None:
        spec = LoadSpec.from_tensor(
            name="layers.0.experts.fused_w1w3.weight",
            hf_keys=["k0", "k1"],
            tensor=torch.empty(128, 64),
            origin_shape=(120, 64),
        )

        assert spec.global_hf_keys == ["k0", "k1"]
        assert spec.global_shape == (128, 64)
        assert spec.fused_dim == 0
        assert spec.shards == []
        assert spec.origin_shape == (120, 64)

    def test_from_tensor_derives_dtensor_shards(self, single_rank_group: dist.ProcessGroup) -> None:
        assert single_rank_group is not None
        mesh = DeviceMesh("cpu", [0])
        tensor = distribute_tensor(torch.empty(128, 64), mesh, [DTensorShard(0)])

        spec = LoadSpec.from_tensor(name="layers.0.mlp.gate.weight", hf_keys=["gate"], tensor=tensor)

        assert spec.global_hf_keys == ["gate"]
        assert spec.global_shape == (128, 64)
        assert spec.fused_dim is None
        assert [(shard.dim, shard.start, shard.end) for shard in spec.shards] == [(0, 0, 128)]

    def test_dtensor_shards_follow_explicit_placement_order(self, single_rank_group: dist.ProcessGroup) -> None:
        class FakeDeviceMesh:
            shape = (2, 2)

            def size(self, mesh_dim: int) -> int:
                return self.shape[mesh_dim]

            def get_local_rank(self, mesh_dim: int) -> int:
                return (1, 0)[mesh_dim]

            def get_group(self, mesh_dim: int) -> dist.ProcessGroup:
                return single_rank_group

        class FakeDTensor:
            shape = (8,)
            placements = (_StridedShard(0, split_factor=2), DTensorShard(0))
            device_mesh = FakeDeviceMesh()

        shards = load_spec_module._dtensor_shards(FakeDTensor())  # type: ignore[arg-type]

        assert [(shard.dim, shard.start, shard.end) for shard in shards] == [(0, 0, 4), (0, 2, 4)]

    def test_fused_spec_requires_fused_dim(self) -> None:
        with pytest.raises(ValidationError, match="fused_dim"):
            LoadSpec(
                name="layers.0.mlp.fused_w1w3.weight",
                global_hf_keys=[
                    "model.layers.0.mlp.experts.0.gate_proj.weight",
                    "model.layers.0.mlp.experts.0.up_proj.weight",
                ],
                global_shape=(256, 64),
            )

    def test_multi_axis_shards_preserve_order(self, single_rank_group: dist.ProcessGroup) -> None:
        ep = ShardDescriptor(dim=0, start=64, end=128, group=single_rank_group)
        fsdp = ShardDescriptor(dim=0, start=16, end=32, group=single_rank_group)
        spec = LoadSpec(
            name="layers.0.experts.fused_w1w3.weight",
            global_hf_keys=[
                "model.layers.0.mlp.experts.0.gate_proj.weight",
                "model.layers.0.mlp.experts.0.up_proj.weight",
            ],
            global_shape=(256, 64),
            fused_dim=0,
            shards=[ep, fsdp],
        )

        assert [(shard.start, shard.end) for shard in spec.shards] == [(64, 128), (16, 32)]
        assert spec.is_fused is True
        assert spec.is_sharded is True

    def test_ordered_shard_bounds_are_validated(self, single_rank_group: dist.ProcessGroup) -> None:
        with pytest.raises(ValidationError, match="Invalid shard descriptor"):
            LoadSpec(
                name="layers.0.experts.fused_w1w3.weight",
                global_hf_keys=["model.layers.0.mlp.experts.0.gate_proj.weight"],
                global_shape=(128, 64),
                shards=[
                    ShardDescriptor(dim=0, start=64, end=128, group=single_rank_group),
                    ShardDescriptor(dim=0, start=65, end=80, group=single_rank_group),
                ],
            )

    def test_zero_size_dtensor_shards_are_valid(self, single_rank_group: dist.ProcessGroup) -> None:
        spec = LoadSpec(
            name="embeddings.cls_embedding",
            global_hf_keys=["embeddings.cls_embedding"],
            global_shape=(1, 1, 1024),
            shards=[ShardDescriptor(dim=0, start=1, end=1, group=single_rank_group)],
        )

        plan = spec.plan_hf_load()

        assert plan.zero_fill is True
        assert plan.hf_keys == []


class TestHFLoadPlan:
    """LoadSpec should derive HF read plans from shards only."""

    def test_fused_slice_selects_overlapping_hf_keys(self, single_rank_group: dist.ProcessGroup) -> None:
        spec = LoadSpec(
            name="layers.0.experts.fused_w1w3.weight",
            global_hf_keys=["k0", "k1", "k2", "k3"],
            global_shape=(400, 64),
            fused_dim=0,
            shards=[ShardDescriptor(dim=0, start=150, end=260, group=single_rank_group)],
        )

        plan = spec.plan_hf_load()

        assert plan.hf_keys == ["k1", "k2"]
        assert plan.fused_dim == 0
        assert [(load_slice.dim, load_slice.start, load_slice.end) for load_slice in plan.slices] == [(0, 50, 160)]
        assert not hasattr(plan, "loaded_shape")

    def test_non_fused_slice_keeps_single_hf_key(self, single_rank_group: dist.ProcessGroup) -> None:
        spec = LoadSpec(
            name="layers.0.self_attn.q_proj.weight",
            global_hf_keys=["q_proj"],
            global_shape=(128, 256),
            shards=[ShardDescriptor(dim=1, start=64, end=192, group=single_rank_group)],
        )

        plan = spec.plan_hf_load()

        assert plan.hf_keys == ["q_proj"]
        assert plan.fused_dim is None
        assert [(load_slice.dim, load_slice.start, load_slice.end) for load_slice in plan.slices] == [(1, 64, 192)]

    def test_origin_shape_clips_runtime_padding(self, single_rank_group: dist.ProcessGroup) -> None:
        spec = LoadSpec(
            name="layers.0.experts.fused_w1w3.weight",
            global_hf_keys=["k0", "k1", "k2", "k3"],
            global_shape=(480, 64),
            fused_dim=0,
            shards=[ShardDescriptor(dim=0, start=350, end=450, group=single_rank_group)],
            origin_shape=(400, 64),
        )

        plan = spec.plan_hf_load()

        assert plan.hf_keys == ["k3"]
        assert [(load_slice.dim, load_slice.start, load_slice.end) for load_slice in plan.slices] == [(0, 50, 100)]
        assert plan.zero_fill is False

    def test_origin_shape_returns_zero_fill_for_pad_only_rank(self, single_rank_group: dist.ProcessGroup) -> None:
        spec = LoadSpec(
            name="layers.0.experts.fused_w1w3.weight",
            global_hf_keys=["k0", "k1", "k2", "k3"],
            global_shape=(480, 64),
            fused_dim=0,
            shards=[ShardDescriptor(dim=0, start=420, end=480, group=single_rank_group)],
            origin_shape=(400, 64),
        )

        plan = spec.plan_hf_load()

        assert plan.zero_fill is True
        assert plan.hf_keys == []
        assert plan.slices == []


class TestHFSavePolicy:
    """HF save should preserve the old distributed write policy from the new schema."""

    def test_fused_keys_are_split_across_save_ranks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        model = BaseModel(XTunerBaseModelConfig())
        model.config.hf_save_cfg.max_save_rank = 4
        spec = LoadSpec(
            name="layers.0.experts.fused_w1w3.weight",
            global_hf_keys=[f"k{i}" for i in range(8)],
            global_shape=(800, 64),
            fused_dim=0,
        )

        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "get_world_size", lambda group=None: 8)

        expected_ranges = {
            0: (0, 2),
            1: (2, 4),
            2: (4, 6),
            3: (6, 8),
            4: (0, 0),
        }
        for rank, expected_range in expected_ranges.items():
            monkeypatch.setattr(dist, "get_rank", lambda group=None, rank=rank: rank)
            assert model._hf_save_key_range(spec.plan_hf_save(distributed_save=True)) == expected_range

    def test_preserved_fused_shard_exposes_local_hf_keys(self, single_rank_group: dist.ProcessGroup) -> None:
        spec = LoadSpec(
            name="layers.0.experts.fused_w1w3.weight",
            global_hf_keys=["k0", "k1", "k2", "k3"],
            global_shape=(400, 64),
            fused_dim=0,
            shards=[ShardDescriptor(dim=0, start=100, end=200, group=single_rank_group)],
        )

        save_plan = spec.plan_hf_save(preserve_process_group=single_rank_group)

        assert save_plan.preserves_shards is True
        assert save_plan.hf_keys == ["k1"]

    def test_preserved_fused_shard_must_align_with_hf_key_boundary(self, single_rank_group: dist.ProcessGroup) -> None:
        spec = LoadSpec(
            name="layers.0.experts.fused_w1w3.weight",
            global_hf_keys=["k0", "k1", "k2", "k3"],
            global_shape=(400, 64),
            fused_dim=0,
            shards=[ShardDescriptor(dim=0, start=50, end=150, group=single_rank_group)],
        )

        with pytest.raises(AssertionError, match="must align with HF key size"):
            spec.plan_hf_save(preserve_process_group=single_rank_group)


class TestHFSaveUnshardScheduler:
    """Save unshard should batch independent work without violating per-tensor dependencies."""

    @staticmethod
    def _patch_foreach_all_gather(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
        calls: list[dict[str, object]] = []

        def fake_foreach_all_gather(
            tensor_list: list[torch.Tensor],
            group: dist.ProcessGroup,
        ) -> list[list[torch.Tensor]]:
            calls.append(
                {
                    "group": group,
                    "shapes": [tuple(tensor.shape) for tensor in tensor_list],
                    "dtypes": [tensor.dtype for tensor in tensor_list],
                }
            )
            return [[tensor] for tensor in tensor_list]

        monkeypatch.setattr(load_spec_module, "foreach_all_gather", fake_foreach_all_gather)
        return calls

    def test_single_tensor_single_step(
        self, monkeypatch: pytest.MonkeyPatch, single_rank_group: dist.ProcessGroup
    ) -> None:
        calls = self._patch_foreach_all_gather(monkeypatch)
        spec = LoadSpec(
            name="layers.0.mlp.gate.weight",
            global_hf_keys=["gate"],
            global_shape=(4, 2),
            shards=[ShardDescriptor(dim=0, start=1, end=3, group=single_rank_group)],
        )

        output = unshard_tensors_for_hf_save(
            [torch.ones(2, 2)],
            [spec.plan_hf_save()],
        )

        assert [tuple(tensor.shape) for tensor in output] == [(4, 2)]
        assert [call["shapes"] for call in calls] == [[(4, 2)]]

    def test_same_group_same_dtype_tensors_are_batched(
        self, monkeypatch: pytest.MonkeyPatch, single_rank_group: dist.ProcessGroup
    ) -> None:
        calls = self._patch_foreach_all_gather(monkeypatch)
        specs = [
            LoadSpec(
                name="layers.0.mlp.gate.weight",
                global_hf_keys=["gate"],
                global_shape=(4, 2),
                shards=[ShardDescriptor(dim=0, start=1, end=3, group=single_rank_group)],
            ),
            LoadSpec(
                name="layers.0.mlp.up.weight",
                global_hf_keys=["up"],
                global_shape=(6, 2),
                shards=[ShardDescriptor(dim=0, start=2, end=5, group=single_rank_group)],
            ),
        ]

        output = unshard_tensors_for_hf_save(
            [torch.ones(2, 2), torch.ones(3, 2)],
            [spec.plan_hf_save() for spec in specs],
        )

        assert [tuple(tensor.shape) for tensor in output] == [(4, 2), (6, 2)]
        assert [call["shapes"] for call in calls] == [[(4, 2), (6, 2)]]

    def test_same_group_different_dtype_tensors_are_split(
        self, monkeypatch: pytest.MonkeyPatch, single_rank_group: dist.ProcessGroup
    ) -> None:
        calls = self._patch_foreach_all_gather(monkeypatch)
        specs = [
            LoadSpec(
                name="layers.0.mlp.gate.weight",
                global_hf_keys=["gate"],
                global_shape=(4, 2),
                shards=[ShardDescriptor(dim=0, start=1, end=3, group=single_rank_group)],
            ),
            LoadSpec(
                name="layers.0.mlp.up.weight",
                global_hf_keys=["up"],
                global_shape=(4, 2),
                shards=[ShardDescriptor(dim=0, start=1, end=3, group=single_rank_group)],
            ),
        ]

        output = unshard_tensors_for_hf_save(
            [torch.ones(2, 2, dtype=torch.float32), torch.ones(2, 2, dtype=torch.float64)],
            [spec.plan_hf_save() for spec in specs],
        )

        assert [tuple(tensor.shape) for tensor in output] == [(4, 2), (4, 2)]
        assert [call["dtypes"] for call in calls] == [[torch.float32], [torch.float64]]

    def test_multi_step_tensor_waits_for_previous_step(
        self, monkeypatch: pytest.MonkeyPatch, single_rank_group: dist.ProcessGroup
    ) -> None:
        calls = self._patch_foreach_all_gather(monkeypatch)
        specs = [
            LoadSpec(
                name="layers.0.experts.fused_w1w3.weight",
                global_hf_keys=["k0", "k1"],
                global_shape=(8, 2),
                fused_dim=0,
                shards=[
                    ShardDescriptor(dim=0, start=0, end=4, group=single_rank_group),
                    ShardDescriptor(dim=0, start=1, end=3, group=single_rank_group),
                ],
            ),
            LoadSpec(
                name="layers.0.mlp.gate.weight",
                global_hf_keys=["gate"],
                global_shape=(4, 2),
                shards=[ShardDescriptor(dim=0, start=1, end=3, group=single_rank_group)],
            ),
        ]

        output = unshard_tensors_for_hf_save(
            [torch.ones(2, 2), torch.ones(2, 2)],
            [spec.plan_hf_save() for spec in specs],
        )

        assert [tuple(tensor.shape) for tensor in output] == [(8, 2), (4, 2)]
        assert [call["shapes"] for call in calls] == [[(4, 2), (4, 2)], [(8, 2)]]


class TestBaseModelHFSave:
    """BaseModel save should preserve state semantics outside LoadSpec."""

    def test_non_dtensor_buffers_keep_runtime_dtype(self) -> None:
        class BufferModel(BaseModel):
            def __init__(self) -> None:
                super().__init__(XTunerBaseModelConfig())
                self.register_buffer("rotary_coef", torch.tensor([1.25], dtype=torch.float32), persistent=True)
                self._init_load_spec()

            def to_hf_key_list(self, key: str) -> list[str]:
                return [key]

        model = BufferModel()

        [(names, tensors)] = list(
            model._get_hf_param(model._load_spec_params(), dtype=torch.bfloat16, distributed_save=True)
        )

        assert names == ["rotary_coef"]
        assert tensors[0].dtype == torch.float32
        assert torch.equal(tensors[0], model.rotary_coef)
