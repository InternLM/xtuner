from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import torch.distributed as dist
from torch import nn

from xtuner.v1.float8 import Float8Config, ScalingGranularity
from xtuner.v1.float8.fsdp_utils import WeightWithDynamicTilewiseFloat8CastTensor
from xtuner.v1.float8.triton_kernels.per_block_quant_gemm import per_block_quant_torch
from xtuner.v1.model.base import BaseModel, HFSaveCfg, XTunerBaseModelConfig
from xtuner.v1.model.compose.base import BaseComposeConfig
from xtuner.v1.rl.weight_update.data import RolloutBackend, RolloutWeightUpdateInfo, RolloutWeightUpdateTarget
from xtuner.v1.rl.weight_update.weight_iterator import WeightIterator
from xtuner.v1.utils import get_device
from xtuner.v1.utils import load_spec as load_spec_module
from xtuner.v1.utils.load_spec import LoadSpec, ShardDescriptor


class MixedDtypeModel(BaseModel):
    def __init__(self) -> None:
        config = XTunerBaseModelConfig(
            hf_save_cfg=HFSaveCfg(fp32_keys_pattern=[r"fp32_weight"]),
        )
        super().__init__(config)
        self.bf16_weight = nn.Parameter(torch.ones(2, device=get_device(), dtype=torch.bfloat16))
        self.fp32_weight = nn.Parameter(torch.ones(2, device=get_device(), dtype=torch.float32))
        self._init_load_spec()

    def to_hf_key_list(self, key: str) -> list[str]:
        return [key]


class ExpertShardModel(BaseModel):
    def __init__(self, ep_group: dist.ProcessGroup, ep_rank: int) -> None:
        super().__init__(XTunerBaseModelConfig())
        self.experts = nn.Parameter(
            torch.arange(ep_rank * 4, (ep_rank + 1) * 4, device=get_device(), dtype=torch.bfloat16)
        )
        self.fsdp_config = SimpleNamespace(ep_size=2)
        self.ep_mesh = SimpleNamespace(size=lambda: 2, get_group=lambda: ep_group)
        self.load_spec_mapping = {
            "experts": LoadSpec(
                name="experts",
                global_hf_keys=[f"expert_{index}" for index in range(8)],
                global_shape=(8,),
                fused_dim=0,
                shards=[ShardDescriptor(dim=0, group=ep_group)],
                local_shape=(4,),
            )
        }

    def to_hf_key_list(self, key: str) -> list[str]:
        return [f"expert_{index}" for index in range(8)]


class DirectFloat8WeightModel(BaseModel):
    def __init__(self, master: torch.Tensor) -> None:
        super().__init__(
            XTunerBaseModelConfig(
                float8_cfg=Float8Config(scaling_granularity_gemm=ScalingGranularity.TILEWISE),
            )
        )
        layer = nn.Module()
        layer.weight = nn.Parameter(
            WeightWithDynamicTilewiseFloat8CastTensor(
                master,
                torch.float8_e4m3fn,
                tuple(master.shape),
            )
        )
        self.layers = nn.ModuleDict({"0": layer})
        self.fsdp_config = SimpleNamespace(ep_size=1)
        self._init_load_spec()

    def to_hf_key_list(self, key: str) -> list[str]:
        return [key]


def _single_rank_rollout_info(*, backend: RolloutBackend) -> RolloutWeightUpdateInfo:
    return RolloutWeightUpdateInfo(
        rollout_config=cast(Any, SimpleNamespace()),
        weight_update_targets=(
            RolloutWeightUpdateTarget(
                endpoint_rank=0,
                update_ranks=(0,),
                inference_engine_ranks=(0,),
                server_url="http://rollout",
                lifecycle_state="active",
            ),
        ),
        train_rank=0,
        transport_type="ipc",
        backend=backend,
    )


def test_hf_weight_update_batches_have_one_dtype() -> None:
    model = MixedDtypeModel()
    rollout_info = _single_rank_rollout_info(backend="pytorch")
    iterator = WeightIterator(
        config=SimpleNamespace(update_weight_bucket_size_in_gb=1, model_cfg=None),
        engine=SimpleNamespace(model=model),
        rollout_info=rollout_info,
        global_hf_keys_mapping_cache={},
    )

    batches = list(iterator.iter_hf_batches())

    assert all(len({tensor.dtype for tensor in batch.state_dict.values()}) == 1 for batch in batches)
    state_dict = {name: tensor for batch in batches for name, tensor in batch.state_dict.items()}
    assert set(state_dict) == {"bf16_weight", "fp32_weight"}
    assert state_dict["bf16_weight"].dtype == torch.bfloat16
    assert state_dict["fp32_weight"].dtype == torch.float32


def test_fp8_hf_weight_update_uses_bfloat16() -> None:
    model = MixedDtypeModel()
    model.config.float8_cfg = Float8Config(scaling_granularity_gemm=ScalingGranularity.TILEWISE)
    model.fsdp_config = SimpleNamespace(ep_size=1)
    rollout_info = _single_rank_rollout_info(backend="pytorch")
    iterator = WeightIterator(
        config=SimpleNamespace(update_weight_bucket_size_in_gb=1, model_cfg=None),
        engine=SimpleNamespace(model=model),
        rollout_info=rollout_info,
        global_hf_keys_mapping_cache={},
    )

    state_dict = {name: tensor for batch in iterator.iter_hf_batches() for name, tensor in batch.state_dict.items()}

    assert state_dict["bf16_weight"].dtype == torch.bfloat16
    assert state_dict["fp32_weight"].dtype == torch.float32


def test_direct_fp8_weight_update_quantizes_bfloat16() -> None:
    generator = torch.Generator().manual_seed(11)
    checkpoint = torch.randn(128, 128, generator=generator).to(torch.bfloat16)
    master = checkpoint.to(torch.float32) + torch.randn(128, 128, generator=generator) * 1e-3
    expected_data, expected_scale = per_block_quant_torch(master.to(torch.bfloat16))
    fp16_data, fp16_scale = per_block_quant_torch(master.to(torch.float16))
    assert not (torch.equal(fp16_data, expected_data) and torch.equal(fp16_scale, expected_scale))

    model = DirectFloat8WeightModel(master)
    rollout_info = _single_rank_rollout_info(backend="turbomind")
    iterator = WeightIterator(
        config=SimpleNamespace(model_cfg=model.config),
        engine=SimpleNamespace(model=model),
        rollout_info=rollout_info,
        global_hf_keys_mapping_cache={},
    )

    (batch,) = list(iterator.iter_layer_batches())

    assert torch.equal(batch.state_dict["model.layers.0.weight"], expected_data)
    assert torch.equal(batch.state_dict["model.layers.0.weight_scale_inv"], expected_scale)


@pytest.mark.parametrize(
    ("rollout_ep", "rollout_tp", "train_rank", "expected_indices"),
    [
        pytest.param(1, 2, 0, range(8), id="rollout-tp2-rank0"),
        pytest.param(1, 2, 1, range(8), id="rollout-tp2-rank1"),
        pytest.param(2, 1, 0, range(0, 4), id="rollout-ep2-rank0"),
        pytest.param(2, 1, 1, range(4, 8), id="rollout-ep2-rank1"),
        pytest.param(4, 1, 0, range(0, 2), id="rollout-ep4-rank0"),
        pytest.param(4, 1, 1, range(2, 4), id="rollout-ep4-rank1"),
        pytest.param(4, 1, 2, range(4, 6), id="rollout-ep4-rank2"),
        pytest.param(4, 1, 3, range(6, 8), id="rollout-ep4-rank3"),
    ],
)
def test_ipc_hf_weight_batches_follow_rollout_expert_topology(
    monkeypatch: pytest.MonkeyPatch,
    rollout_ep: int,
    rollout_tp: int,
    train_rank: int,
    expected_indices: range,
) -> None:
    ep_group = dist.ProcessGroup(dist.HashStore(), 0, 1)
    train_ep_rank = train_rank % 2
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 2 if group is ep_group else 1)
    monkeypatch.setattr(dist, "get_rank", lambda group=None: train_ep_rank if group is ep_group else train_rank)
    model = ExpertShardModel(ep_group, train_ep_rank)

    def gather_train_ep_shards(
        tensor_list: list[torch.Tensor],
        group: dist.ProcessGroup,
    ) -> list[list[torch.Tensor]]:
        assert group is ep_group
        return [
            [
                torch.arange(0, 4, device=tensor.device, dtype=tensor.dtype),
                torch.arange(4, 8, device=tensor.device, dtype=tensor.dtype),
            ]
            for tensor in tensor_list
        ]

    monkeypatch.setattr(load_spec_module, "foreach_all_gather", gather_train_ep_shards)
    rollout_info = RolloutWeightUpdateInfo(
        rollout_config=cast(
            Any,
            SimpleNamespace(
                expert_parallel_size=rollout_ep,
                tensor_parallel_size=rollout_tp,
            ),
        ),
        weight_update_targets=(
            RolloutWeightUpdateTarget(
                endpoint_rank=train_rank,
                update_ranks=tuple(range(max(rollout_ep, rollout_tp))),
                inference_engine_ranks=tuple(range(max(rollout_ep, rollout_tp))),
                server_url="http://rollout",
                lifecycle_state="active",
            ),
        ),
        train_rank=train_rank,
        transport_type="ipc",
        backend="pytorch",
    )
    iterator = WeightIterator(
        config=SimpleNamespace(update_weight_bucket_size_in_gb=1, model_cfg=None),
        engine=SimpleNamespace(model=model),
        rollout_info=rollout_info,
        global_hf_keys_mapping_cache={},
    )

    state_dict = {name: tensor for batch in iterator.iter_hf_batches() for name, tensor in batch.state_dict.items()}

    assert set(state_dict) == {f"expert_{index}" for index in expected_indices}
    for index in expected_indices:
        torch.testing.assert_close(
            state_dict[f"expert_{index}"].cpu(),
            torch.tensor([index], dtype=torch.bfloat16),
        )


def _bf16_param(values: list[int]) -> nn.Parameter:
    return nn.Parameter(torch.tensor(values, device=get_device(), dtype=torch.bfloat16))


class _DecoupledLanguageTower(BaseModel):
    """Language tower on the decoupled EP/FSDP layout as seen from rank 0.

    ``layers.0.experts.weight`` is EP-sharded and then FSDP-sharded on ``efsdp``; the dense
    parameters are FSDP-sharded on ``dp_shard``. Every local block holds ``[0, 1]``.
    """

    def __init__(self, groups: dict[str, dist.ProcessGroup], expert_fsdp_ndim: int) -> None:
        super().__init__(XTunerBaseModelConfig())
        layer = nn.Module()
        layer.attn = nn.Module()
        layer.attn.weight = _bf16_param([0, 1])
        layer.experts = nn.Module()
        layer.experts.weight = _bf16_param([0, 1])
        self.layers = nn.ModuleDict({"0": layer})
        self.norm = nn.Module()
        self.norm.weight = _bf16_param([0, 1])
        self.fsdp_mesh = SimpleNamespace(get_group=lambda: groups["dp_shard"])
        self.expert_fsdp_mesh = SimpleNamespace(ndim=expert_fsdp_ndim, get_group=lambda dim=None: groups["efsdp"])
        dense_shards = [ShardDescriptor(dim=0, group=groups["dp_shard"])]
        self.load_spec_mapping = {
            "layers.0.attn.weight": LoadSpec(
                name="layers.0.attn.weight",
                global_hf_keys=["attn"],
                global_shape=(8,),
                shards=dense_shards,
                local_shape=(2,),
            ),
            "layers.0.experts.weight": LoadSpec(
                name="layers.0.experts.weight",
                global_hf_keys=[f"expert_{index}" for index in range(8)],
                global_shape=(8,),
                fused_dim=0,
                shards=[ShardDescriptor(dim=0, group=groups["ep"]), ShardDescriptor(dim=0, group=groups["efsdp"])],
                local_shape=(2,),
            ),
            "norm.weight": LoadSpec(
                name="norm.weight",
                global_hf_keys=["norm"],
                global_shape=(8,),
                shards=dense_shards,
                local_shape=(2,),
            ),
        }

    def to_hf_key_list(self, key: str) -> list[str]:
        return [key]


class _WorldShardedTower(BaseModel):
    """Vision tower / projector stand-in: one parameter FSDP-sharded on the world mesh."""

    def __init__(self, block_name: str, world_group: dist.ProcessGroup, world_size: int) -> None:
        super().__init__(XTunerBaseModelConfig())
        block = nn.Module()
        block.weight = _bf16_param(list(range(8 // world_size)))
        setattr(self, block_name, block)
        self.fsdp_mesh = SimpleNamespace(get_group=lambda: world_group)
        self.load_spec_mapping = {
            f"{block_name}.weight": LoadSpec(
                name=f"{block_name}.weight",
                global_hf_keys=[block_name],
                global_shape=(8,),
                shards=[ShardDescriptor(dim=0, group=world_group)],
                local_shape=(8 // world_size,),
            )
        }

    def to_hf_key_list(self, key: str) -> list[str]:
        return [key]


class _ComposeModel(BaseModel):
    """Compose model stand-in: like ``BaseComposeModel.fully_shard`` it is wrapped on the world
    mesh and has neither an ``expert_fsdp_mesh`` nor a ``load_spec_mapping`` of its own."""

    def __init__(
        self,
        language_model: BaseModel,
        vision_tower: BaseModel,
        projector: BaseModel,
        world_group: dist.ProcessGroup,
    ) -> None:
        super().__init__(XTunerBaseModelConfig())
        # `iter_layer_batches` only inspects `isinstance(model.config, BaseComposeConfig)`.
        self.config = BaseComposeConfig.model_construct()
        self.language_model = language_model
        self.vision_tower = vision_tower
        self.multi_modal_projector = projector
        self.fsdp_mesh = SimpleNamespace(get_group=lambda: world_group)


class TestLayerBatchesGatherWithParamOwner:
    """``iter_layer_batches`` must gather every parameter with the module that owns it.

    Regression test for compose MoE models on the decoupled EP/FSDP layout with the IPC/Turbomind
    transport: the outer compose model is wrapped on the world mesh, so gathering with it kept the
    language tower's ``efsdp`` (and HSDP ``dp_shard``) shards local and streamed rank-local
    fragments instead of complete weights.
    """

    @pytest.mark.parametrize(
        ("world_size", "expert_fsdp_ndim"),
        [
            pytest.param(4, 1, id="ep2-efsdp2"),
            pytest.param(8, 2, id="hsdp-replicate2-ep2-efsdp2"),
        ],
    )
    def test_compose_layer_batches_are_gathered_per_owner(
        self,
        monkeypatch: pytest.MonkeyPatch,
        world_size: int,
        expert_fsdp_ndim: int,
    ) -> None:
        groups = {name: dist.ProcessGroup(dist.HashStore(), 0, 1) for name in ("world", "dp_shard", "ep", "efsdp")}
        group_ranks = {
            groups["world"]: list(range(world_size)),
            groups["dp_shard"]: [0, 1, 2, 3],
            groups["ep"]: [0, 1],
            groups["efsdp"]: [0, 2],
        }
        monkeypatch.setattr(
            dist,
            "get_world_size",
            lambda group=None: world_size if group is None else len(group_ranks[group]),
        )
        monkeypatch.setattr(dist, "get_rank", lambda group=None: 0)
        monkeypatch.setattr(dist, "get_process_group_ranks", lambda group: group_ranks[group])

        gathered_groups: list[dist.ProcessGroup] = []

        def gather_contiguous_blocks(
            tensor_list: list[torch.Tensor],
            group: dist.ProcessGroup,
        ) -> list[list[torch.Tensor]]:
            # Rank ``r`` of ``group`` holds the r-th contiguous block, each block the size of ours.
            gathered_groups.append(group)
            return [
                [tensor + rank * tensor.numel() for rank in range(len(group_ranks[group]))] for tensor in tensor_list
            ]

        monkeypatch.setattr(load_spec_module, "foreach_all_gather", gather_contiguous_blocks)

        language_model = _DecoupledLanguageTower(groups, expert_fsdp_ndim)
        vision_tower = _WorldShardedTower("patch", groups["world"], world_size)
        projector = _WorldShardedTower("proj", groups["world"], world_size)
        model = _ComposeModel(language_model, vision_tower, projector, groups["world"])
        rollout_info = RolloutWeightUpdateInfo(
            rollout_config=cast(Any, SimpleNamespace()),
            weight_update_targets=(
                RolloutWeightUpdateTarget(
                    endpoint_rank=0,
                    update_ranks=(0,),
                    inference_engine_ranks=(0,),
                    server_url="http://rollout",
                    lifecycle_state="active",
                ),
            ),
            train_rank=0,
            transport_type="ipc",
            backend="turbomind",
        )
        iterator = WeightIterator(
            config=SimpleNamespace(model_cfg=model.config),
            engine=SimpleNamespace(model=model),
            rollout_info=rollout_info,
            global_hf_keys_mapping_cache={},
        )

        state_dict = {
            name: tensor.cpu() for batch in iterator.iter_layer_batches() for name, tensor in batch.state_dict.items()
        }

        full = torch.arange(8, dtype=torch.bfloat16)
        expected = {
            # The EP shard stays local (experts 0-3 of this EP rank); only the `efsdp` shard is gathered.
            "model.language_model.layers.0.mlp.experts.weight": full[:4],
            "model.language_model.layers.0.attn.weight": full,
            "model.language_model.norm.weight": full,
            "model.vision_tower.patch.weight": full,
            "model.multi_modal_projector.proj.weight": full,
        }
        assert set(state_dict) == set(expected)
        for name, tensor in expected.items():
            torch.testing.assert_close(state_dict[name], tensor, rtol=0, atol=0)
        assert {groups["efsdp"], groups["dp_shard"], groups["world"]} <= set(gathered_groups)
        assert groups["ep"] not in gathered_groups
