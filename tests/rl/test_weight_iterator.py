from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import torch.distributed as dist
from torch import nn

from xtuner.v1.model.base import BaseModel, HFSaveCfg, XTunerBaseModelConfig
from xtuner.v1.rl.weight_update.data import RolloutWeightUpdateInfo, RolloutWeightUpdateTarget
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


def test_hf_weight_update_batches_have_one_dtype() -> None:
    model = MixedDtypeModel()
    rollout_info = RolloutWeightUpdateInfo(
        rollout_config=cast(Any, SimpleNamespace()),
        weight_update_targets=(),
        train_rank=0,
        transport_type="ipc",
        backend="pytorch",
    )
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
                endpoint_rank=0,
                update_ranks=tuple(range(max(rollout_ep, rollout_tp))),
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
