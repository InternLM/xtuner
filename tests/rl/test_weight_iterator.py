from types import SimpleNamespace
from typing import Any, cast

import torch
from torch import nn

from xtuner.v1.model.base import BaseModel, HFSaveCfg, XTunerBaseModelConfig
from xtuner.v1.rl.weight_update.data import RolloutWeightUpdateInfo
from xtuner.v1.rl.weight_update.weight_iterator import WeightIterator
from xtuner.v1.utils import get_device


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
