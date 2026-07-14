from pathlib import Path

import pytest
import torch
from pydantic import ValidationError
from torch import nn

from xtuner.v1.model import Qwen3_5_VLMoE35BA3ValueConfig
from xtuner.v1.model.moe.qwen3_5_text import Qwen3_5_VLTextMoE
from xtuner.v1.model.moe.qwen3_5_value import (
    Qwen3_5_VLTextMoE35BA3BValueConfig,
    Qwen3_5_VLTextMoEValueModel,
)


class TestQwen3_5ValueModel:
    def test_config_forces_critic_mesh_and_disables_mtp(self) -> None:
        config = Qwen3_5_VLMoE35BA3ValueConfig()

        assert config.text_config.mesh_prefix == "critic"
        assert config.text_config.mtp_config is None

        with pytest.raises(ValidationError):
            Qwen3_5_VLTextMoE35BA3BValueConfig(mesh_prefix="actor")  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            Qwen3_5_VLTextMoE35BA3BValueConfig(mtp_config={"num_layers": 1})  # type: ignore[arg-type]

    def test_scalar_head_has_unbounded_linear_output(self) -> None:
        config = Qwen3_5_VLTextMoE35BA3BValueConfig()
        model = object.__new__(Qwen3_5_VLTextMoEValueModel)
        head = model.build_head(config)
        hidden_states = torch.zeros(1, 1, config.hidden_size)
        hidden_states[..., :2] = torch.tensor([1.0, -2.0])

        with torch.no_grad():
            head.weight.zero_()
            head.weight[..., :2] = 1.0

        loss, (values, extra_info) = head(hidden_states)

        assert loss is None
        assert extra_info == {}
        assert values is not None
        assert values.shape == (1, 1, 1)
        assert values.item() == -1.0
        assert head.out_features == 1
        assert head.bias is None
        assert model.to_hf_key_list("lm_head.weight") == ["value_head.weight"]

    def test_fresh_actor_load_zero_initializes_only_missing_value_head(self, monkeypatch: pytest.MonkeyPatch) -> None:
        model = object.__new__(Qwen3_5_VLTextMoEValueModel)
        nn.Module.__init__(model)
        model.lm_head = nn.Linear(4, 1, bias=False)
        with torch.no_grad():
            model.lm_head.weight.fill_(1.0)

        def fake_from_hf(
            _self: Qwen3_5_VLTextMoE,
            _hf_path: str | Path,
            strict: bool = True,
        ) -> tuple[set[str], set[str], set[str]]:
            del strict
            return set(), {"lm_head.weight"}, {"value_head.weight"}

        monkeypatch.setattr(Qwen3_5_VLTextMoE, "from_hf", fake_from_hf)

        _, unloaded_keys, missing_keys = model.from_hf("actor-checkpoint")

        assert torch.count_nonzero(model.lm_head.weight) == 0
        assert unloaded_keys == set()
        assert missing_keys == set()
