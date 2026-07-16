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
        assert config.text_config.freeze_attention is False

        with pytest.raises(ValidationError):
            Qwen3_5_VLTextMoE35BA3BValueConfig(mesh_prefix="actor")  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            Qwen3_5_VLTextMoE35BA3BValueConfig(mtp_config={"num_layers": 1})  # type: ignore[arg-type]

    @pytest.mark.parametrize("freeze_attention", [False, True])
    def test_attention_freeze_is_value_model_specific_and_configurable(self, freeze_attention: bool) -> None:
        class DummyDecoderLayer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.self_attn = nn.Linear(4, 4, bias=False)
                self.mlp = nn.Linear(4, 4, bias=False)
                self.input_layernorm = nn.LayerNorm(4)

        model = object.__new__(Qwen3_5_VLTextMoEValueModel)
        nn.Module.__init__(model)
        model.config = Qwen3_5_VLTextMoE35BA3BValueConfig(freeze_attention=freeze_attention)
        model.layers = nn.ModuleDict({"0": DummyDecoderLayer(), "1": DummyDecoderLayer()})

        model._freeze_attention_modules()

        assert all(
            parameter.requires_grad == (not freeze_attention)
            for layer in model.layers.values()
            for parameter in layer.self_attn.parameters()
        )
        assert all(parameter.requires_grad for layer in model.layers.values() for parameter in layer.mlp.parameters())
        assert all(
            parameter.requires_grad
            for layer in model.layers.values()
            for parameter in layer.input_layernorm.parameters()
        )
        assert all(layer.self_attn.training for layer in model.layers.values())

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

    def test_fresh_actor_load_uses_small_normal_for_only_missing_value_head(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = object.__new__(Qwen3_5_VLTextMoEValueModel)
        nn.Module.__init__(model)
        model.config = Qwen3_5_VLTextMoE35BA3BValueConfig(hidden_size=4)
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

        with torch.random.fork_rng():
            expected = torch.empty_like(model.lm_head.weight)
            torch.manual_seed(0)
            torch.nn.init.normal_(expected, mean=0.0, std=1.0 / 5)
            torch.manual_seed(0)
            _, unloaded_keys, missing_keys = model.from_hf("actor-checkpoint")

        torch.testing.assert_close(model.lm_head.weight, expected)
        assert unloaded_keys == set()
        assert missing_keys == set()
