import torch
import torch.nn as nn

from xtuner.v1.model.adapter.lora import LoraConfig, LoraModel
from xtuner.v1.module.grouped_linear.moe_group_linear import GroupedLinear
from xtuner.v1.module.lora_linear.lora_grouped_linear import LoraGroupedLinear
from xtuner.v1.module.lora_linear.lora_linear import LoraLinear
from xtuner.v1.utils.load_spec import LoadEnum, LoadSpec


def test_lora_linear_merge_round_trip():
    base_layer = nn.Linear(4, 3, bias=False)
    lora_layer = LoraLinear(base_layer, rank=2, alpha=4)
    x = torch.randn(5, 4)

    torch.testing.assert_close(lora_layer(x), base_layer(x))

    nn.init.normal_(lora_layer.lora_A.weight)
    nn.init.normal_(lora_layer.lora_B.weight)
    expected = lora_layer(x)
    base_weight = base_layer.weight.detach().clone()

    lora_layer.merge_lora()
    torch.testing.assert_close(lora_layer(x), expected)
    lora_layer.unmerge_lora()
    torch.testing.assert_close(base_layer.weight, base_weight)


def test_lora_linear_can_initialize_after_meta_materialization():
    with torch.device("meta"):
        lora_layer = LoraLinear(nn.Linear(4, 3, bias=False), rank=2, alpha=4)

    lora_layer.to_empty(device="cpu")
    lora_layer.reset_parameters()
    assert torch.count_nonzero(lora_layer.lora_A.weight) > 0
    assert torch.count_nonzero(lora_layer.lora_B.weight) == 0


def test_lora_grouped_linear_merge_round_trip():
    base_layer = GroupedLinear(in_features=4, out_features=3, num_routed_experts=2)
    lora_layer = LoraGroupedLinear(base_layer, rank=2, alpha=4)

    nn.init.normal_(base_layer.weight)
    nn.init.normal_(lora_layer.lora_A.weight)
    nn.init.normal_(lora_layer.lora_B.weight)
    base_weight = base_layer.weight.detach().clone()
    a = lora_layer.lora_A.weight.view(2, 2, 4)
    b = lora_layer.lora_B.weight.view(2, 3, 2)
    expected = base_weight.view(2, 3, 4) + torch.bmm(b, a) * lora_layer.scale

    lora_layer.merge_lora()
    torch.testing.assert_close(base_layer.weight.view(2, 3, 4), expected)
    lora_layer.unmerge_lora()
    torch.testing.assert_close(base_layer.weight, base_weight)


class _ToyModel(nn.Module):
    fsdp_mesh = None

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.lm_head = nn.Linear(4, 4)
        self.forward_marker = object()

    def to_hf_key_list(self, key: str) -> list[str]:
        return [key]

    def _init_load_spec(self):
        self.load_spec_mapping = {}
        for name, param in self.state_dict().items():
            hf_key = self.to_hf_key_list(name)[0]
            self.load_spec_mapping[name] = LoadSpec(
                name=name,
                hf_keys=[hf_key],
                shape=tuple(param.shape),
                load_enum=LoadEnum.SAME,
            )

    @staticmethod
    def _clean_param_name(name: str) -> str:
        return name

    @staticmethod
    def _load_same_hf_param(param, load_spec, checkpoint_loader):
        tensor = checkpoint_loader.load(load_spec.hf_keys[0])
        if tensor is None:
            return load_spec.hf_keys.copy()
        with torch.no_grad():
            param.copy_(tensor)
        return []

    @staticmethod
    def _load_fused_hf_param(*_):
        raise AssertionError("Toy model has no fused parameters")

    @staticmethod
    def _load_shard_hf_param(*_):
        raise AssertionError("Toy model has no sharded parameters")

    def set_hf(self, hf_path):
        self.hf_path = hf_path

    def forwarded_method(self):
        return self.forward_marker


def test_lora_model_forwards_base_model_api_and_preserves_modules_to_save():
    base_model = _ToyModel()
    model = LoraModel(
        base_model,
        LoraConfig(target_modules=["q_proj", "lm_head"], modules_to_save=["lm_head"]),
    )

    assert isinstance(base_model.q_proj, LoraLinear)
    assert isinstance(base_model.lm_head, nn.Linear)
    assert all(param.requires_grad for param in base_model.lm_head.parameters())
    assert model.forwarded_method() is base_model.forward_marker


def test_adapter_save_load_round_trip(tmp_path):
    config = LoraConfig(target_modules=["q_proj"], modules_to_save=["lm_head"], base_model_name_or_path="base")
    model = LoraModel(_ToyModel(), config)
    nn.init.normal_(model.base_model.q_proj.lora_A.weight)
    nn.init.normal_(model.base_model.q_proj.lora_B.weight)
    nn.init.normal_(model.base_model.lm_head.weight)
    nn.init.normal_(model.base_model.lm_head.bias)
    model._save_hf(tmp_path)

    restored = LoraModel(_ToyModel(), config.model_copy(deep=True))
    restored._load_adapter(tmp_path, strict=True)

    expected = {name: param for name, param in model.base_model.named_parameters() if param.requires_grad}
    actual = {name: param for name, param in restored.base_model.named_parameters() if param.requires_grad}
    assert expected.keys() == actual.keys()
    for name in expected:
        torch.testing.assert_close(actual[name].bfloat16(), expected[name].bfloat16(), rtol=0, atol=0)
