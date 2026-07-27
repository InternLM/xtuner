"""Gradient checkpointing and recompute-unit regression tests.

TestCheckpointWrapper
    test_wrapper_is_transparent_to_state_dict_and_attributes: 包裹后参数名/state_dict/属性访问不变。
    test_non_tensor_signature_preserves_gradients: 关键字参数 + dict 返回值下梯度与不重算一致。
    test_checkpointing_keeps_the_module_itself: 换类而非套壳，isinstance/属性/容器协议原生可用。
    test_module_without_a_protocol_does_not_gain_one: 被包裹模块没有的协议不会凭空出现。
    test_unset_cfg_keeps_full_recompute: `None` 不改变显存行为，解析为不留驻。
    test_true_selects_every_supported_unit: `True` 选中模型声明的全部 unit。
    test_explicit_units_select_only_themselves: 显式 list 只选中对应 unit。
    test_string_units_are_accepted: 配置文件里的字符串能解析成 RecomputeUnit。
    test_unsupported_unit_is_rejected: 模型不支持的 unit 在构造时报错并列出支持项。
    test_disable_propagates_into_nested_configs: `False` 递归关闭嵌套子模型配置。
    test_disable_reaches_every_sub_model_of_a_real_compose_config: 真实 compose 配置的三个子配置都被关闭。
    test_units_round_trip_through_json: enum 序列化成可读字符串并能读回。
    test_declared_targets_resolve: 声明表里的 op 名与 callable 名都能解析到真实对象。
    test_no_unit_names_the_method_that_holds_most_compilation: 没有 unit 点名承载最多编译的那个方法。
    test_an_op_identity_unit_costs_no_compilation: KeptOps 不改动编译集合。
    test_a_callable_unit_keeps_its_callers_compiled: KeptCallables 只退出自身，调用者仍编译。
    test_no_unit_withdraws_the_method_that_holds_most_compilation: 没有 unit 撤出编译占比最大的方法。
    test_attention_is_kept_by_op_identity: attention 走 op identity 而非撤出 callable。
"""



import torch
from torch import nn

from xtuner.v1.model.utils import apply_gradient_checkpointing


class _KeywordOnlyBlock(nn.Module):
    """A forward shape only the non-reentrant implementation supports.

    Tensors arrive nested in a dict and behind a keyword-only argument, and the result is returned
    as a dict rather than a tensor or a tuple of tensors.
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.tag = "block"

    def forward(self, inputs: dict[str, torch.Tensor], *, scale: float) -> dict[str, torch.Tensor]:
        return {"out": self.linear(inputs["x"]) * scale}


class _ContainerBlock(nn.Module):
    """A container module, the shape whose protocols the wrapper has to forward."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, index: int) -> nn.Module:
        return self.layers[index]

    def __iter__(self):
        return iter(self.layers)

    def __contains__(self, item: object) -> bool:
        return item in self.layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class TestCheckpointWrapper:
    def test_wrapper_is_transparent_to_state_dict_and_attributes(self):
        # 包裹层不能出现在参数名里，否则 checkpoint 的存/取与非重算模型不兼容。
        plain = _KeywordOnlyBlock()
        wrapped = apply_gradient_checkpointing(_KeywordOnlyBlock())
        wrapped.load_state_dict(plain.state_dict())

        assert sorted(wrapped.state_dict()) == sorted(plain.state_dict())
        assert sorted(name for name, _ in wrapped.named_parameters()) == sorted(
            name for name, _ in plain.named_parameters()
        )
        assert torch.equal(wrapped.state_dict()["linear.weight"], plain.state_dict()["linear.weight"])
        assert wrapped.tag == "block"

    def test_non_tensor_signature_preserves_gradients(self):
        # 非 tensor 签名下梯度必须与不重算完全一致。
        torch.manual_seed(0)
        plain = _KeywordOnlyBlock()
        wrapped = apply_gradient_checkpointing(_KeywordOnlyBlock())
        wrapped.load_state_dict(plain.state_dict())

        x = torch.randn(2, 4, requires_grad=True)
        plain({"x": x}, scale=2.0)["out"].square().sum().backward()
        baseline_input_grad, x.grad = x.grad.clone(), None

        wrapped({"x": x}, scale=2.0)["out"].square().sum().backward()

        assert torch.equal(x.grad, baseline_input_grad)
        assert torch.equal(wrapped.linear.weight.grad, plain.linear.weight.grad)

    def test_checkpointing_keeps_the_module_itself(self):
        # 不再套壳，而是把 mixin 插进模块自己的 MRO（同 fully_shard 的做法）：
        # isinstance 仍成立，属性、类属性、容器协议都原生可用，不需要任何转发。
        block = _ContainerBlock()
        checkpointed = apply_gradient_checkpointing(block)

        assert checkpointed is block
        assert isinstance(checkpointed, _ContainerBlock)
        assert len(checkpointed) == 3
        assert list(checkpointed) == list(block.layers)
        assert checkpointed[0] is block.layers[0]

    def test_module_without_a_protocol_does_not_gain_one(self):
        # 反面：被包裹模块没有的协议不能凭空出现。`__len__` 一旦恒存在，`bool(module)` 就会去调
        # 它，`module or default`（nn.Module 恒为真）会对任何非 Sized 模块抛错——
        # `BaseModel._fully_shard` 里的 `target = module or self` 正是这样被打挂过。
        checkpointed = apply_gradient_checkpointing(_KeywordOnlyBlock())

        assert bool(checkpointed) is True
        assert not hasattr(type(checkpointed), "__len__")
