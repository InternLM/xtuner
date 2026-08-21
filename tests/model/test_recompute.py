"""Activation checkpoint public-behavior regression tests."""

import pytest
import torch
from torch import nn
from torch.autograd.graph import saved_tensors_hooks

from xtuner.v1.model.utils import apply_activation_checkpointing
from xtuner.v1.utils import clean_param_name


class _KeywordOnlyBlock(nn.Module):
    """A forward shape that requires pytree adaptation with reentrant checkpointing.

    Tensors arrive nested in a dict and behind a keyword-only argument, and the result is returned
    as a dict rather than a tensor or a tuple of tensors.
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.tag = "block"

    def forward(self, inputs: dict[str, torch.Tensor], *, scale: float) -> dict[str, torch.Tensor]:
        return {"out": self.linear(inputs["x"]) * scale}


class _FlexibleBlock(nn.Module):
    """接受任意摆放的输入：位置的容器、字典、关键字参数，用来覆盖各种嵌套形状。"""

    def __init__(self) -> None:
        super().__init__()
        # 输入 4 维、输出 6 维：输出与输入形状不同，断言才不会把输出误当成输入。
        self.linear = nn.Linear(4, 6)

    def forward(self, inputs, *, scale: float, extra: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        tensors = list(inputs.values()) if isinstance(inputs, dict) else list(inputs)
        if extra is not None:
            tensors.append(extra)
        return {"out": sum(self.linear(t) * scale for t in tensors)}


class _GradModeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.grad_modes: list[bool] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.grad_modes.append(torch.is_grad_enabled())
        return self.linear(x)


class _ParameterOnlyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([2.0]))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.weight * inputs


class _ReservedNamesBlock(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        *,
        function: float,
        preserve_rng_state: float,
        context_fn: float,
    ) -> dict[str, torch.Tensor]:
        return {"out": x * function * preserve_rng_state * context_fn}


class _ChangingOutputBlock(nn.Module):
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor] | list[torch.Tensor]:
        output = x.square()
        return [output] if torch.is_grad_enabled() else {"out": output}


class TestCheckpointWrapper:
    def test_wrapper_is_transparent_to_state_dict_and_attributes(self):
        # 包裹层不能出现在参数名里，否则 checkpoint 的存/取与非重算模型不兼容。
        plain = _KeywordOnlyBlock()
        wrapped = apply_activation_checkpointing(_KeywordOnlyBlock())
        wrapped.load_state_dict(plain.state_dict())

        assert sorted(wrapped.state_dict()) == sorted(plain.state_dict())
        assert sorted(name for name, _ in wrapped.named_parameters()) == sorted(
            name for name, _ in plain.named_parameters()
        )
        assert torch.equal(wrapped.state_dict()["linear.weight"], plain.state_dict()["linear.weight"])
        assert wrapped.tag == "block"

    def test_checkpoint_is_fixed_reentrant(self):
        wrapped = apply_activation_checkpointing(_GradModeBlock())
        wrapped(torch.randn(2, 4, requires_grad=True)).sum().backward()

        # Reentrant checkpoint runs the original pass without a graph, then replays it with grad.
        assert wrapped.grad_modes == [False, True]

    def test_detached_inputs_still_update_module_parameters(self):
        # MTP may detach every backbone input, but replay must still produce module gradients.
        wrapped = apply_activation_checkpointing(_ParameterOnlyBlock())
        detached_inputs = torch.tensor([3.0])

        wrapped(detached_inputs).sum().backward()

        assert wrapped.weight.grad is not None
        torch.testing.assert_close(wrapped.weight.grad, torch.tensor([3.0]))
        assert detached_inputs.grad is None

    def test_checkpoint_option_names_are_forwarded_to_the_module(self):
        x = torch.tensor(2.0, requires_grad=True)
        wrapped = apply_activation_checkpointing(_ReservedNamesBlock())

        output = wrapped(
            x,
            function=3.0,
            preserve_rng_state=5.0,
            context_fn=7.0,
        )["out"]
        output.backward()

        assert output.item() == 210.0
        assert x.grad.item() == 105.0

    def test_preserve_rng_state_replays_the_same_dropout_mask(self):
        wrapped = apply_activation_checkpointing(nn.Dropout(p=0.5), preserve_rng_state=True)
        x = torch.ones(32, requires_grad=True)

        torch.manual_seed(0)
        output = wrapped(x)
        output.sum().backward()

        torch.testing.assert_close(x.grad, output)

    def test_non_tensor_signature_preserves_gradients(self):
        # 非 tensor 签名下梯度必须与不重算完全一致。
        torch.manual_seed(0)
        plain = _KeywordOnlyBlock()
        wrapped = apply_activation_checkpointing(_KeywordOnlyBlock())
        wrapped.load_state_dict(plain.state_dict())

        x = torch.randn(2, 4, requires_grad=True)
        plain({"x": x}, scale=2.0)["out"].square().sum().backward()
        baseline_input_grad, x.grad = x.grad.clone(), None

        wrapped({"x": x}, scale=2.0)["out"].square().sum().backward()

        assert torch.equal(x.grad, baseline_input_grad)
        assert torch.equal(wrapped.linear.weight.grad, plain.linear.weight.grad)

    def test_root_parameter_names_can_be_normalized(self):
        root = nn.Module()
        root.block = apply_activation_checkpointing(_KeywordOnlyBlock())

        names = {clean_param_name(name) for name, _ in root.named_parameters()}

        assert names == {"block.linear.weight", "block.linear.bias"}

    def test_output_structure_must_match_during_replay(self):
        wrapped = apply_activation_checkpointing(_ChangingOutputBlock())
        x = torch.randn(2, requires_grad=True)

        with pytest.raises(RuntimeError, match="different output PyTree structure"):
            wrapped(x)["out"].sum().backward()

    @pytest.mark.parametrize(
        "make_call",
        [
            pytest.param(lambda block, x: block([x], scale=2.0), id="nested-in-list"),
            pytest.param(lambda block, x: block({"x": x}, scale=2.0), id="nested-in-dict"),
            pytest.param(lambda block, x: block([], scale=2.0, extra=x), id="passed-by-keyword"),
        ],
    )
    def test_input_tensors_reach_the_ambient_saved_tensor_hooks(self, make_call):
        # 激活 offload 是靠外层 saved_tensors_hooks 拿到层输入的，而 checkpoint 只把**顶层**
        # tensor 参数包成 SavedVariable（构造它才会触发 hook）。所以嵌套在容器里、或走关键字
        # 传进来的 tensor 会一个 hook 都不经过——offload 静默空转，梯度却完全正确，没有任何
        # 现象能暴露它。这里直接断言 hook 收得到。
        packed: list[int] = []

        class _Record(saved_tensors_hooks):
            # 按 data_ptr 认张量，不按 shape：区域的输出很容易和输入同形，
            # 按 shape 断言会把输出当成输入，测试变成恒绿。
            def __init__(self) -> None:
                super().__init__(lambda t: (packed.append(t.data_ptr()), t)[1], lambda t: t)

        wrapped = apply_activation_checkpointing(_FlexibleBlock())
        x = torch.randn(2, 4, requires_grad=True)

        with _Record():
            make_call(wrapped, x)["out"].square().sum().backward()

        assert x.data_ptr() in packed
