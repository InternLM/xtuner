"""Gradient checkpointing regression tests.

TestCheckpointWrapper
    test_wrapper_is_transparent_to_state_dict_and_attributes: 包裹后参数名/state_dict/属性访问不变。
    test_non_tensor_signature_preserves_gradients: 关键字参数 + dict 返回值下梯度与不重算一致。
    test_wrapper_forwards_container_protocols: 被包裹模块的 len/iter/in/索引协议在包裹后仍可用。
    test_wrapper_does_not_claim_protocols_the_module_lacks: 被包裹模块没有的协议不会出现在包裹层上。
"""


import pytest
import torch
from torch import nn

from xtuner.v1.model.utils import apply_gradient_checkpointing


class _KeywordOnlyBlock(nn.Module):
    """A forward shape the legacy reentrant checkpoint could not support.

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
        # 非 tensor 签名下梯度必须与不重算完全一致；legacy reentrant 在这里会直接断梯度。
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

    def test_wrapper_forwards_container_protocols(self):
        # 特殊方法在类型上查找，__getattr__ 看不到，必须逐个转发；只转发 __getitem__ 时
        # iter() 会退化到序列协议，报出与真实原因无关的 "not subscriptable"。
        plain = _ContainerBlock()
        wrapped = apply_gradient_checkpointing(plain)

        assert len(wrapped) == 3
        assert list(wrapped) == list(plain)
        assert wrapped[0] is plain[0]
        assert plain[1] in wrapped

    def test_wrapper_does_not_claim_protocols_the_module_lacks(self):
        # 协议按被包裹类型逐个决定，不能无条件定义：`__len__` 一旦恒存在，`bool(wrapper)` 就会
        # 去调它，于是 `module or default`（nn.Module 恒为真）对任何非 Sized 模块都会抛错——
        # `BaseModel._fully_shard` 里的 `target = module or self` 正是这样被打挂的。
        wrapped = apply_gradient_checkpointing(_KeywordOnlyBlock())

        assert bool(wrapped) is True
        assert not hasattr(type(wrapped), "__len__")
        with pytest.raises(TypeError, match="CheckpointWrapper"):
            len(wrapped)
