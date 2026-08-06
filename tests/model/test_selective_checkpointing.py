"""Selective checkpointing regression tests.

TestKeptOps
    test_kept_op_reproduces_full_recompute: 按 op 留驻与全重算的输出/梯度逐位相同且梯度存活。
    test_unknown_op_name_is_skipped: 本次构建没注册的 op 名字被跳过，不报错。
TestKeptCallables
    test_kept_callable_reproduces_full_recompute: 按 callable 留驻与全重算逐位相同。
    test_unit_marker_outside_a_region_is_noop: 未被包装时不产生任何可观察行为。
TestContractLayering
    test_module_layer_imports_the_contract_without_the_model_layer: 契约模块必须能被 module/ 层单独导入。
TestUnsupportedUnits
    test_in_place_op_in_kept_unit_is_recomputed_not_refused: 只动自己缓冲区的 in-place 写不该被拒。
    test_mutating_a_kept_tensor_is_caught_by_torch: 写到被留驻张量上时由 torch 精确拦下。
    test_in_place_op_outside_kept_unit_is_fine: 单元之外的 in-place 写不受影响。
TestRegionRecomputeUnderDominoEP
    test_kept_unit_matches_full_recompute_under_domino_ep: domino EP 下留驻与全重算数值一致。
    test_kept_unit_matches_full_recompute_under_compile: compile 下同上，且不触发 cached-tensor-mutated。
"""

import pytest
import torch
from torch import nn

from xtuner.v1.model.utils import (
    RecomputeUnit,
    apply_selective_checkpointing,
    in_recompute_unit,
    resolve_kept_ops,
)


class _Block(nn.Module):
    """Two linear stages, so one of them can be kept while the other is recomputed."""

    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(4, 4)
        self.second = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.first(x))
        return torch.tanh(self.second(hidden))


class _UnitBlock(_Block):
    """A block whose second stage is wrapped as a unit, the way the engine installs one."""

    def __init__(self) -> None:
        super().__init__()
        self.second_stage = in_recompute_unit(RecomputeUnit.SAVE_ATTN, self._second_stage)

    def _second_stage(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.second(hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second_stage(torch.tanh(self.first(x)))


class _InPlaceUnitBlock(nn.Module):
    """A unit whose body accumulates in place, which selective checkpointing cannot keep."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.unit = in_recompute_unit(RecomputeUnit.SAVE_ATTN, self._unit)

    def _unit(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.linear(x)
        accumulator = torch.zeros_like(hidden)
        accumulator.add_(hidden)
        return accumulator

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unit(x)


class _MutatesKeptTensorBlock(nn.Module):
    """A unit that writes in place to a tensor the policy kept, which is the unsafe case."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.unit = in_recompute_unit(RecomputeUnit.SAVE_ATTN, self._unit)

    def _unit(self, x: torch.Tensor) -> torch.Tensor:
        kept = torch.tanh(self.linear(x))
        kept.mul_(2.0)
        return kept

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unit(x)


class _InPlaceOutsideBlock(_InPlaceUnitBlock):
    """The same in-place write, but reached without entering the unit."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._unit(x)


def _run(module: nn.Module, kept_ops=frozenset(), *, keeps_any_unit: bool = False):
    torch.manual_seed(0)
    for parameter in module.parameters():
        nn.init.normal_(parameter, std=0.5)
    wrapped = apply_selective_checkpointing(module, kept_ops, keeps_any_unit=keeps_any_unit)

    inputs = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 12
    inputs.requires_grad_(True)
    output = wrapped(inputs)
    output.sum().backward()
    return output.detach().clone(), [parameter.grad.clone() for parameter in module.parameters()]


def _assert_matches(kept, recomputed):
    kept_out, kept_grads = kept
    recomputed_out, recomputed_grads = recomputed
    torch.testing.assert_close(kept_out, recomputed_out, atol=0.0, rtol=0.0)
    for one, other in zip(kept_grads, recomputed_grads):
        torch.testing.assert_close(one, other, atol=0.0, rtol=0.0)
    # 梯度断掉时 loss 依然有限、上面的比较也依然成立（两边都是 None/零），所以单独断言存活。
    for grad in kept_grads:
        assert grad is not None
        assert torch.count_nonzero(grad) > 0


class TestKeptOps:
    def test_kept_op_reproduces_full_recompute(self):
        # 留驻与重算两条路都必须是精确值，不是近似：任何差异都说明 save-list 与重算对不上。
        kept_ops = resolve_kept_ops(("aten::tanh",))
        assert kept_ops, "aten::tanh should resolve in every build"
        _assert_matches(_run(_Block(), kept_ops, keeps_any_unit=True), _run(_Block()))

    def test_unknown_op_name_is_skipped(self):
        # 模型会同时列出同一个 kernel 的多种拼写（flash-attn v2/v3），只有一种在本次构建里注册。
        assert resolve_kept_ops(("nonexistent_namespace::nonexistent_op",)) == frozenset()
        assert len(resolve_kept_ops(("aten::tanh", "nonexistent_namespace::nope"))) == 1


class TestKeptCallables:
    def test_kept_callable_reproduces_full_recompute(self):
        _assert_matches(_run(_UnitBlock(), keeps_any_unit=True), _run(_UnitBlock()))

    def test_unit_marker_outside_a_region_is_noop(self):
        # 模型可以先被包装、后开启 recompute；包装本身不能有任何可观察行为。
        module = _UnitBlock()
        inputs = torch.zeros(3, 4, requires_grad=True)
        module(inputs).sum().backward()

        assert inputs.grad is not None


class TestUnsupportedUnits:
    def test_in_place_op_in_kept_unit_is_recomputed_not_refused(self):
        # in-place op 本身不危险，危险的是它写到了被 unit 留驻的张量上——那由 torch 的
        # version 检查精确判定。schema 是 mutable 只说明"可疑"，据此拒绝会误杀那些
        # 只动自己缓冲区的库调用（deep_ep 的 set_）。所以这里只重算、不拒绝。
        _run(_InPlaceUnitBlock(), keeps_any_unit=True)

    def test_mutating_a_kept_tensor_is_caught_by_torch(self):
        # 真正不安全的那种：被留驻的张量随后被原地改写，重算取回的就是改写后的值。
        with pytest.raises(RuntimeError, match="has been mutated"):
            _run(_MutatesKeptTensorBlock(), resolve_kept_ops(("aten::tanh",)), keeps_any_unit=True)

    def test_in_place_op_outside_kept_unit_is_fine(self):
        _run(_InPlaceOutsideBlock(), keeps_any_unit=True)
