"""选择性重算（SAC）的回归测试。

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
TestRecomputeIsObservable
    test_recompute_reproduces_the_plain_gradients: 重算与不重算的梯度逐位相同。
    test_checkpointing_actually_recomputes: 重算确实发生——op 执行次数翻倍。
    test_a_kept_op_is_not_recomputed: 被留驻的 op 不参与重算，计数不翻倍。
TestRegionRecomputeUnderDominoEP
    test_kept_unit_matches_full_recompute_under_domino_ep: domino EP 下留驻与全重算数值一致。
    test_kept_unit_matches_full_recompute_under_compile: compile 下同上，且不触发 cached-tensor-mutated。
"""

import subprocess
import sys
from collections import Counter

import pytest
import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from xtuner.v1.model.utils import (
    KeptOps,
    SaveUnit,
    apply_selective_checkpointing,
    in_recompute_unit,
    resolve_kept_ops,
)


class _Block(nn.Module):
    """两级 linear，方便留驻其中一级、重算另一级。"""

    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(4, 4)
        self.second = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.first(x))
        return torch.tanh(self.second(hidden))


class _UnitBlock(_Block):
    """第二级被包成一个 unit，形态与引擎实际装上去的一致。"""

    def __init__(self) -> None:
        super().__init__()
        self.second_stage = in_recompute_unit(SaveUnit.ATTN, self._second_stage)

    def _second_stage(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.second(hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second_stage(torch.tanh(self.first(x)))


class _InPlaceUnitBlock(nn.Module):
    """unit 内部做原地累加——这种写法不能被留驻。"""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.unit = in_recompute_unit(SaveUnit.ATTN, self._unit)

    def _unit(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.linear(x)
        accumulator = torch.zeros_like(hidden)
        accumulator.add_(hidden)
        return accumulator

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unit(x)


class _MutatesKeptTensorBlock(nn.Module):
    """unit 原地改写了一个已被留驻的张量，这才是真正不安全的情形。"""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.unit = in_recompute_unit(SaveUnit.ATTN, self._unit)

    def _unit(self, x: torch.Tensor) -> torch.Tensor:
        kept = torch.tanh(self.linear(x))
        kept.mul_(2.0)
        return kept

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unit(x)


class _InPlaceOutsideBlock(_InPlaceUnitBlock):
    """同样的原地写，但没有进入 unit。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._unit(x)


class _OpCounter(TorchDispatchMode):
    """直接数每个 op 实际执行了多少次。

    这是判断「重算/SAC 到底有没有生效」最直接的观察点：不看 policy 返回了什么，只看 op 跑了几遍。
    前向跑一遍、重算再跑一遍，所以被重算的 op 计数翻倍；被留驻的 op 不参与重算，计数保持一遍。
    """

    def __init__(self) -> None:
        super().__init__()
        self.counts: Counter = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.counts[func] += 1
        return func(*args, **(kwargs or {}))


def _count_ops(make_module, kept_ops=frozenset(), *, checkpointed: bool, keeps_any_unit: bool = False):
    """跑一次前反向，返回 (op 计数, 输入梯度, 权重梯度)。

    收工厂而不是收实例：权重必须在同一个 seed 下构造，否则比较的是两组不同的随机权重。
    """
    torch.manual_seed(0)
    module = make_module()
    target = apply_selective_checkpointing(module, kept_ops, keeps_any_unit=keeps_any_unit) if checkpointed else module
    x = torch.randn(2, 4, requires_grad=True)

    counter = _OpCounter()
    with counter:
        target(x).square().sum().backward()
    return counter.counts, x.grad.clone(), module.first.weight.grad.clone()


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


class TestContractLayering:
    def test_module_layer_imports_the_contract_without_the_model_layer(self):
        # 契约之所以在 xtuner/v1/utils 而不是挨着 engine，是因为它命名的 callable 在
        # xtuner/v1/module 里：一旦契约里出现指向 model/ 的 import，这条独立导入就会变成
        # 循环导入而失败。必须用干净的解释器：同进程里 xtuner.v1.model 早就被导入了。
        result = subprocess.run(
            [sys.executable, "-c", "import xtuner.v1.module.decoder_layer.moe_decoder_layer"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr


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


class TestDeclarations:
    def test_kept_ops_and_callables_are_distinct_targets(self):
        # 两种解析方式的代价完全不同（一个零编译代价，一个要退出编译集合），所以类型必须能区分。
        from xtuner.v1.model.moe.moe import MOE_RECOMPUTE_CFG

        assert isinstance(MOE_RECOMPUTE_CFG[SaveUnit.ATTN], KeptOps)
        assert set(MOE_RECOMPUTE_CFG) == {
            SaveUnit.ATTN,
            SaveUnit.MOE_GATE,
            SaveUnit.MOE_DISPATCH,
        }


class TestRecomputeIsObservable:
    """按 reviewer 的三个目标直接观察：精度对齐、重算生效、重算+SAC 生效。"""

    def test_recompute_reproduces_the_plain_gradients(self):
        # 目标 1：精度与不重算完全一致（逐位，不给容差）。
        _, plain_x, plain_w = _count_ops(_Block, checkpointed=False)
        _, ckpt_x, ckpt_w = _count_ops(_Block, checkpointed=True)

        assert torch.equal(plain_x, ckpt_x)
        assert torch.equal(plain_w, ckpt_w)

    def test_checkpointing_actually_recomputes(self):
        # 目标 2：重算真的发生了——前向的 op 在 backward 里又跑了一遍，计数翻倍。
        plain, _, _ = _count_ops(_Block, checkpointed=False)
        ckpt, _, _ = _count_ops(_Block, checkpointed=True)

        tanh = torch.ops.aten.tanh.default
        assert plain[tanh] == 2, plain[tanh]
        assert ckpt[tanh] == 2 * plain[tanh]

    def test_a_kept_op_is_not_recomputed(self):
        # 目标 3：SAC 生效——被留驻的那个 op 不参与重算，计数回到只跑一遍。
        kept = resolve_kept_ops(("aten::tanh",))
        ckpt, _, _ = _count_ops(_Block, checkpointed=True)
        sac, sac_x, sac_w = _count_ops(_Block, kept, checkpointed=True, keeps_any_unit=True)

        tanh = torch.ops.aten.tanh.default
        assert ckpt[tanh] == 4
        assert sac[tanh] == 2, "留驻的 op 不该在重算里再跑一遍"

        # 少跑不等于算对，所以顺带把梯度也对一遍。
        _, plain_x, plain_w = _count_ops(_Block, checkpointed=False)
        assert torch.equal(plain_x, sac_x)
        assert torch.equal(plain_w, sac_w)
