"""KDA forget gate 的 FP32 参数梯度与编译精度。

TestKDAGatePrecision
    test_forward_and_gradients_match_reference  前向不变，输入、衰减参数和 bias 梯度匹配独立公式
    test_compile_preserves_forward_and_gradients  fullgraph 编译保持前向及全部梯度
"""

import pytest
import torch
import torch.nn.functional as F

from xtuner.v1.ops.kda.fused_kda_gate import fused_kda_gate


def _reference_gate(
    g: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    lower_bound: float | None,
) -> torch.Tensor:
    values = g.float()
    if dt_bias is not None:
        values = values + dt_bias.view(g.shape[-2:])
    decay = a_log.exp().view(-1, 1)
    if lower_bound is None:
        return -decay * F.softplus(values)
    return lower_bound * torch.sigmoid(decay * values)


def _inputs(dtype: torch.dtype, seq_len: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(17)
    g = (torch.randn(1, seq_len, 2, 16, device="cuda") - 0.5).to(dtype).requires_grad_()
    a_log = torch.tensor([-0.3, 0.3], device="cuda", requires_grad=True)
    dt_bias = (torch.randn(32, device="cuda") * 0.2 - 0.5).requires_grad_()
    return g, a_log, dt_bias


class TestKDAGatePrecision:
    """验证低精度投影不会使 FP32 gate 参数梯度在归约前舍入。"""

    @pytest.mark.gpu
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
    @pytest.mark.parametrize("lower_bound", [-5.0, None], ids=["safe", "softplus"])
    @pytest.mark.parametrize("with_bias", [True, False], ids=["bias", "no_bias"])
    def test_forward_and_gradients_match_reference(
        self, dtype: torch.dtype, lower_bound: float | None, with_bias: bool
    ) -> None:
        # 真实 FLA 前向必须保持不变；FP32 bias 梯度应先归约，再独立于 BF16 输入梯度转换。
        from fla.ops.kda.gate import fused_kda_gate as fla_gate

        values = _inputs(dtype, seq_len=2048)
        inputs = values if with_bias else values[:2]
        reference_inputs = tuple(value.detach().clone().requires_grad_() for value in inputs)
        bias = inputs[2] if with_bias else None
        reference_bias = reference_inputs[2] if with_bias else None

        output = fused_kda_gate(inputs[0], inputs[1], bias, lower_bound)
        expected = _reference_gate(reference_inputs[0], reference_inputs[1], reference_bias, lower_bound)
        with torch.no_grad():
            original = fla_gate(inputs[0], inputs[1], bias, lower_bound)
        torch.testing.assert_close(output, original, atol=0, rtol=0)
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)

        upstream = torch.rand_like(output) + 0.5
        gradients = torch.autograd.grad(output, inputs, upstream)
        reference_gradients = torch.autograd.grad(expected, reference_inputs, upstream)
        for name, value, actual, reference in zip(["g", "A_log", "dt_bias"], inputs, gradients, reference_gradients):
            assert actual.dtype == value.dtype, name
            assert torch.isfinite(actual).all(), name
            rtol = 1e-2 if value.dtype == torch.bfloat16 else 1e-5
            torch.testing.assert_close(actual, reference, atol=1e-6, rtol=rtol, msg=name)

    @pytest.mark.gpu
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
    def test_compile_preserves_forward_and_gradients(self, dtype: torch.dtype) -> None:
        # 实际 fullgraph 编译包含外部 dtype 转换和自定义反向，三种梯度均与 eager 一致。
        inputs = _inputs(dtype, seq_len=257)
        compiled_inputs = tuple(value.detach().clone().requires_grad_() for value in inputs)
        compiled_gate = torch.compile(fused_kda_gate, fullgraph=True)
        expected = fused_kda_gate(*inputs, lower_bound=-5.0)
        output = compiled_gate(*compiled_inputs, lower_bound=-5.0)
        torch.testing.assert_close(output, expected, atol=0, rtol=0)

        upstream = torch.rand_like(output) + 0.5
        expected_gradients = torch.autograd.grad(expected, inputs, upstream)
        gradients = torch.autograd.grad(output, compiled_inputs, upstream)
        for actual, reference in zip(gradients, expected_gradients):
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)
