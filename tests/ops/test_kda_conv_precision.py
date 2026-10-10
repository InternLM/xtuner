# Copyright (c) OpenMMLab. All rights reserved.
"""KDA 卷积的计算精度与参数梯度精度。

TestKDAConvPrecision
    test_fp32_parameter_gradients_preserve_bf16_forward  前向不变，参数梯度在归约前保留 FP32
    test_compiled_fp32_parameter_gradients              compile 与 eager 的前反向一致
"""

import pytest
import torch
from fla.modules.conv import causal_conv1d as fla_causal_conv1d
from torch.nn import functional as F

from xtuner.v1.ops.kda.causal_conv1d import causal_conv1d


@pytest.mark.gpu
class TestKDAConvPrecision:
    """验证真实卷积入口的 BF16 前向和 FP32 权重/偏置梯度。"""

    @pytest.mark.parametrize("activation", [None, "silu"])
    def test_fp32_parameter_gradients_preserve_bf16_forward(self, activation: str | None) -> None:
        # 前向与原 BF16 算子逐位相同，dw/db 对照不提前舍入的独立公式。
        torch.manual_seed(128)
        x = torch.randn(1, 2048, 32, device="cuda", dtype=torch.bfloat16).requires_grad_()
        w = (torch.randn(32, 4, device="cuda") * 0.1).requires_grad_()
        b = (torch.randn(32, device="cuda") * 0.1).requires_grad_()
        offsets = torch.tensor([0, 769, 2048], device="cuda", dtype=torch.int32)
        dy = torch.randn_like(x)
        y, _ = causal_conv1d(x, w, b, activation=activation, cu_seqlens=offsets)
        baseline, _ = fla_causal_conv1d(
            x.detach(),
            w.detach().bfloat16(),
            b.detach().bfloat16(),
            activation=activation,
            cu_seqlens=offsets,
            backend="triton",
        )
        torch.testing.assert_close(y, baseline, atol=0, rtol=0)
        y.backward(dy)

        dw = torch.zeros_like(w)
        db = torch.zeros_like(b)
        for start, stop in [(0, 769), (769, 2048)]:
            chunk = x.detach()[:, start:stop].float()
            windows = F.pad(chunk.transpose(1, 2), (3, 0)).unfold(-1, 4, 1)
            dz = dy[:, start:stop].float().transpose(1, 2)
            if activation == "silu":
                # FLA backward recomputes the preactivation in x.dtype.
                z = (windows * w.detach().bfloat16().float()[None, :, None]).sum(-1)
                z = (z + b.detach().bfloat16().float()[None, :, None]).bfloat16().float()
                sigmoid = z.sigmoid()
                dz = dz * sigmoid * (1 + z * (1 - sigmoid))
            dw += (windows * dz[..., None]).sum((0, 2))
            db += dz.sum((0, 2))
        for actual, expected in [(w.grad, dw), (b.grad, db)]:
            assert actual is not None and actual.dtype == torch.float32
            relative_error = (actual - expected).norm() / expected.norm()
            assert relative_error < 2e-5, relative_error
            assert torch.any(actual != actual.bfloat16().float())

    def test_compiled_fp32_parameter_gradients(self) -> None:
        # fullgraph 编译必须保留 FP32 dw/db，且前反向与 eager 一致。
        torch.manual_seed(129)
        inputs = [
            torch.randn(1, 128, 32, device="cuda", dtype=torch.bfloat16),
            torch.randn(32, 4, device="cuda"),
            torch.randn(32, device="cuda"),
        ]

        def forward(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return causal_conv1d(x, w, b, activation="silu")[0]

        eager_inputs = [v.detach().clone().requires_grad_() for v in inputs]
        compiled_inputs = [v.detach().clone().requires_grad_() for v in inputs]
        eager = forward(*eager_inputs)
        compiled = torch.compile(forward, fullgraph=True)(*compiled_inputs)
        torch.testing.assert_close(compiled, eager, atol=0, rtol=0)
        dy = torch.randn_like(eager)
        eager.backward(dy)
        compiled.backward(dy)
        for left, right in zip(eager_inputs, compiled_inputs):
            torch.testing.assert_close(right.grad, left.grad, atol=0, rtol=0)
