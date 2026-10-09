"""原生 RMSNorm 在多维输入上的数值与编译后反向的开销。

TestNativeRMSNorm
    test_matches_float32_reference_on_3d_input: [B, S, H] 输入的输出、输入梯度和权重梯度与 float32 参考一致。
    test_compiled_backward_of_3d_input_costs_the_same_as_2d: 编译后 [1, S, H] 输入的反向不比 [S, H] 慢。

回归背景：Inductor 把 3D 输入的 dW 归约生成为每个 hidden 列一个 program、沿全部 token 按 hidden 跨步读，
[1, 65536, 6144] 上 13.4ms，而 2D 输入走两段式归约只要 0.93ms（GLM-5.2 512K 每步 11.1s）。
"""

import pytest
import torch
import torch.nn.functional as F

from xtuner.v1.ops.rms_norm import native_rms_norm


EPS = 1e-5


def _decoder_like(hidden: torch.Tensor, attn_out: torch.Tensor, weight: torch.Tensor, router: torch.Tensor):
    # 与解码层相同的图：残差相加 -> RMSNorm -> router 式消费者 -> 残差相加。
    hidden = hidden + attn_out
    normed = native_rms_norm(hidden, weight, EPS)
    return hidden + normed * F.linear(normed, router).sigmoid().mean(-1, keepdim=True)


def _backward_ms(fn, shape: tuple[int, ...], weight: torch.Tensor, router: torch.Tensor) -> float:
    torch.manual_seed(0)
    hidden = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    grad = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    times = []
    for _ in range(8):
        x, a = hidden.clone().requires_grad_(), hidden.clone().requires_grad_()
        out = fn(x, a, weight, router)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        out.backward(grad)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return min(times[3:])


class TestNativeRMSNorm:
    def test_matches_float32_reference_on_3d_input(self):
        # 验证 [B, S, H] 输入的输出和梯度与逐行 float32 公式一致，batch 维不会被混到一起。
        torch.manual_seed(0)
        x = torch.randn(2, 5, 8, dtype=torch.float32, requires_grad=True)
        weight = torch.rand(8, dtype=torch.float32).add_(0.5).requires_grad_()
        grad = torch.randn(2, 5, 8, dtype=torch.float32)
        x_ref = x.detach().clone().requires_grad_()
        weight_ref = weight.detach().clone().requires_grad_()

        out = native_rms_norm(x, weight, EPS)
        out.backward(grad)
        expected = x_ref * torch.rsqrt(x_ref.pow(2).mean(-1, keepdim=True) + EPS) * weight_ref
        expected.backward(grad)

        assert out.shape == x.shape
        torch.testing.assert_close(out, expected)
        torch.testing.assert_close(x.grad, x_ref.grad)
        torch.testing.assert_close(weight.grad, weight_ref.grad)

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_compiled_backward_of_3d_input_costs_the_same_as_2d(self):
        # 验证编译后 [1, S, H] 输入的反向与 [S, H] 同速；回归前 dW 归约让前者慢 3 倍以上。
        seq, hidden = 32768, 6144
        weight = torch.nn.Parameter(torch.linspace(0.5, 1.5, hidden, device="cuda", dtype=torch.bfloat16))
        router = torch.nn.Parameter(torch.randn(256, hidden, device="cuda", dtype=torch.bfloat16) * 0.01)

        ms_3d = _backward_ms(torch.compile(_decoder_like, dynamic=True), (1, seq, hidden), weight, router)
        ms_2d = _backward_ms(torch.compile(_decoder_like, dynamic=True), (seq, hidden), weight, router)

        assert ms_3d < 1.5 * ms_2d, f"3D backward {ms_3d:.2f} ms vs 2D {ms_2d:.2f} ms"
