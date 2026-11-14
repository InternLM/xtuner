# Copyright (c) OpenMMLab. All rights reserved.
"""Forward quantization parity and the grouped GEMM's existing STE backward."""

import pytest
import torch
import torch.nn.functional as F

from xtuner.v1.float8.triton_kernels.per_tile_quant import per_tile_quant


@pytest.mark.gpu
@pytest.mark.parametrize("rows", [1, 8, 128, 129])
@pytest.mark.parametrize("width", [2048, 4096])
def test_forward_quantization_matches_lmdeploy(rows: int, width: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    serving = pytest.importorskip("lmdeploy.pytorch.kernels.cuda.blocked_gemm_fp8")
    torch.manual_seed(138)
    x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
    actual, scale = per_tile_quant(x, use_reciprocal=True)
    expected, expected_scale = serving.per_token_group_quant_fp8(x, 128, scale_fmt=None)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(scale, expected_scale)


@pytest.mark.gpu
def test_midpoint_rounding_and_small_groups() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    x = torch.zeros(3, 128, device="cuda")
    x[0, :2] = torch.tensor([-1.6171875, 8.625], device="cuda")
    x[2].fill_(1e-8)
    original, old_scale = per_tile_quant(x)
    actual, scale = per_tile_quant(x, use_reciprocal=True)
    # The exact scaled value is -84, an FP8 midpoint. The FP32 rounding order matters.
    assert original[0, 0].float() == -80
    assert actual[0, 0].float() == -88
    assert torch.equal(scale[0], old_scale[0])
    assert torch.equal(actual[1].float(), torch.zeros_like(x[1]))
    assert torch.isfinite(actual.float()).all() and torch.isfinite(scale).all()
    assert torch.equal(scale[1:], torch.full_like(scale[1:], 1e-6 * (1 / 448)))


@pytest.mark.gpu
def test_reciprocal_quantization_compiles() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    def quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values, scales = per_tile_quant(x, use_reciprocal=True)
        return values.float(), scales

    x = torch.randn(129, 2048, device="cuda", dtype=torch.bfloat16)
    actual = torch.compile(quantize, fullgraph=True, backend="aot_eager")(x)
    for value, expected in zip(actual, quantize(x)):
        assert torch.equal(value, expected)


@pytest.mark.gpu
def test_grouped_gemm_nonzero_gradients_and_update() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pytest.importorskip("adaptive_gemm")
    from xtuner.v1.float8.float8_gmm_tile_wise import (
        fp8_gmm_weight_per_block_act_per_tile,
        weight_to_per_block_float8_dynamic,
    )

    torch.manual_seed(139)
    counts = torch.tensor([32, 0, 31], device="cuda", dtype=torch.int64)
    x = torch.randn(63, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(3, 128, 128, device="cuda", requires_grad=True)
    weight_fp8 = weight_to_per_block_float8_dynamic.apply(weight, torch.float8_e4m3fn, 128)
    actual = fp8_gmm_weight_per_block_act_per_tile.apply(x, weight_fp8, counts)
    upstream = torch.randn_like(actual)
    dx, dw = torch.autograd.grad(actual, (x, weight), upstream)

    # Independent dense same-math oracle; production retains its quantized STE.
    reference_x = x.detach().float().requires_grad_()
    reference_w = weight.detach().clone().requires_grad_()
    expected = torch.cat((F.linear(reference_x[:32], reference_w[0]), F.linear(reference_x[32:], reference_w[2])))
    oracle = torch.autograd.grad(expected, (reference_x, reference_w), upstream.float())
    for gradient, reference in zip((dx, dw), oracle):
        assert torch.isfinite(gradient).all() and gradient.norm() > 0
        error = (gradient.float() - reference).norm() / reference.norm()
        assert error < 0.1
        assert (1.25 * gradient.float() - reference).norm() / reference.norm() > 0.1
    assert torch.count_nonzero(dw[1]) == 0
    before = weight.detach().clone()
    weight.grad = dw
    torch.optim.SGD([weight], lr=0.01).step()
    torch.testing.assert_close(weight, before - 0.01 * dw, rtol=0, atol=1e-6)
    assert not torch.equal(weight, before)
