# Copyright (c) OpenMMLab. All rights reserved.
import inspect

import pytest
import torch
import torch.nn.functional as F

from xtuner.v1.ops import act_fn


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel required")


@pytest.fixture
def lmdeploy_activation():
    activation = pytest.importorskip("lmdeploy.pytorch.kernels.cuda.activation")
    if "precise_mul" not in inspect.signature(activation.silu_and_mul).parameters:
        pytest.skip("LMDeploy with precise_mul support required")
    return activation.silu_and_mul


@requires_cuda
@pytest.mark.usefixtures("lmdeploy_activation")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape,split_dim", [((1, 4096), -1), ((31, 4096), -1), ((6, 32, 3), 1), ((0, 4096), -1)])
def test_forward_matches_inference_and_disabled_native(monkeypatch, dtype, shape, split_dim):
    from lmdeploy.pytorch.kernels.cuda.activation import silu_and_mul

    x = torch.randn(shape, device="cuda", dtype=dtype) * 3
    monkeypatch.setattr(act_fn, "_LMDEPLOY_SWIGLU", True)
    actual = act_fn.native_clamped_swiglu(x, split_dim=split_dim)
    moved = x.movedim(split_dim, -1)
    expected = (
        x.new_empty(actual.shape)
        if not x.numel()
        else silu_and_mul(moved.reshape(-1, moved.shape[-1]).contiguous(), swiglu_limit=10.0, precise_mul=True)
        .reshape(*moved.shape[:-1], moved.shape[-1] // 2)
        .movedim(-1, split_dim)
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    monkeypatch.setattr(act_fn, "_LMDEPLOY_SWIGLU", False)
    gate, up = x.chunk(2, dim=split_dim)
    native = F.silu(gate.clamp(max=10)) * up.clamp(-10, 10)
    torch.testing.assert_close(act_fn.native_clamped_swiglu(x, split_dim=split_dim), native, rtol=0, atol=0)


@requires_cuda
@pytest.mark.usefixtures("lmdeploy_activation")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_vjp_against_fp64_oracle_and_scaled_error(monkeypatch, dtype):
    monkeypatch.setattr(act_fn, "_LMDEPLOY_SWIGLU", True)
    x = (torch.randn((16, 4096), device="cuda") * 4).to(dtype)
    x[:, :8] = torch.tensor([-11.0, -10.0, -9.0, 0.0, 9.0, 10.0, 11.0, 12.0], device="cuda", dtype=dtype)
    x[:, 2048:2056] = x[:, :8]
    x.requires_grad_()
    grad_output = torch.randn((16, 2048), device="cuda", dtype=dtype)
    output = act_fn.native_clamped_swiglu(x)
    actual = torch.autograd.grad(output, x, grad_output)[0]
    reference_x = x.detach().double().requires_grad_()
    gate, up = reference_x.chunk(2, dim=-1)
    reference = F.silu(gate.clamp(max=10)) * up.clamp(-10, 10)
    expected = torch.autograd.grad(reference, reference_x, grad_output.double())[0].to(dtype)
    rtol, atol = (1e-2, 2e-2) if dtype == torch.bfloat16 else (1e-5, 1e-5)
    assert torch.isfinite(actual).all() and actual.abs().sum() > 0
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(actual * 1.1, expected, rtol=rtol, atol=atol)


@requires_cuda
@pytest.mark.usefixtures("lmdeploy_activation")
def test_separate_gate_up_and_fullgraph_backward(monkeypatch):
    monkeypatch.setattr(act_fn, "_LMDEPLOY_SWIGLU", True)
    gate = torch.randn((4, 128), device="cuda", requires_grad=True)
    up = torch.randn_like(gate, requires_grad=True)
    activation = act_fn.get_gated_act_fn("silu", swiglu_limit=10.0)
    combined = act_fn.native_clamped_swiglu(torch.cat((gate, up), dim=-1))
    torch.testing.assert_close(activation(gate, up), combined, rtol=0, atol=0)
    compiled = torch.compile(activation, backend="aot_eager", fullgraph=True)
    output = compiled(gate, up)
    grads = torch.autograd.grad(output.sum(), (gate, up))
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
    torch.testing.assert_close(output, combined, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_native_mathematical_backward_on_cpu(dtype):
    from types import SimpleNamespace

    from xtuner.v1.ops.lmdeploy_swiglu import _backward

    torch.manual_seed(621)
    x = (torch.randn(11, 128) * 4).to(dtype)
    x[:, :6] = torch.tensor([-11.0, -10.0, 0.0, 9.0, 10.0, 11.0], dtype=dtype)
    x[:, 64:70] = x[:, :6]
    dy = torch.randn(11, 64).to(dtype)
    actual, _ = _backward(SimpleNamespace(saved_tensors=(x,), limit=10.0), dy)
    ref = x.double().requires_grad_()
    gate, up = ref.chunk(2, -1)
    y = F.silu(gate.clamp(max=10.0)) * up.clamp(-10.0, 10.0)
    expected = torch.autograd.grad(y, ref, dy.double())[0].to(dtype)
    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-2 if dtype == torch.bfloat16 else 1e-5,
        atol=2e-2 if dtype == torch.bfloat16 else 1e-5,
    )
    assert actual.abs().sum() > 0


def test_enabled_flag_preserves_cpu_fallback(monkeypatch):
    monkeypatch.setattr(act_fn, "_LMDEPLOY_SWIGLU", True)
    x = torch.randn(3, 128, dtype=torch.bfloat16, requires_grad=True)
    gate, up = x.chunk(2, -1)
    expected = F.silu(gate.clamp(max=10.0)) * up.clamp(-10.0, 10.0)
    torch.testing.assert_close(act_fn.native_clamped_swiglu(x), expected, rtol=0, atol=0)


@requires_cuda
@pytest.mark.usefixtures("lmdeploy_activation")
def test_enabled_swiglu_updates_parameters(monkeypatch):
    monkeypatch.setattr(act_fn, "_LMDEPLOY_SWIGLU", True)
    torch.manual_seed(2300)
    parameter = torch.nn.Parameter(torch.randn(8, 128, device="cuda"))
    before = parameter.detach().clone()
    optimizer = torch.optim.SGD([parameter], lr=0.01)
    loss = act_fn.native_clamped_swiglu(parameter).square().mean()
    loss.backward()
    assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert parameter.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(parameter.detach(), before)
