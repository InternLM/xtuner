"""Opt-in KDA QKV precision, default parity, and independent recurrent VJPs."""

import importlib

import pytest
import torch


op = importlib.import_module("xtuner.v1.ops.kda.chunk_kda")
FLAG = "XTUNER_RL_KDA_INTERNAL_FP32"


def _reference(q, k, v, g, beta, lengths):
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    output = []
    start = 0
    for length in lengths:
        state = q.new_zeros(q.shape[0], q.shape[2], v.shape[-1], k.shape[-1])
        for index in range(start, start + length):
            state = state * g[:, index].exp().unsqueeze(-2)
            delta = (v[:, index] - torch.einsum("bhvk,bhk->bhv", state, k[:, index])) * beta[:, index, :, None]
            state = state + delta[..., None] * k[:, index, :, None, :]
            output.append(torch.einsum("bhvk,bhk->bhv", state, q[:, index]) * q.shape[-1] ** -0.5)
        start += length
    return torch.stack(output, dim=1)


def _relative_l2(actual, expected):
    actual, expected = actual.double().cpu(), expected.double().cpu()
    return float((actual - expected).norm() / expected.norm().clamp_min(1e-30))


class TestKDAInternalPrecision:
    @pytest.mark.parametrize("flag", [None, "0", "1"])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_wrapper_dispatch_and_output_dtype(self, monkeypatch, flag, dtype):
        if flag is None:
            monkeypatch.delenv(FLAG, raising=False)
        else:
            monkeypatch.setenv(FLAG, flag)
        q = torch.randn(1, 4, 1, 128, dtype=dtype)
        g = torch.randn_like(q, dtype=torch.float32)
        beta = torch.rand(1, 4, 1)
        captured = []

        def apply(*args):
            captured.append(args)
            return args[0] * 0.5, None

        monkeypatch.setattr(op.ChunkKDAFunction, "apply", apply)
        result, final_state = op.chunk_kda(q, q, q, g, beta)
        expected_dtype = torch.float32 if flag == "1" else dtype
        assert all(x.dtype == expected_dtype for x in captured[0][:3])
        assert captured[0][3] is g and captured[0][4] is beta
        if flag != "1":
            assert all(x is q for x in captured[0][:3])
        assert result.dtype == dtype and result.shape == q.shape and final_state is None

    @pytest.mark.gpu
    @pytest.mark.parametrize("lengths", [(16,), (7, 9), (65,), (129,)])
    def test_native_shape_forward_backward(self, monkeypatch, lengths):
        if not torch.cuda.is_available():
            pytest.skip("CUDA is required")
        torch.manual_seed(20261104)
        t = sum(lengths)
        q = torch.randn(1, t, 2, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)
        g = (-(torch.rand_like(q, dtype=torch.float32) * 2 + 0.1)).requires_grad_()
        beta = torch.rand(1, t, 2, device="cuda", requires_grad=True)
        inputs = (q, k, v, g, beta)
        reference_inputs = tuple(x.detach().cpu().double().requires_grad_() for x in inputs)
        cotangent = torch.randn_like(q)
        reference = _reference(*reference_inputs, lengths)
        expected_grads = torch.autograd.grad(reference, reference_inputs, cotangent.cpu().double())
        cu_cpu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
        cu = cu_cpu.cuda()
        kwargs = dict(safe_gate=True, transpose_state_layout=True, cu_seqlens=cu, cu_seqlens_cpu=cu_cpu)

        monkeypatch.delenv(FLAG, raising=False)
        native = op.chunk_kda(*inputs, **kwargs)[0]
        native_grads = torch.autograd.grad(native, inputs, cotangent)
        direct = op.ChunkKDAFunction.apply(*inputs, 128**-0.5, cu, cu_cpu, True, None, True)[0]
        direct_grads = torch.autograd.grad(direct, inputs, cotangent)
        assert torch.equal(native, direct)
        assert all(torch.equal(x, y) for x, y in zip(native_grads, direct_grads))

        monkeypatch.setenv(FLAG, "1")
        actual = op.chunk_kda(*inputs, **kwargs)[0]
        actual_grads = torch.autograd.grad(actual, inputs, cotangent)
        assert actual.dtype == q.dtype
        assert _relative_l2(actual, reference) < 0.01
        for got, target, original in zip(actual_grads, expected_grads, inputs):
            assert got.dtype == original.dtype and torch.isfinite(got).all()
            assert _relative_l2(got, target) < 0.01
