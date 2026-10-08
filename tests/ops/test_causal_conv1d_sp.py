"""CUDA regression: packing documents must not change deterministic dweight."""

import pytest
import torch

from xtuner.v1.ops.gated_deltanet.causal_conv1d import causal_conv1d_fn


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("activation", [None, "silu"])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("channels", [64, 256])
@pytest.mark.parametrize("with_bias", [False, True])
def test_packed_document_gradients(activation, batch_size, channels, with_bias):
    old = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        torch.manual_seed(123)
        lengths = [64, 127, 129]
        x = torch.randn(batch_size, sum(lengths), channels, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(channels, 4, device="cuda", dtype=torch.float32, requires_grad=True)
        bias = torch.randn(channels, device="cuda", dtype=torch.float32, requires_grad=True) if with_bias else None
        params = (x, weight, bias) if with_bias else (x, weight)
        x.requires_grad_()
        dout = torch.randn_like(x)
        ids = (
            torch.repeat_interleave(
                torch.arange(3, device="cuda", dtype=torch.int32), torch.tensor(lengths, device="cuda")
            )
            .expand(batch_size, -1)
            .contiguous()
        )
        out = causal_conv1d_fn(x, weight, bias=bias, seq_idx=ids, activation=activation)
        grads = torch.autograd.grad(out, params, dout)
        dx, dw = grads[:2]

        expected_dw = torch.zeros_like(weight)
        expected_db = torch.zeros_like(bias) if with_bias else None
        expected_out, expected_dx = torch.empty_like(out), torch.empty_like(x)
        for b in range(batch_size):
            start = 0
            for length in lengths:
                end = start + length
                doc = x[b : b + 1, start:end].detach().contiguous().requires_grad_()
                doc_out = causal_conv1d_fn(
                    doc,
                    weight,
                    bias=bias,
                    seq_idx=torch.zeros((1, length), device="cuda", dtype=torch.int32),
                    activation=activation,
                )
                doc_params = (doc, weight, bias) if with_bias else (doc, weight)
                doc_grads = torch.autograd.grad(doc_out, doc_params, dout[b : b + 1, start:end].contiguous())
                doc_dx, doc_dw = doc_grads[:2]
                expected_dw.add_(doc_dw)
                if with_bias:
                    expected_db.add_(doc_grads[2])
                expected_out[b : b + 1, start:end] = doc_out.detach()
                expected_dx[b : b + 1, start:end] = doc_dx
                start = end
        assert dw.dtype == torch.float32
        torch.testing.assert_close(out, expected_out, rtol=0, atol=0)
        torch.testing.assert_close(dx, expected_dx, rtol=0, atol=0)
        torch.testing.assert_close(dw, expected_dw, rtol=0, atol=0)
        if with_bias:
            assert grads[2].dtype == torch.float32
            torch.testing.assert_close(grads[2], expected_db, rtol=0, atol=0)
    finally:
        torch.use_deterministic_algorithms(old, warn_only=warn_only)
