from collections.abc import Callable

import pytest
import torch


pytest.importorskip("deep_ep")

from xtuner.v1.module.dispatcher import deepep as implementation  # noqa: E402
from xtuner.v1.ops import permute, unpermute  # noqa: E402


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.fixture
def combine() -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    # This local stage needs no process group or communication buffer.
    dispatcher = implementation.DeepEPDispatcher.__new__(implementation.DeepEPDispatcher)

    def operation(hidden: torch.Tensor, weights: torch.Tensor, row_map: torch.Tensor) -> torch.Tensor:
        return dispatcher.combine_preprocess(
            hidden_states=hidden,
            pre_dispatched={},
            dispatched={"topk_weights": weights},
            post_dispatched={"row_ids_map": row_map},
        )["hidden_states"]

    return operation


def _payload(rows: int, dtype: torch.dtype = torch.bfloat16):
    torch.manual_seed(146 + rows)
    width, topk = 4096, 8
    ids = torch.randperm(rows * topk, device="cuda").reshape(rows, topk).int()
    dummy = torch.arange(rows, device="cuda").to(dtype)[:, None].expand(rows, width).contiguous()
    permuted, row_map = permute(dummy, ids)
    torch.testing.assert_close(permuted, dummy[ids.flatten().argsort() // topk], rtol=0, atol=0)
    hidden = torch.randn(rows * topk, width, device="cuda", dtype=dtype, requires_grad=True)
    weights = torch.randn(rows, topk, device="cuda").softmax(-1).detach().requires_grad_()
    return hidden, weights, ids.long(), row_map


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_disabled_preserves_native_output_and_gradients(combine, monkeypatch, dtype):
    monkeypatch.setattr(implementation, "_COMBINE_FP32", False)
    hidden, weights, _, row_map = _payload(9, dtype)
    actual = combine(hidden, weights, row_map)
    legacy = unpermute(hidden, row_map, probs=weights)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, legacy, rtol=0, atol=0)
    upstream = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (hidden, weights), upstream)
    legacy_grads = torch.autograd.grad(legacy, (hidden, weights), upstream)
    for actual_grad, legacy_grad in zip(actual_grads, legacy_grads):
        torch.testing.assert_close(actual_grad, legacy_grad, rtol=0, atol=0)


@pytest.mark.parametrize("rows", [9, 128])
def test_fp32_weighted_sum_vjp_and_update(combine, monkeypatch, rows):
    monkeypatch.setattr(implementation, "_COMBINE_FP32", True)
    hidden, weights, ids, row_map = _payload(rows)
    actual = combine(hidden, weights, row_map)
    ref_hidden = hidden.detach().clone().requires_grad_()
    ref_weights = weights.detach().clone().requires_grad_()
    # Independent token/expert indexing; do not reuse the kernel's row map.
    expected = (ref_hidden[ids].float() * ref_weights[..., None]).sum(1).bfloat16()
    assert actual.dtype == hidden.dtype
    assert actual.numel() > 0 and torch.isfinite(actual).all()
    error = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert error < 1e-4
    upstream = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (hidden, weights), upstream)
    expected_grads = torch.autograd.grad(expected, (ref_hidden, ref_weights), upstream)
    for grad, reference, tolerance in zip(actual_grads, expected_grads, [0.005, 2e-5]):
        assert torch.isfinite(grad).all() and grad.float().norm() > 0
        relative_error = (grad.float() - reference.float()).norm() / reference.float().norm()
        assert relative_error < tolerance
        # The oracle must reject a plausible, but wrong, backward scaling.
        assert (1.25 * grad.float() - reference.float()).norm() / reference.float().norm() > 0.1
    before = weights.detach().clone()
    weights.grad = actual_grads[1]
    torch.optim.SGD([weights], lr=0.01).step()
    assert not torch.equal(weights, before)
    torch.testing.assert_close(weights, before - 0.01 * expected_grads[1], rtol=0, atol=1e-6)


def test_fp32_fullgraph_forward_and_backward(combine, monkeypatch):
    monkeypatch.setattr(implementation, "_COMBINE_FP32", True)
    hidden, weights, _, row_map = _payload(9)
    eager = combine(hidden, weights, row_map)
    compiled = torch.compile(combine, fullgraph=True, backend="aot_eager")(hidden, weights, row_map)
    torch.testing.assert_close(compiled, eager, rtol=0, atol=0)
    upstream = torch.randn_like(eager)
    eager_grads = torch.autograd.grad(eager, (hidden, weights), upstream)
    compiled_grads = torch.autograd.grad(compiled, (hidden, weights), upstream)
    for compiled_grad, eager_grad in zip(compiled_grads, eager_grads):
        torch.testing.assert_close(compiled_grad, eager_grad, rtol=0, atol=0)
