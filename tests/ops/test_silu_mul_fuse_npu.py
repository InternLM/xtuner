"""Zero-centered gated-norm silu-gate tail (``y * silu(g)``) regression tests.

:func:`xtuner.v1.ops.act_fn.fused_silu_mul` serves the gated-RMSNorm silu
tail with shape dispatch. At or above ``_SILU_MUL_TRITON_MIN_NUMEL`` (2**25)
a triton kernel pair runs: fp32 compute with a single bf16 round per output,
so forward and ``dy`` are bitwise equal to the eager fp32 chain
``(y.float() * F.silu(g.float())).to(bf16)`` and ``dg`` differs only within
aclnnSiluBackward's internal fp32 op order (~6e-6 of elements, <=1 bf16 ulp).
Below the threshold the plain bf16 tail ``y * F.silu(g.to(y.dtype))`` runs
unchanged -- measured fastest at small shapes (2 native aclnn enqueues vs the
~30-60us autograd-Function + launch overhead of any custom kernel) and
bitwise identical to the expression it replaces. Relative to that bf16 tail,
the triton route's outputs differ on ~27.5% of elements by exactly 1 bf16
ulp (~25.7% for ``dg``): the tail rounds silu's bf16 output a second time in
the mul, while the kernel keeps fp32 until the store, landing on the
correctly rounded value. Everything failing the dispatch -- shapes below the
threshold (the element-count check short-circuits before the operand gate, so
small-shape calls pay only a ``numel`` comparison) or operands failing the
gate -- runs the plain bf16 tail unchanged, which is exactly what the tail
expression did before the dispatch existed. Double backward
(``create_graph=True`` twice) is
supported on the triton route: its backward switches to an autograd-trackable
eager formulation (arbitrary order) whose first- and second-order grads match
the eager fp32 chain bit for bit except ``ddg`` (aclnnSiluBackward
differentiated twice vs the explicit composition, ~4e-5 of elements, <=0.05
max rel).

Run with ``XTUNER_DETERMINISTIC=false`` if the project's deterministic patch is
active: it can interfere with triton kernel launches under the default setting.

TestSiluMulFuseAgainstEagerFp32Chain(@requires_npu)
    test_production_shape_triton_route: production gated-norm shape
        [1, 8192, 32, 128] (>= 2**25, triton route) -- forward and dy bitwise
        equal to the eager fp32 chain, dg bounded vs it, and the diff vs the
        plain bf16 tail bounded at 1 bf16 ulp on a minority of elements.
    test_small_shapes_bitwise_match_bf16_tail: shapes below the threshold --
        forward, dy and dg bitwise equal to the plain bf16 tail.
    test_various_pow2_widths_small_shapes: last-dim in {64, 128, 256, 512}
        (below threshold) -- bitwise equal to the bf16 tail.
    test_non_contiguous_falls_back_to_bf16_tail: non-contiguous bf16 g at a
        large shape -- gate rejects, the fallback matches the plain bf16
        tail expression exactly.
    test_fp32_inputs_fall_back_to_bf16_tail: fp32 inputs -- gate rejects,
        the fallback matches the plain bf16 tail expression exactly.
    test_threshold_dispatch_forced_routes: threshold monkeypatched to 0 (tiny
        shape takes the triton route, bitwise vs the fp32 chain) and to a
        huge value (mid shape takes the bf16 tail, bitwise vs it).
    test_triton_forced_allocator_poisoning: threshold 0, NaN-fill then del
        proves no unwritten dst rows on the triton route.
    test_double_backward_matches_eager: create_graph=True twice (triton
        forced) -- forward, dy, dg, ddy bitwise; ddg bounded in fraction and
        magnitude.
    test_double_backward_leaf_grad_outputs: leaf ``torch.ones`` grad outputs
        still take the differentiable path and match eager (triton forced).
    test_third_order_runs_finite: third-order grads via the eager branch
        (triton forced).
"""

import os

import pytest
import torch
import torch.nn.functional as F


os.environ.setdefault("XTUNER_DETERMINISTIC", "false")

import xtuner.v1.ops.act_fn as _act_fn_mod  # noqa: E402
from xtuner.v1.ops.act_fn import _silu_mul_ok, fused_silu_mul  # noqa: E402
from xtuner.v1.utils.device import get_device  # noqa: E402


DEVICE = get_device()
requires_npu = pytest.mark.skipif(DEVICE != "npu", reason="NPU-only fused kernel")


def _fp32_chain(y: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    return (y.float() * F.silu(g.float())).to(y.dtype)


def _bf16_tail(y: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    return y * F.silu(g.to(y.dtype))


def _run_pair(
    y: torch.Tensor,
    g: torch.Tensor,
    ref_fn,
    seed_grad: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run reference and fused forward+backward, return outputs and grads.

    Args:
        y (torch.Tensor): Normed input (bf16, contiguous).
        g (torch.Tensor): Gate, same shape as ``y`` (bf16, contiguous).
        ref_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
            Reference tail to compare against.
        seed_grad (torch.Tensor | None): Backward seed gradient; defaults to
            ``torch.ones_like`` of the output.

    Returns:
        tuple: ``(out_ref, out_fused, dy_ref, dy_fused, dg_ref, dg_fused)``.
    """
    y_r = y.detach().clone().requires_grad_(True)
    g_r = g.detach().clone().requires_grad_(True)
    out_r = ref_fn(y_r, g_r)
    grad = torch.ones_like(out_r) if seed_grad is None else seed_grad
    out_r.backward(grad)
    y_f = y.detach().clone().requires_grad_(True)
    g_f = g.detach().clone().requires_grad_(True)
    out_f = fused_silu_mul(y_f, g_f)
    out_f.backward(grad)
    return out_r, out_f, y_r.grad, y_f.grad, g_r.grad, g_f.grad


def _assert_fwd_dy_bitwise(out_r, out_f, dy_r, dy_f) -> None:
    assert torch.equal(out_r, out_f), "forward output must be bitwise equal"
    assert torch.equal(dy_r, dy_f), "dy must be bitwise equal"


def _assert_dg_within_ulp(dg_r: torch.Tensor, dg_f: torch.Tensor) -> None:
    mism = (dg_r != dg_f).sum().item()
    assert mism / dg_r.numel() <= 1e-4, f"dg mismatch fraction too high: {mism}/{dg_r.numel()}"
    rel = ((dg_r.float() - dg_f.float()).abs() / dg_f.float().abs().clamp_min(1e-6)).max().item()
    assert rel <= 0.05, f"dg max relative diff too high: {rel}"


def _assert_within_1ulp_of_bf16_tail(ref: torch.Tensor, fus: torch.Tensor) -> None:
    """Triton route vs the plain bf16 tail: minority of elements, <=1 ulp."""
    mism = (ref != fus).sum().item()
    assert mism / ref.numel() <= 0.5, f"bf16-tail mismatch fraction too high: {mism}/{ref.numel()}"
    rel = ((ref.float() - fus.float()).abs() / fus.float().abs().clamp_min(1e-6)).max().item()
    assert rel <= 0.0079, f"bf16-tail max relative diff too high (must stay <=1 bf16 ulp): {rel}"


def _dbl_backward(y: torch.Tensor, g: torch.Tensor, seed_fn) -> tuple[torch.Tensor, ...]:
    """Double backward through ``seed_fn``; returns (out, dy, dg, ddy, ddg).

    Args:
        y (torch.Tensor): Normed input (bf16).
        g (torch.Tensor): Gate, same shape as ``y``.
        seed_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): The
            forward tail whose double backward is taken.

    Returns:
        tuple: ``(out, dy, dg, ddy, ddg)`` detached except ``ddy``/``ddg``.
    """
    y1 = y.detach().clone().requires_grad_(True)
    g1 = g.detach().clone().requires_grad_(True)
    out = seed_fn(y1, g1)
    dy, dg = torch.autograd.grad(out, (y1, g1), grad_outputs=torch.ones_like(out), create_graph=True)
    loss2 = (dy * 2).sum() + (dg * 3).sum()
    ddy, ddg = torch.autograd.grad(loss2, (y1, g1))
    return out.detach(), dy.detach(), dg.detach(), ddy, ddg


@requires_npu
class TestSiluMulFuseAgainstEagerFp32Chain:
    def test_production_shape_triton_route(self):
        torch.manual_seed(0)
        y = torch.randn(1, 8192, 32, 128, device=DEVICE, dtype=torch.bfloat16)
        g = torch.randn(1, 8192, 32, 128, device=DEVICE, dtype=torch.bfloat16)
        grad = torch.randn(1, 8192, 32, 128, device=DEVICE, dtype=torch.bfloat16)
        assert _silu_mul_ok(y, g)
        out_e, out_f, dy_e, dy_f, dg_e, dg_f = _run_pair(y, g, _fp32_chain, grad)
        _assert_fwd_dy_bitwise(out_e, out_f, dy_e, dy_f)
        _assert_dg_within_ulp(dg_e, dg_f)
        # vs the plain bf16 tail (the small-shape route): a minority of
        # elements differ, each by at most 1 bf16 ulp -- the kernel's fp32
        # recipe vs the tail's extra intermediate rounding.
        out_t, _, _, _, dg_t, _ = _run_pair(y, g, _bf16_tail, grad)
        _assert_within_1ulp_of_bf16_tail(out_t, out_f)
        _assert_within_1ulp_of_bf16_tail(dg_t, dg_f)

    @pytest.mark.parametrize("shape", [(1, 512, 4, 128), (1, 1024, 32, 128), (4, 1024, 32, 128)])
    def test_small_shapes_bitwise_match_bf16_tail(self, shape: tuple[int, ...]):
        torch.manual_seed(1)
        y = torch.randn(*shape, device=DEVICE, dtype=torch.bfloat16)
        g = torch.randn(*shape, device=DEVICE, dtype=torch.bfloat16)
        grad = torch.randn(*shape, device=DEVICE, dtype=torch.bfloat16)
        out_t, out_f, dy_t, dy_f, dg_t, dg_f = _run_pair(y, g, _bf16_tail, grad)
        _assert_fwd_dy_bitwise(out_t, out_f, dy_t, dy_f)
        assert torch.equal(dg_t, dg_f), "small-shape dg must be bitwise equal to the bf16 tail"

    @pytest.mark.parametrize("width", [64, 128, 256, 512])
    def test_various_pow2_widths_small_shapes(self, width: int):
        torch.manual_seed(2)
        y = torch.randn(1, 2048, width, device=DEVICE, dtype=torch.bfloat16)
        g = torch.randn(1, 2048, width, device=DEVICE, dtype=torch.bfloat16)
        out_t, out_f, dy_t, dy_f, _, _ = _run_pair(y, g, _bf16_tail)
        _assert_fwd_dy_bitwise(out_t, out_f, dy_t, dy_f)

    def test_non_contiguous_falls_back_to_bf16_tail(self):
        torch.manual_seed(3)
        y = torch.randn(1, 2048, 16384, device=DEVICE, dtype=torch.bfloat16)
        g = torch.randn(1, 16384, 2048, device=DEVICE, dtype=torch.bfloat16).transpose(1, 2)
        assert y.numel() >= _act_fn_mod._SILU_MUL_TRITON_MIN_NUMEL
        assert not _silu_mul_ok(y, g)
        out_f = fused_silu_mul(y, g)
        out_e = _bf16_tail(y, g)
        assert torch.equal(out_e, out_f), "gate-fail fallback must match the plain bf16 tail"

    def test_fp32_inputs_fall_back_to_bf16_tail(self):
        torch.manual_seed(4)
        y = torch.randn(1, 2048, 128, device=DEVICE, dtype=torch.float32)
        g = torch.randn(1, 2048, 128, device=DEVICE, dtype=torch.float32)
        assert not _silu_mul_ok(y, g)
        out_f = fused_silu_mul(y, g)
        out_e = _bf16_tail(y, g)
        assert torch.equal(out_e, out_f), "gate-fail fallback must match the plain bf16 tail"

    def test_threshold_dispatch_forced_routes(self, monkeypatch: pytest.MonkeyPatch):
        torch.manual_seed(5)
        y = torch.randn(1, 512, 4, 128, device=DEVICE, dtype=torch.bfloat16)
        g = torch.randn(1, 512, 4, 128, device=DEVICE, dtype=torch.bfloat16)
        grad = torch.randn(1, 512, 4, 128, device=DEVICE, dtype=torch.bfloat16)
        # Threshold 0: the tiny shape takes the triton route -- bitwise vs the
        # fp32 chain, dg within the fused-backward residual.
        monkeypatch.setattr(_act_fn_mod, "_SILU_MUL_TRITON_MIN_NUMEL", 0)
        out_e, out_f, dy_e, dy_f, dg_e, dg_f = _run_pair(y, g, _fp32_chain, grad)
        _assert_fwd_dy_bitwise(out_e, out_f, dy_e, dy_f)
        _assert_dg_within_ulp(dg_e, dg_f)
        # Threshold huge: the mid shape takes the plain bf16 tail -- bitwise
        # vs the tail expression. A dispatch bug routing to triton would break
        # this, since the triton route differs on ~27.5% of elements.
        monkeypatch.setattr(_act_fn_mod, "_SILU_MUL_TRITON_MIN_NUMEL", 1 << 40)
        y2 = torch.randn(4, 1024, 32, 128, device=DEVICE, dtype=torch.bfloat16)
        g2 = torch.randn(4, 1024, 32, 128, device=DEVICE, dtype=torch.bfloat16)
        out_t, out_f, dy_t, dy_f, _, _ = _run_pair(y2, g2, _bf16_tail)
        _assert_fwd_dy_bitwise(out_t, out_f, dy_t, dy_f)

    def test_triton_forced_allocator_poisoning(self, monkeypatch: pytest.MonkeyPatch):
        torch.manual_seed(6)
        monkeypatch.setattr(_act_fn_mod, "_SILU_MUL_TRITON_MIN_NUMEL", 0)
        y = torch.randn(1, 512, 4, 128, device=DEVICE, dtype=torch.bfloat16)
        g = torch.randn(1, 512, 4, 128, device=DEVICE, dtype=torch.bfloat16)
        out_e, out_f, dy_e, dy_f, _, _ = _run_pair(y, g, _fp32_chain)
        _assert_fwd_dy_bitwise(out_e, out_f, dy_e, dy_f)
        # Allocator poisoning: fill fresh allocations with NaN, free them, then
        # run the kernel -- any unwritten dst row would surface as NaN.
        poison = torch.empty_like(out_f)
        poison.fill_(float("nan"))
        del poison
        y2 = y.detach().clone().requires_grad_(True)
        g2 = g.detach().clone().requires_grad_(True)
        out2 = fused_silu_mul(y2, g2)
        assert torch.equal(out2, out_e), "poisoned-allocation run must still be bitwise equal"

    def test_double_backward_matches_eager(self, monkeypatch: pytest.MonkeyPatch):
        torch.manual_seed(7)
        monkeypatch.setattr(_act_fn_mod, "_SILU_MUL_TRITON_MIN_NUMEL", 0)
        y = torch.randn(4, 1024, 32, 128, device=DEVICE, dtype=torch.bfloat16)
        g = torch.randn(4, 1024, 32, 128, device=DEVICE, dtype=torch.bfloat16)
        out_e, dy_e, dg_e, ddy_e, ddg_e = _dbl_backward(y, g, _fp32_chain)
        out_f, dy_f, dg_f, ddy_f, ddg_f = _dbl_backward(y, g, fused_silu_mul)
        assert torch.equal(out_e, out_f), "forward must be bitwise equal"
        # Under create_graph=True the fused backward takes an eager formulation
        # mirroring the kernel's op order, which reproduces the fp32 chain
        # exactly at first order.
        assert torch.equal(dy_e, dy_f), "first-order dy must be bitwise equal"
        assert torch.equal(dg_e, dg_f), "first-order dg must be bitwise equal"
        assert torch.equal(ddy_e, ddy_f), "second-order ddy must be bitwise equal"
        # ddg: aclnnSiluBackward differentiated twice vs the explicit
        # composition -- measured ~4.0e-5 of elements, max rel 1.5e-2 at 16.8M.
        mism = (ddg_e != ddg_f).sum().item()
        assert mism / ddg_e.numel() <= 1e-4, f"ddg mismatch fraction too high: {mism}/{ddg_e.numel()}"
        rel = ((ddg_e.float() - ddg_f.float()).abs() / ddg_f.float().abs().clamp_min(1e-6)).max().item()
        assert rel <= 0.05, f"ddg max relative diff too high: {rel}"

    def test_double_backward_leaf_grad_outputs(self, monkeypatch: pytest.MonkeyPatch):
        torch.manual_seed(8)
        monkeypatch.setattr(_act_fn_mod, "_SILU_MUL_TRITON_MIN_NUMEL", 0)
        y = torch.randn(1, 512, 4, 128, device=DEVICE, dtype=torch.bfloat16)
        g = torch.randn(1, 512, 4, 128, device=DEVICE, dtype=torch.bfloat16)
        results = []
        for fn in (_fp32_chain, fused_silu_mul):
            y1 = y.detach().clone().requires_grad_(True)
            g1 = g.detach().clone().requires_grad_(True)
            out = fn(y1, g1)
            # Leaf torch.ones seeds (requires_grad=False) with create_graph:
            # the differentiable backward path must still engage.
            dy, dg = torch.autograd.grad(out, (y1, g1), grad_outputs=torch.ones_like(out), create_graph=True)
            loss2 = dy.square().mean() + dg.square().mean()
            ddy, ddg = torch.autograd.grad(loss2, (y1, g1))
            results.append((out, dy, dg, ddy, ddg))
        (out_e, dy_e, dg_e, ddy_e, _), (out_f, dy_f, dg_f, ddy_f, _) = results
        assert torch.equal(out_e, out_f)
        assert torch.equal(dy_e, dy_f)
        assert torch.equal(dg_e, dg_f)
        assert torch.equal(ddy_e, ddy_f)
        assert bool(torch.isfinite(results[1][3]).all() and torch.isfinite(results[1][4]).all())

    def test_third_order_runs_finite(self, monkeypatch: pytest.MonkeyPatch):
        torch.manual_seed(9)
        monkeypatch.setattr(_act_fn_mod, "_SILU_MUL_TRITON_MIN_NUMEL", 0)
        y = torch.randn(1, 64, 128, device=DEVICE, dtype=torch.bfloat16)
        g = torch.randn(1, 64, 128, device=DEVICE, dtype=torch.bfloat16)
        y1 = y.detach().clone().requires_grad_(True)
        g1 = g.detach().clone().requires_grad_(True)
        out = fused_silu_mul(y1, g1)
        dy, dg = torch.autograd.grad(out, (y1, g1), grad_outputs=torch.ones_like(out), create_graph=True)
        loss2 = (dy * 2).sum() + (dg * 3).sum()
        ddy, ddg = torch.autograd.grad(loss2, (y1, g1), create_graph=True)
        d3y, d3g = torch.autograd.grad(ddy.sum() + ddg.sum(), (y1, g1))
        assert bool(torch.isfinite(d3y).all() and torch.isfinite(d3g).all())
