"""Regression tests for the fused Triton ``hc_post`` (xtuner.v1.ops.hc_post).

The fused kernel must (1) match the eager reference within bf16 rounding, (2)
produce correct gradients for all four inputs, and (3) trace cleanly under
``torch.compile`` (it is registered as a ``torch.library.custom_op`` so the V4
decoder layer can keep compiling around it). The eager reference is the body of
:func:`xtuner.v1.module.decoder_layer.hc_block._hc_post_eager`.
"""

import pytest
import torch


def _hc_post_ref(x, residual, post, comb):
    # Mirror of hc_block._hc_post_eager: fp32-accumulate broadcast-multiply + sum.
    post_dt = post.to(residual.dtype)
    comb_b = comb.to(residual.dtype)
    mixed = (
        comb_b.float().transpose(-1, -2).unsqueeze(-1) * residual.float().unsqueeze(-3)
    ).sum(-2).to(residual.dtype)
    return post_dt.unsqueeze(-1) * x.unsqueeze(-2) + mixed


def _make_inputs(B, S, H, D, device, requires_grad):
    x = torch.randn(B, S, D, device=device, dtype=torch.bfloat16, requires_grad=requires_grad)
    residual = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16, requires_grad=requires_grad)
    # post / comb arrive from the fp32 Sinkhorn output in the real model.
    post = torch.rand(B, S, H, device=device, dtype=torch.float32, requires_grad=requires_grad)
    comb = torch.softmax(
        torch.randn(B, S, H, H, device=device, dtype=torch.float32), dim=-1
    ).requires_grad_(requires_grad)
    return x, residual, post, comb


class TestHCPostFused:
    @pytest.mark.gpu
    @pytest.mark.parametrize("S", [1000, 2048, 4096])
    def test_forward_matches_reference(self, S):
        from xtuner.v1.ops.hc_post import hc_post_fused

        torch.manual_seed(0)
        x, residual, post, comb = _make_inputs(1, S, 4, 4096, "cuda", requires_grad=False)
        ref = _hc_post_ref(x, residual, post, comb)
        out = hc_post_fused(x, residual, post, comb)

        # The fused kernel uses a different (equally valid) bf16 reduction order,
        # so it is not bit-identical. The max abs diff is ~1 bf16 ULP at the
        # output's largest magnitude (~2**-5 at |out|~4); the mean stays tiny,
        # which is what rules out a systematic error.
        diff = (ref.float() - out.float()).abs()
        assert diff.max().item() < 6e-2, f"max abs diff {diff.max().item():.3e}"
        assert diff.mean().item() < 3e-3, f"mean abs diff {diff.mean().item():.3e}"

    @pytest.mark.gpu
    def test_forward_no_worse_than_reference_vs_fp32(self):
        # The fused path keeps more of the accumulation in fp32, so it must be at
        # least as close to the fp32 ground truth as the eager reference.
        from xtuner.v1.ops.hc_post import hc_post_fused

        torch.manual_seed(1)
        x, residual, post, comb = _make_inputs(1, 4096, 4, 4096, "cuda", requires_grad=False)
        gt = (
            comb.float().transpose(-1, -2).unsqueeze(-1) * residual.float().unsqueeze(-3)
        ).sum(-2)
        gt = post.float().unsqueeze(-1) * x.float().unsqueeze(-2) + gt

        err_ref = (gt - _hc_post_ref(x, residual, post, comb).float()).abs().mean()
        err_fused = (gt - hc_post_fused(x, residual, post, comb).float()).abs().mean()
        assert err_fused <= err_ref * 1.05

    @pytest.mark.gpu
    def test_backward_matches_reference(self):
        from xtuner.v1.ops.hc_post import hc_post_fused

        torch.manual_seed(2)
        ref_inputs = _make_inputs(1, 1024, 4, 4096, "cuda", requires_grad=True)
        fused_inputs = [t.detach().clone().requires_grad_(True) for t in ref_inputs]

        out_ref = _hc_post_ref(*ref_inputs)
        out_fused = hc_post_fused(*fused_inputs)
        grad_out = torch.randn_like(out_ref)
        out_ref.backward(grad_out)
        out_fused.backward(grad_out)

        names = ["grad_x", "grad_residual", "grad_post", "grad_comb"]
        for name, ref_t, fused_t in zip(names, ref_inputs, fused_inputs):
            assert ref_t.grad.dtype == fused_t.grad.dtype, f"{name} dtype mismatch"
            denom = ref_t.grad.float().abs().max().item() + 1e-9
            rel = (ref_t.grad.float() - fused_t.grad.float()).abs().max().item() / denom
            assert rel < 1e-2, f"{name} rel diff {rel:.3e} too large"

    @pytest.mark.gpu
    def test_compile_fullgraph(self):
        from xtuner.v1.ops.hc_post import hc_post_fused

        torch.manual_seed(3)
        x, residual, post, comb = _make_inputs(1, 1024, 4, 4096, "cuda", requires_grad=True)
        grad_out = torch.randn(1, 1024, 4, 4096, device="cuda", dtype=torch.bfloat16)

        eager = hc_post_fused(x, residual, post, comb)

        compiled_fn = torch.compile(hc_post_fused, fullgraph=True)
        x2, residual2, post2, comb2 = [
            t.detach().clone().requires_grad_(True) for t in (x, residual, post, comb)
        ]
        compiled = compiled_fn(x2, residual2, post2, comb2)
        # custom_op is opaque to dynamo → forward must be bit-identical to eager.
        assert torch.equal(eager, compiled)
        compiled.backward(grad_out)
        assert x2.grad is not None and comb2.grad is not None
