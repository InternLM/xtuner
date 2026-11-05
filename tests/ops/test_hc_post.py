"""mHC hc_post 的 Triton 融合实现，见 doc/xtuner_glm5p3flash_design.md F4。

TestHCPostFused
    test_forward_matches_reference                前向与 eager 参考一致
    test_forward_no_worse_than_reference_vs_fp32  相对 fp32 真值不劣于 eager
    test_backward_matches_reference               反向四个梯度与 eager 参考一致
    test_compile_fullgraph                        可在 fullgraph 编译区内使用
"""

import pytest
import torch


def _hc_post_ref(x, residual, post, comb):
    # Keep coefficients and the contraction in FP32, then cast only the output.
    out = post.float().unsqueeze(-1) * x.float().unsqueeze(-2)
    for hi in range(residual.size(-2)):
        out = out + comb[..., hi, :].float().unsqueeze(-1) * residual[..., hi, :].float().unsqueeze(-2)
    return out.to(residual.dtype)


def _make_inputs(B, S, H, D, device, requires_grad):
    x = torch.randn(B, S, D, device=device, dtype=torch.bfloat16, requires_grad=requires_grad)
    residual = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16, requires_grad=requires_grad)
    # post / comb arrive from the fp32 Sinkhorn output in the real model.
    post = torch.rand(B, S, H, device=device, dtype=torch.float32, requires_grad=requires_grad)
    comb = torch.softmax(torch.randn(B, S, H, H, device=device, dtype=torch.float32), dim=-1).requires_grad_(
        requires_grad
    )
    return x, residual, post, comb


class TestHCPostFused:
    @pytest.mark.gpu
    @pytest.mark.parametrize("S", [1000, 2048, 4096])
    def test_forward_matches_reference(self, S):
        # 融合 kernel 的前向与 eager 参考在 bf16 容差内一致。
        from xtuner.v1.ops.hc_post import hc_post_fused

        torch.manual_seed(0)
        x, residual, post, comb = _make_inputs(1, S, 4, 4096, "cuda", requires_grad=False)
        ref = _hc_post_ref(x, residual, post, comb)
        out = hc_post_fused(x, residual, post, comb)

        # FMA contraction may differ from the separate eager multiply/add operations.
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
        gt = (comb.float().transpose(-1, -2).unsqueeze(-1) * residual.float().unsqueeze(-3)).sum(-2)
        gt = post.float().unsqueeze(-1) * x.float().unsqueeze(-2) + gt

        mixed_bf16 = (
            (comb.to(residual.dtype).float().transpose(-1, -2).unsqueeze(-1) * residual.float().unsqueeze(-3))
            .sum(-2)
            .to(residual.dtype)
        )
        old_out = post.to(x.dtype).unsqueeze(-1) * x.unsqueeze(-2) + mixed_bf16
        err_ref = (gt - old_out.float()).abs().mean()
        err_fused = (gt - hc_post_fused(x, residual, post, comb).float()).abs().mean()
        assert err_fused <= err_ref * 1.05

    @pytest.mark.gpu
    def test_backward_matches_reference(self):
        # 四个梯度（x/residual/post/comb）都要与 eager 参考一致。
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
        # 自定义算子要能进 fullgraph 编译区，否则调用方会整块回退。
        from xtuner.v1.ops.hc_post import hc_post_fused

        torch.manual_seed(3)
        x, residual, post, comb = _make_inputs(1, 1024, 4, 4096, "cuda", requires_grad=True)
        grad_out = torch.randn(1, 1024, 4, 4096, device="cuda", dtype=torch.bfloat16)

        eager = hc_post_fused(x, residual, post, comb)

        compiled_fn = torch.compile(hc_post_fused, fullgraph=True)
        x2, residual2, post2, comb2 = [t.detach().clone().requires_grad_(True) for t in (x, residual, post, comb)]
        compiled = compiled_fn(x2, residual2, post2, comb2)
        # custom_op is opaque to dynamo → forward must be bit-identical to eager.
        assert torch.equal(eager, compiled)
        expected_grads = torch.autograd.grad(eager, (x, residual, post, comb), grad_out)
        actual_grads = torch.autograd.grad(compiled, (x2, residual2, post2, comb2), grad_out)
        for actual, expected in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)


class TestHCPostFP32Precision:
    @pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
    def test_coefficients_survive_cancellation(self, device: str) -> None:
        from xtuner.v1.module.decoder_layer.mhc import hc_post

        # All coefficients are positive; rounding them to BF16 loses this signal.
        x = torch.ones(1, 1, 3, device=device, dtype=torch.bfloat16, requires_grad=True)
        residual = -torch.ones(1, 1, 2, 3, device=device, dtype=torch.bfloat16, requires_grad=True)
        residual = residual.detach().requires_grad_(True)
        post = torch.tensor([[[1.003, 1.0]]], device=device, requires_grad=True)
        comb = torch.tensor([[[[0.5, 0.5], [0.5, 0.5]]]], device=device, requires_grad=True)
        out = hc_post(x, residual, post, comb)
        expected = torch.tensor([[[[0.003] * 3, [0.0] * 3]]], device=device, dtype=torch.bfloat16)
        torch.testing.assert_close(out, expected, atol=0, rtol=0)
        grad_out = torch.tensor([[[[1.0] * 3, [-1.0] * 3]]], device=device, dtype=torch.bfloat16)
        grads = torch.autograd.grad(out, (x, residual, post, comb), grad_out)
        torch.testing.assert_close(grads[0], torch.full_like(x, 0.003), atol=0, rtol=0)
        torch.testing.assert_close(grads[1], torch.zeros_like(residual), atol=0, rtol=0)
        torch.testing.assert_close(grads[2], torch.tensor([[[3.0, -3.0]]], device=device), atol=0, rtol=0)
        torch.testing.assert_close(
            grads[3], torch.tensor([[[[-3.0, 3.0], [-3.0, 3.0]]]], device=device), atol=0, rtol=0
        )

    @pytest.mark.gpu
    @pytest.mark.parametrize("H,D", [(1, 31), (2, 257), (4, 4096), (8, 2053)])
    @pytest.mark.parametrize("noncontiguous", [False, True])
    def test_forward_and_all_gradients_against_fp64(self, H: int, D: int, noncontiguous: bool) -> None:
        from xtuner.v1.module.decoder_layer.mhc import _hc_post_eager, hc_post

        torch.manual_seed(42)
        inputs = _make_inputs(2, 3, H, D, "cuda", requires_grad=False)
        if noncontiguous:
            inputs = [t.transpose(0, 1).contiguous().transpose(0, 1) for t in inputs]
            assert all(not t.is_contiguous() for t in inputs)
        x, residual, post, comb = [t.double() for t in inputs]
        grad = torch.randn_like(inputs[1])
        g = grad.double()
        truth = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.einsum("...ij,...id->...jd", comb, residual)
        # Independent analytic VJP, not the autograd of the implementation under test.
        expected_grads = (
            (post.unsqueeze(-1) * g).sum(-2),
            torch.einsum("...ij,...jd->...id", comb, g),
            (x.unsqueeze(-2) * g).sum(-1),
            torch.einsum("...id,...jd->...ij", residual, g),
        )
        for fn in (hc_post, _hc_post_eager):
            values = [t.detach().requires_grad_(True) for t in inputs]
            out = fn(*values)
            torch.testing.assert_close(out, truth.to(out.dtype), atol=2e-5, rtol=8e-3)
            grads = torch.autograd.grad(out, values, grad)
            for actual, expected, value in zip(grads, expected_grads, values):
                assert actual.dtype == value.dtype
                assert torch.isfinite(actual).all()
                tol = 8e-3 if actual.dtype == torch.bfloat16 else 2e-5
                torch.testing.assert_close(actual, expected.to(actual.dtype), atol=2e-5, rtol=tol)

    @pytest.mark.gpu
    @pytest.mark.parametrize("compiled", [False, True])
    def test_hc_pre_post_parameter_gradients(self, compiled: bool) -> None:
        from xtuner.v1.module.decoder_layer.mhc import hc_post, hc_pre

        torch.manual_seed(29)
        H, D = 4, 64
        mix_dim = (H + 2) * H
        inputs = (
            torch.randn(1, 17, H, D, device="cuda", dtype=torch.bfloat16),
            0.02 * torch.randn(mix_dim, H * D, device="cuda"),
            torch.tensor([0.7, 0.6, 0.5], device="cuda"),
            0.1 * torch.randn(mix_dim, device="cuda"),
            0.02 * torch.randn(D, D, device="cuda", dtype=torch.bfloat16),
        )

        def forward(
            streams: torch.Tensor,
            fn: torch.Tensor,
            scale: torch.Tensor,
            base: torch.Tensor,
            block_weight: torch.Tensor,
        ) -> torch.Tensor:
            collapsed, post, comb = hc_pre(streams, fn, scale, base, H, 20, 1e-6, 1e-6)
            x = torch.nn.functional.linear(collapsed, block_weight)
            return hc_post(x, streams, post, comb)

        def reference(
            streams: torch.Tensor,
            fn: torch.Tensor,
            scale: torch.Tensor,
            base: torch.Tensor,
            block_weight: torch.Tensor,
        ) -> torch.Tensor:
            collapsed, post, comb = hc_pre(streams, fn, scale, base, H, 20, 1e-6, 1e-6)
            x = torch.nn.functional.linear(collapsed, block_weight)
            # Independent contraction; checks coefficient orientation and chain rule.
            out = post.unsqueeze(-1) * x.float().unsqueeze(-2)
            return (out + torch.einsum("...ij,...id->...jd", comb, streams.float())).to(streams.dtype)

        values = [t.detach().requires_grad_(True) for t in inputs]
        refs = [t.detach().clone().requires_grad_(True) for t in inputs]
        fn = torch.compile(forward, fullgraph=True) if compiled else forward
        actual = fn(*values)
        reference_fn = torch.compile(reference, fullgraph=True) if compiled else reference
        expected = reference_fn(*refs)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=8e-3)
        grad = torch.randn_like(actual)
        actual_grads = torch.autograd.grad(actual, values, grad)
        expected_grads = torch.autograd.grad(expected, refs, grad)
        for actual, expected in zip(actual_grads, expected_grads):
            assert torch.isfinite(actual).all()
            if actual.dtype == torch.bfloat16:
                # Compiled pre/post chains can fuse BF16 gradient additions differently.
                # Use normwise bounds near cancellation; the isolated VJP test above
                # checks all four post gradients against independent FP64 formulas.
                error = (actual.float() - expected.float()).abs()
                assert error.norm() <= 5e-3 * expected.float().norm()
                assert error.max() <= 1e-2 * expected.float().abs().max()
            else:
                torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-2)
