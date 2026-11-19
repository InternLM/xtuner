# Copyright (c) OpenMMLab. All rights reserved.
"""Same-input forward, independent VJP and optimizer checks for GLM alignment."""

import pytest
import torch
from torch.nn import functional as F

from xtuner.v1.ops import lmdeploy_alignment as alignment


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("strided", [False, True])
def test_mhc_backward_row_chunks_match_full_vjp(monkeypatch, strided):
    """Changing row workspace size must retain the same mathematical VJP."""
    torch.manual_seed(332)
    rows, h, d = 129, 4, 128
    inputs = (torch.randn(1, rows, h, d, device="cuda", dtype=torch.bfloat16),
              torch.randn(24, h * d, device="cuda") * 0.01,
              torch.tensor([1.0, 0.5, 0.5], device="cuda"),
              torch.zeros(24, device="cuda"),
              torch.ones(d, device="cuda", dtype=torch.bfloat16))
    if strided:
        inputs = (inputs[0].transpose(1, 2).contiguous().transpose(1, 2), *inputs[1:])
    seeds = (torch.randn(1, rows, d, device="cuda", dtype=torch.bfloat16),
             torch.randn(1, rows, h, device="cuda"), torch.randn(1, rows, h, h, device="cuda"))
    monkeypatch.setattr(alignment, "_HC_BACKWARD_CHUNK_ROWS", rows)
    expected = alignment._hc_backward_chunked(*seeds, *inputs, 20, 1e-6, 1e-5)
    monkeypatch.setattr(alignment, "_HC_BACKWARD_CHUNK_ROWS", 17)
    actual = alignment._hc_backward_chunked(*seeds, *inputs, 20, 1e-6, 1e-5)
    for grad, ref in zip(actual, expected):
        relative = (grad.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-20)
        assert relative < 0.003
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0


@pytest.fixture(autouse=True)
def aligned(monkeypatch):
    from lmdeploy.pytorch.nn import glm53_alignment

    from xtuner.v1.module.decoder_layer import mhc
    from xtuner.v1.module.router import noaux_router
    from xtuner.v1.ops import hc_post

    monkeypatch.setattr(alignment, "ALIGN_FORWARD", True)
    monkeypatch.setattr(glm53_alignment, "ALIGN_FORWARD", True)
    monkeypatch.setattr(mhc, "ALIGN_FORWARD", True)
    monkeypatch.setattr(noaux_router, "ALIGN_FORWARD", True)
    monkeypatch.setattr(hc_post, "ALIGN_FORWARD", True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(42)


@pytest.mark.parametrize("rows", [1, 6, 32, 33, 65])
def test_head_batch_and_loss_paths(rows):
    from lmdeploy.pytorch.models.glm5_next import Glm5NextForConditionalGeneration

    from xtuner.v1.loss.rl_loss import LogProbConfig, LogProbContext, LogProbKwargs
    from xtuner.v1.rl.loss.grpo_loss import GRPOLossConfig, GRPOLossContext, GRPOLossKwargs

    x = torch.randn(1, rows, 128, device="cuda", requires_grad=True)
    head = torch.nn.Linear(128, 1024, device="cuda")
    model = torch.nn.Module()
    model.lm_head = head
    logits = Glm5NextForConditionalGeneration.get_logits(model, x)
    alone = Glm5NextForConditionalGeneration.get_logits(model, x[:, :1])
    assert torch.equal(logits[:, :1], alone)
    labels = torch.randint(0, 1024, (1, rows), device="cuda")
    expected = logits.log_softmax(-1).gather(-1, labels[..., None]).squeeze(-1)
    cfg = LogProbConfig(mode="chunk", chunk_size=6)
    read = LogProbContext(cfg, LogProbKwargs(shifted_labels=labels))
    actual, _ = read.loss_fn(x, head.weight, head.bias, read.loss_kwargs)
    chunked, _ = read.chunk_mode(x, head.weight, head.bias, read.loss_kwargs)
    assert torch.equal(actual, expected)
    assert torch.equal(chunked, expected)
    cfg = GRPOLossConfig(policy_loss_cfg=dict(loss_type="vanilla", cliprange_low=0.2, cliprange_high=0.2))
    kwargs = GRPOLossKwargs(
        shifted_labels=labels,
        old_logprobs=expected.detach(),
        advantages=torch.ones_like(expected),
        policy_loss_weight=torch.ones_like(expected) / rows,
    )
    ctx = GRPOLossContext(cfg, kwargs)
    loss, (train_logits, _) = ctx.loss_fn(x, head.weight, head.bias, kwargs)
    assert torch.equal(train_logits, logits)
    loss.backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    before = head.weight.detach().clone()
    torch.optim.SGD(head.parameters(), lr=0.01).step()
    assert not torch.equal(before, head.weight)


def test_production_width_head_logprob_and_policy_loss_paths():
    from lmdeploy.pytorch.models.glm5_next import Glm5NextForConditionalGeneration

    from xtuner.v1.loss.rl_loss import LogProbConfig, LogProbContext, LogProbKwargs
    from xtuner.v1.rl.loss.grpo_loss import GRPOLossConfig, GRPOLossContext, GRPOLossKwargs

    x = torch.randn(1, 33, 4096, device="cuda", requires_grad=True)
    head = torch.nn.Linear(4096, 4097, device="cuda")
    model = torch.nn.Module()
    model.lm_head = head
    rollout_logits = Glm5NextForConditionalGeneration.get_logits(model, x)
    labels = torch.randint(0, 4097, (1, 33), device="cuda")
    expected = rollout_logits.log_softmax(-1).gather(-1, labels[..., None]).squeeze(-1)
    for chunk in (1, 6, 17):
        ctx = LogProbContext(LogProbConfig(mode="chunk", chunk_size=chunk),
                             LogProbKwargs(shifted_labels=labels))
        actual, _ = ctx.chunk_mode(x, head.weight, head.bias, ctx.loss_kwargs)
        assert torch.equal(actual, expected)
    kwargs = GRPOLossKwargs(shifted_labels=labels, old_logprobs=expected.detach(),
                           advantages=torch.ones_like(expected),
                           policy_loss_weight=torch.ones_like(expected)/33)
    ctx = GRPOLossContext(GRPOLossConfig(policy_loss_cfg=dict(
        loss_type="vanilla", cliprange_low=0.2, cliprange_high=0.2)), kwargs)
    loss, (train_logits, _) = ctx.loss_fn(x, head.weight, head.bias, kwargs)
    assert torch.equal(train_logits, rollout_logits)
    gradients = torch.autograd.grad(loss, (x, head.weight))
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)


@pytest.mark.parametrize("rows", [1, 6, 33, 65])
def test_mhc_forward_batch_and_vjp(rows):
    from lmdeploy.pytorch.backends.cuda.hc_prepost import TritonHCPrePostImpl
    from lmdeploy.pytorch.nn.glm53_alignment import hc_pre as serving_pre

    from xtuner.v1.module.decoder_layer import mhc

    # Keep the production geometry: four streams, hidden 4096, twenty iterations.
    h, d = 4, 4096
    x = torch.randn(1, rows, h, d, device="cuda", dtype=torch.bfloat16).requires_grad_()
    fn = (torch.randn(24, h * d, device="cuda") * 0.01).requires_grad_()
    scale = torch.tensor([1.0, 0.5, 0.5], device="cuda", requires_grad=True)
    base = (torch.randn(24, device="cuda") * 0.1).requires_grad_()
    norm = torch.ones(d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    args = (x, fn, scale, base, norm, 20, 1e-6, 1e-5)
    actual = alignment.hc_pre(*args)
    expected = serving_pre(*args)
    for a, b, c in zip(actual, expected, serving_pre(x[:, :1], *args[1:])):
        assert torch.equal(a, b)
        assert torch.equal(a[:, :1], c)
    from xtuner.v1.module.rms_norm.rms_norm import RMSNorm

    norm_module = RMSNorm(d, eps=1e-5).to(device="cuda", dtype=torch.bfloat16)
    integrated = mhc.hc_pre_norm(x, fn, scale, base, mhc.MHCConfig(hc_mult=4), norm_module)
    for a, b in zip(actual, integrated):
        assert torch.equal(a, b)
    seeds = [torch.randn_like(v) for v in actual]
    grads = torch.autograd.grad(actual, args[:5], seeds)
    for grad in grads:
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    # FP64 reference uses the mathematical equations, independent of custom backward.
    xd, wd, sd, bd, nd = [v.detach().double().requires_grad_() for v in args[:5]]
    flat = xd.flatten(-2)
    mixes = F.linear(F.rms_norm(flat, (h * d,), eps=1e-5), wd)
    pre = (mixes[..., :h] * sd[0] + bd[:h]).sigmoid() + 1e-6
    post = 2 * (mixes[..., h : 2 * h] * sd[1] + bd[h : 2 * h]).sigmoid()
    comb = (mixes[..., 2 * h :] * sd[2] + bd[2 * h :]).reshape(1, rows, h, h).softmax(-1) + 1e-6
    comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
    for _ in range(19):
        comb = comb / (comb.sum(-1, keepdim=True) + 1e-6)
        comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
    reduced = (pre[..., None] * xd).sum(-2)
    # Quantization is STE: inject the rounded value while preserving its derivative.
    reduced = reduced + (reduced.to(torch.bfloat16).double() - reduced).detach()
    y = F.rms_norm(reduced, (d,), eps=1e-5) * nd
    oracle = torch.autograd.grad((y, post, comb), (xd, wd, sd, bd, nd), [v.double() for v in seeds])
    for grad, ref in zip(grads, oracle):
        torch.testing.assert_close(grad.double(), ref.to(grad.dtype).double(), rtol=0.02, atol=0.02)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(grads[1].double() * 1.25, oracle[1], rtol=0.02, atol=0.02)
    out = torch.randn_like(actual[0], requires_grad=True)
    result = mhc.hc_post(out, x, actual[1], actual[2])
    expected = TritonHCPrePostImpl(4, 20, 1e-6).post_expand(out, x, actual[1], actual[2])
    assert torch.equal(result, expected)


@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_compiled_mhc_and_router_backward(backend, monkeypatch):
    monkeypatch.setattr(alignment, "_HC_BACKWARD_CHUNK_ROWS", 1)
    x = torch.randn(1, 2, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    fn = (torch.randn(24, 256, device="cuda") * 0.01).requires_grad_()
    scale = torch.ones(3, device="cuda", requires_grad=True)
    base = torch.zeros(24, device="cuda", requires_grad=True)
    norm = torch.ones(64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    compiled = torch.compile(alignment.hc_pre, backend=backend, fullgraph=True)
    y, post, comb = compiled(x, fn, scale, base, norm, 20, 1e-6, 1e-5)
    (y.float().square().mean() + post.square().mean() + comb.square().mean()).backward()
    assert torch.isfinite(fn.grad).all() and fn.grad.abs().sum() > 0
    logits = torch.randn(6, 288, device="cuda", requires_grad=True)
    compiled_router = torch.compile(alignment.router, backend=backend, fullgraph=True)
    weights, _ = compiled_router(logits, torch.zeros(288, device="cuda"), None)
    weights.square().sum().backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


def test_mhc_post_vjp():
    from xtuner.v1.module.decoder_layer.mhc import hc_post

    x = torch.randn(1, 6, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    residual = torch.randn(1, 6, 4, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    post = torch.randn(1, 6, 4, device="cuda", requires_grad=True)
    comb = torch.randn(1, 6, 4, 4, device="cuda", requires_grad=True)
    args = (x, residual, post, comb)
    output = hc_post(*args)
    seed = torch.randn_like(output)
    grads = torch.autograd.grad(output, args, seed)
    xd, rd, pd, cd = [v.detach().double().requires_grad_() for v in args]
    reference = pd[..., None] * xd[..., None, :] + cd.transpose(-1, -2) @ rd
    oracle = torch.autograd.grad(reference, (xd, rd, pd, cd), seed.double())
    for grad, ref in zip(grads, oracle):
        torch.testing.assert_close(grad, ref.to(grad.dtype), rtol=2e-4, atol=2e-4)


def test_production_head_geometry_and_empty():
    from lmdeploy.pytorch.nn.glm53_alignment import aligned_linear

    weight = torch.randn(154880, 4096, device="cuda") * 0.01
    x = torch.randn(1, 33, 4096, device="cuda")
    result = aligned_linear(x, weight)
    assert torch.equal(result[:, :1], aligned_linear(x[:, :1], weight))
    assert torch.equal(result[:, :6], aligned_linear(x[:, :6], weight))
    assert aligned_linear(x[:, :0], weight).shape == (1, 0, 154880)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_projection_vjp_and_strides(dtype):
    from lmdeploy.pytorch.nn.glm53_alignment import aligned_linear

    x = torch.randn(1, 34, 65, device="cuda", dtype=dtype)[:, ::2].requires_grad_()
    weight = torch.randn(65, 77, device="cuda", dtype=dtype).t().requires_grad_()
    bias = torch.randn(77, device="cuda", dtype=dtype, requires_grad=True)
    actual = aligned_linear(x, weight, bias)
    seed = torch.randn_like(actual)
    grads = torch.autograd.grad(actual, (x, weight, bias), seed)
    xd, wd, bd = [v.detach().double().requires_grad_() for v in (x, weight, bias)]
    ref = F.linear(xd, wd, bd)
    oracle = torch.autograd.grad(ref, (xd, wd, bd), seed.double())
    tol = 2e-2 if dtype == torch.bfloat16 else 2e-5
    torch.testing.assert_close(actual, ref.to(dtype), rtol=tol, atol=tol)
    for grad, expected in zip(grads, oracle):
        torch.testing.assert_close(grad, expected.to(dtype), rtol=tol, atol=tol)
    with pytest.raises(ValueError, match="feature dimensions"):
        aligned_linear(x, weight[:, :-1])


@pytest.mark.parametrize("forced", [False, True])
def test_router_vjp_ties_and_r3(forced):
    from lmdeploy.pytorch.kernels.cuda.moe.route_noaux_tc import fused_noaux_tc_routing

    x = torch.randn(33, 288, device="cuda", requires_grad=True)
    bias = torch.zeros(288, device="cuda")
    ids = torch.arange(8, device="cuda").expand(33, -1).contiguous() if forced else None
    weights, chosen = alignment.router(x, bias, ids)
    expected, expected_ids = fused_noaux_tc_routing(
        x, bias, num_experts=288, n_group=1, topk_group=1, top_k=8, routed_scaling_factor=1.0, forced_ids=ids
    )
    assert torch.equal(weights, expected) and torch.equal(chosen, expected_ids)
    seed = torch.randn_like(weights)
    (grad,) = torch.autograd.grad(weights, x, seed)
    ref = x.detach().double().requires_grad_()
    score = ref.sigmoid().gather(1, chosen)
    normalized = score / (score.sum(-1, keepdim=True) + 1e-20)
    (oracle,) = torch.autograd.grad(normalized, ref, seed.double())
    torch.testing.assert_close(grad.double(), oracle, rtol=1e-4, atol=1e-7)
    _, tied = alignment.router(torch.zeros_like(x), bias, None)
    assert torch.equal(tied, torch.arange(8, device="cuda").expand_as(tied))
    from xtuner.v1.module.router.noaux_router import NoAuxRouter

    module = NoAuxRouter(
        n_routed_experts=288,
        num_experts_per_tok=8,
        router_scaling_factor=2.5,
        scoring_func="sigmoid",
        n_group=1,
        topk_group=1,
    ).to("cuda")
    module.e_score_correction_bias.zero_()
    result = module(x, ids)
    assert torch.equal(result["topk_ids"], chosen)
    assert torch.equal(result["topk_weights"], expected * 2.5)

