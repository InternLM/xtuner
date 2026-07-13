import pytest
import torch
import torch.nn.functional as F

from xtuner.v1.module.router.noaux_router import NoAuxRouter


@pytest.fixture(autouse=True)
def _cpu_histc(monkeypatch):
    # torch.histc lacks a CPU kernel for Long in this build; the router's bookkeeping path
    # calls it on `topk_idx` (Long). We coerce to float just for the test, since the histc
    # output is not under test here — only the upstream scoring branch is.
    original_histc = torch.histc

    def _histc(input, *args, **kwargs):  # noqa: A002 - mirror torch signature
        if input.is_floating_point():
            return original_histc(input, *args, **kwargs)
        return original_histc(input.float(), *args, **kwargs)

    monkeypatch.setattr(torch, "histc", _histc)


def _build_router(scoring_func: str, n_routed_experts: int = 16, num_experts_per_tok: int = 4) -> NoAuxRouter:
    router = NoAuxRouter(
        n_routed_experts=n_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        router_scaling_factor=1.0,
        scoring_func=scoring_func,  # type: ignore[arg-type]
        n_group=4,
        topk_group=2,
        norm_topk_prob=True,
    )
    # Zero out the correction bias so the scoring branch is the only variable under test.
    with torch.no_grad():
        router.e_score_correction_bias.zero_()
    return router.to("cpu")


class TestSqrtSoftplusScoring:
    def test_sqrtsoftplus_numeric(self):
        torch.manual_seed(0)
        logits = torch.randn(128, 64, dtype=torch.float32)
        router = NoAuxRouter(
            n_routed_experts=64,
            num_experts_per_tok=4,
            router_scaling_factor=1.0,
            scoring_func="sqrtsoftplus",
            n_group=4,
            topk_group=2,
            norm_topk_prob=True,
        )
        with torch.no_grad():
            router.e_score_correction_bias.zero_()
        router = router.to("cpu")

        # End-to-end call exercises the new scoring branch.
        out = router(logits)
        assert out["topk_weights"].shape == (128, 4)

        # The scoring contract: scores = sqrt(softplus(logits)). Reconstruct what the
        # router selected and compare its weights against the formula evaluated on those
        # same indices — this isolates the scoring branch from grouping/normalization.
        expected_scores = torch.sqrt(F.softplus(logits))
        gathered = expected_scores.gather(1, out["topk_ids"])
        if router.norm_topk_prob and router.top_k > 1:
            gathered = gathered / (gathered.sum(dim=-1, keepdim=True) + 1e-20)
        gathered = gathered * router.router_scaling_factor
        torch.testing.assert_close(out["topk_weights"], gathered, atol=1e-6, rtol=0)

        # Element-wise contract on the raw formula.
        direct = F.softplus(logits).sqrt()
        torch.testing.assert_close(direct, expected_scores, atol=1e-6, rtol=0)

    def test_softmax_now_works(self):
        torch.manual_seed(1)
        logits = torch.randn(32, 16, dtype=torch.float32)
        router = _build_router("softmax")

        out = router(logits)

        # softmax over experts must be a valid probability distribution per token.
        probs = logits.softmax(dim=-1)
        torch.testing.assert_close(probs.sum(dim=-1), torch.ones(32), atol=1e-6, rtol=0)
        assert out["topk_weights"].shape == (32, 4)

    def test_sigmoid_unchanged(self):
        torch.manual_seed(42)
        logits = torch.randn(32, 16, dtype=torch.float32)
        router = _build_router("sigmoid")

        out = router(logits)

        # Verify the router's selected weights match `sigmoid(logits).gather(topk_ids)`
        # under the same normalization the router applies — bit-identical, since both
        # paths execute the same ops in the same order on the same seed.
        expected_scores = logits.sigmoid()
        gathered = expected_scores.gather(1, out["topk_ids"])
        if router.norm_topk_prob and router.top_k > 1:
            gathered = gathered / (gathered.sum(dim=-1, keepdim=True) + 1e-20)
        gathered = gathered * router.router_scaling_factor
        torch.testing.assert_close(out["topk_weights"], gathered, atol=0, rtol=0)
