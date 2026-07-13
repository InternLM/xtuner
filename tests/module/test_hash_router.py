# Copyright (c) OpenMMLab. All rights reserved.
"""CPU-only unit tests for :class:`HashRouter` and its protocol compatibility."""
import pytest
import torch

from xtuner.v1.module.router import HashRouter, NoAuxRouter
from xtuner.v1.module.router.protocol import RouterResults


def _make_router(
    *, vocab_size: int = 64, n_routed_experts: int = 8, num_experts_per_tok: int = 2, seed: int = 0
) -> HashRouter:
    """Helper that builds a HashRouter with a deterministically populated tid2eid table."""
    torch.manual_seed(seed)
    router = HashRouter(
        vocab_size=vocab_size,
        n_routed_experts=n_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
    )
    router.tid2eid.copy_(torch.randint(0, n_routed_experts, (vocab_size, num_experts_per_tok), dtype=torch.int32))
    return router


class TestHashRouter:
    def test_deterministic_routing(self):
        router = _make_router()
        input_ids = torch.tensor([0, 3, 7, 15, 31, 63, 5, 5], dtype=torch.long)

        first: RouterResults = router(logits=None, input_ids=input_ids)
        second: RouterResults = router(logits=None, input_ids=input_ids)

        assert torch.equal(first["topk_ids"], second["topk_ids"])
        # Same token id must always map to the same expert tuple.
        assert torch.equal(first["topk_ids"][6], first["topk_ids"][7])

    def test_uniform_topk_weights(self):
        num_experts_per_tok = 2
        router = _make_router(num_experts_per_tok=num_experts_per_tok)
        input_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)

        results = router(logits=None, input_ids=input_ids)
        topk_weights = results["topk_weights"]

        expected = torch.full_like(topk_weights, 1.0 / num_experts_per_tok)
        assert torch.allclose(topk_weights, expected)
        assert torch.allclose(topk_weights.sum(dim=-1), torch.ones(input_ids.shape[0]))

    def test_topk_ids_within_range(self):
        n_routed_experts = 8
        router = _make_router(n_routed_experts=n_routed_experts)
        input_ids = torch.arange(64, dtype=torch.long)

        results = router(logits=None, input_ids=input_ids)

        assert results["topk_ids"].min().item() >= 0
        assert results["topk_ids"].max().item() < n_routed_experts

    def test_packed_input_ids(self):
        num_experts_per_tok = 2
        total_tokens = 17
        router = _make_router(num_experts_per_tok=num_experts_per_tok)
        input_ids = torch.randint(0, 64, (total_tokens,), dtype=torch.long)

        results = router(logits=None, input_ids=input_ids)

        assert results["topk_ids"].shape == (total_tokens, num_experts_per_tok)
        assert results["topk_weights"].shape == (total_tokens, num_experts_per_tok)

    def test_protocol_compatibility(self):
        """NoAuxRouter must accept the new `input_ids` kwarg and ignore it."""
        torch.manual_seed(0)
        n_routed_experts = 8
        num_experts_per_tok = 2
        noaux = NoAuxRouter(
            n_routed_experts=n_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            router_scaling_factor=1.0,
            scoring_func="sigmoid",
            n_group=1,
            topk_group=1,
            norm_topk_prob=True,
        )
        # `e_score_correction_bias` is registered on the default device; clear it explicitly
        # so the test stays deterministic. Use the buffer's own device so the test works
        # regardless of whether CUDA is available.
        device = noaux.e_score_correction_bias.device
        noaux.e_score_correction_bias.zero_()

        logits = torch.randn(5, n_routed_experts, device=device)
        fake_input_ids = torch.randint(0, 64, (5,), dtype=torch.long, device=device)

        without_ids = noaux(logits)
        with_ids = noaux(logits, input_ids=fake_input_ids)

        assert torch.equal(without_ids["topk_ids"], with_ids["topk_ids"])
        assert torch.allclose(without_ids["topk_weights"], with_ids["topk_weights"])

    def test_missing_input_ids_raises(self):
        router = _make_router()
        with pytest.raises(AssertionError, match="HashRouter requires `input_ids`"):
            router(logits=None, input_ids=None)
