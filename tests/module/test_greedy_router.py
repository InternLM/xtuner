import torch

from xtuner.v1.module.router.greedy import GreedyRouterConfig


def test_force_load_balance_is_opt_in():
    config = GreedyRouterConfig(
        scoring_func="sigmoid",
        router_scaling_factor=1.0,
        norm_topk_prob=True,
    )

    assert config.force_load_balance is False
    assert config.build(n_routed_experts=32, num_experts_per_tok=4).force_load_balance is False


def test_force_load_balance_randomizes_routing_and_preserves_gradient():
    torch.manual_seed(0)
    logits = torch.zeros(16384, 32, requires_grad=True)
    router = GreedyRouterConfig(
        scoring_func="sigmoid",
        router_scaling_factor=1.0,
        norm_topk_prob=True,
        force_load_balance=True,
    ).build(n_routed_experts=32, num_experts_per_tok=4)

    router_results = router(logits)
    counts = torch.bincount(router_results["topk_ids"].flatten(), minlength=32).float()

    assert not torch.equal(router_results["logits"], logits)
    assert counts.max() / counts.mean() < 1.1
    router_results["router_weights"].sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.norm() > 0


def test_force_load_balance_random_logits_is_torch_compile_compatible():
    torch.manual_seed(0)
    logits = torch.zeros(8, 4)
    router = GreedyRouterConfig(
        scoring_func="sigmoid",
        router_scaling_factor=1.0,
        norm_topk_prob=True,
        force_load_balance=True,
    ).build(n_routed_experts=4, num_experts_per_tok=2)
    compiled_router = torch.compile(router, backend="eager", fullgraph=True)

    router_results = compiled_router(logits)

    assert router_results["logits"].shape == logits.shape
    assert not torch.equal(router_results["logits"], logits)
