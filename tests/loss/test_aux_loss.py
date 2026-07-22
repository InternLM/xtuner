import torch

from xtuner.v1.loss import AuxLossConfig
from xtuner.v1.module.router import NoAuxRouterConfig


def test_aux_loss_counts_experts_selected_by_grouped_router():
    n_routed_experts = 16
    router = NoAuxRouterConfig(
        n_group=1,
        topk_group=1,
        scoring_func="sigmoid",
        norm_topk_prob=True,
        router_scaling_factor=1.0,
        use_grouped_router=True,
        router_n_groups=8,
    ).build(n_routed_experts=n_routed_experts, num_experts_per_tok=8)
    router.e_score_correction_bias.zero_()

    # Grouped routing selects one expert from each adjacent pair. A second
    # global Top-K over router weights would instead select experts 0 through 7.
    router_results = router(torch.arange(10.0, -6.0, -1.0).unsqueeze(0))
    selected_experts = router_results["topk_ids"]
    assert selected_experts.tolist() == [[0, 2, 4, 6, 8, 10, 12, 14]]

    aux_loss = AuxLossConfig(
        n_routed_experts=n_routed_experts,
        num_experts_per_tok=8,
    ).build()
    aux_loss.accumulate(
        selected_router_weights=router_results["router_weights"],
        selected_router_logits=router_results["logits"],
        selected_experts=selected_experts,
        hidden_states=torch.zeros(1),
    )
    _, _, tokens_per_expert = aux_loss.finalize(
        balancing_ctx=None,
        z_ctx=None,
        non_pad_token=1,
    )

    expected = torch.zeros(n_routed_experts, dtype=torch.long)
    expected[selected_experts.flatten()] = 1
    torch.testing.assert_close(tokens_per_expert, expected.unsqueeze(0))


def test_aux_loss_counts_rollout_assignments_returned_by_noaux_router():
    router = NoAuxRouterConfig(
        n_group=1,
        topk_group=1,
        scoring_func="sigmoid",
        norm_topk_prob=True,
        router_scaling_factor=1.0,
    ).build(n_routed_experts=4, num_experts_per_tok=2)
    router.e_score_correction_bias.zero_()

    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    rollout_assignments = torch.tensor([[2, 3]])
    router_results = router(logits, rollout_assignments)
    expected_weights = logits.sigmoid().gather(1, rollout_assignments)
    expected_weights /= expected_weights.sum(dim=-1, keepdim=True)

    torch.testing.assert_close(router_results["topk_ids"], rollout_assignments)
    torch.testing.assert_close(router_results["topk_weights"], expected_weights)
    torch.testing.assert_close(router_results["topkens_per_expert"], torch.tensor([0.0, 0.0, 1.0, 1.0]))

    aux_loss = AuxLossConfig(n_routed_experts=4, num_experts_per_tok=2).build()
    aux_loss.accumulate(
        selected_router_weights=router_results["router_weights"],
        selected_router_logits=router_results["logits"],
        selected_experts=router_results["topk_ids"],
        hidden_states=torch.zeros(1),
    )
    _, _, tokens_per_expert = aux_loss.finalize(
        balancing_ctx=None,
        z_ctx=None,
        non_pad_token=1,
    )

    torch.testing.assert_close(tokens_per_expert, torch.tensor([[0, 0, 1, 1]]))
