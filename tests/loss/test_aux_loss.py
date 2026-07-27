"""AuxLoss 使用真实路由 assignment 的行为测试。

TestAuxLossRouterAssignments
    test_grouped_router_assignments_are_counted: grouped router 的实际专家选择进入负载统计。
    test_rollout_assignments_are_counted: rollout 指定的专家选择进入权重与负载统计。
"""

import torch

from xtuner.v1.loss import AuxLossConfig
from xtuner.v1.module.router import NoAuxRouterConfig


def _build_router(n_routed_experts: int, num_experts_per_tok: int, **kwargs):
    router = NoAuxRouterConfig(
        n_group=1,
        topk_group=1,
        scoring_func="sigmoid",
        norm_topk_prob=True,
        router_scaling_factor=1.0,
        **kwargs,
    ).build(
        n_routed_experts=n_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
    )
    router.e_score_correction_bias.zero_()
    return router


def _tokens_per_expert(router_results, n_routed_experts: int, num_experts_per_tok: int):
    aux_loss = AuxLossConfig(
        n_routed_experts=n_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
    ).build()
    aux_loss.accumulate(
        selected_router_weights=router_results["router_weights"],
        selected_router_logits=router_results["logits"],
        selected_experts=router_results["topk_ids"],
        hidden_states=torch.zeros(1, device=router_results["topk_ids"].device),
    )
    return aux_loss.finalize(
        balancing_ctx=None,
        z_ctx=None,
        non_pad_token=1,
    )[2]


class TestAuxLossRouterAssignments:
    def test_grouped_router_assignments_are_counted(self):
        # 验证负载统计采用 grouped router 的真实选择，而不是对权重再次做全局 Top-K。
        n_routed_experts = 16
        router = _build_router(
            n_routed_experts,
            num_experts_per_tok=8,
            use_grouped_router=True,
            router_n_groups=8,
        )

        device = router.e_score_correction_bias.device
        router_results = router(torch.arange(10.0, -6.0, -1.0, device=device).unsqueeze(0))
        selected_experts = router_results["topk_ids"]
        assert selected_experts.tolist() == [[0, 2, 4, 6, 8, 10, 12, 14]]

        expected = torch.zeros(n_routed_experts, dtype=torch.long, device=device)
        expected[selected_experts.flatten()] = 1
        torch.testing.assert_close(
            _tokens_per_expert(router_results, n_routed_experts, num_experts_per_tok=8),
            expected.unsqueeze(0),
        )

    def test_rollout_assignments_are_counted(self):
        # 验证 rollout assignment 同时决定路由权重和 AuxLoss 的专家计数。
        router = _build_router(n_routed_experts=4, num_experts_per_tok=2)
        device = router.e_score_correction_bias.device
        logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]], device=device)
        rollout_assignments = torch.tensor([[2, 3]], device=device)

        router_results = router(logits, rollout_assignments)
        expected_weights = logits.sigmoid().gather(1, rollout_assignments)
        expected_weights /= expected_weights.sum(dim=-1, keepdim=True)

        torch.testing.assert_close(router_results["topk_ids"], rollout_assignments)
        torch.testing.assert_close(router_results["topk_weights"], expected_weights)
        torch.testing.assert_close(
            _tokens_per_expert(router_results, n_routed_experts=4, num_experts_per_tok=2),
            torch.tensor([[0, 0, 1, 1]], device=device),
        )
