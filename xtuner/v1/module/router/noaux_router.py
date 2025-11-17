import os
from typing import Annotated, Literal

import torch
import torch.nn as nn
from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict

from xtuner.v1.utils.device import get_device

from .protocol import RouterProtocol, RouterResults


class NoAuxRouterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scoring_func: Annotated[Literal["sigmoid", "softmax"], Parameter(group="router")]
    router_scaling_factor: Annotated[float, Parameter(group="router")]
    norm_topk_prob: Annotated[bool, Parameter(group="router")]
    n_group: int
    topk_group: int
    router_bias_update_speed: float = 0.001
    use_grouped_router: bool = False
    router_n_groups: int | None = None

    def build(
        self,
        n_routed_experts: int,
        num_experts_per_tok: int,
    ):
        cfg = self.model_dump()
        use_grouped_router = cfg.pop("use_grouped_router")
        router_n_groups = cfg.pop("router_n_groups")
        if use_grouped_router:
            print("Using NoAuxGroupedRouter")
            assert router_n_groups is not None, "router_n_groups must be specified for NoAuxGroupedRouter"
            return NoAuxGroupedRouter(
                **cfg,
                n_routed_experts=n_routed_experts,
                num_experts_per_tok=num_experts_per_tok,
                router_n_groups=router_n_groups,
            )
        else:
            return NoAuxRouter(
                **cfg,
                n_routed_experts=n_routed_experts,
                num_experts_per_tok=num_experts_per_tok,
            )


class NoAuxRouter(nn.Module, RouterProtocol):
    e_score_correction_bias: torch.FloatTensor

    def __init__(
        self,
        *,
        n_routed_experts: int,
        num_experts_per_tok: int,
        router_scaling_factor: float,
        scoring_func: Literal["sigmoid", "softmax"],
        n_group: int,
        topk_group: int,
        norm_topk_prob: bool = True,
        router_bias_update_speed: float = 0.001,
    ):
        super().__init__()
        self.top_k = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.router_scaling_factor = router_scaling_factor
        self.scoring_func = scoring_func
        self.n_group = n_group
        self.topk_group = topk_group

        self.norm_topk_prob = norm_topk_prob
        self.register_buffer(
            "e_score_correction_bias", torch.empty((self.n_routed_experts), device=get_device(), dtype=torch.float32)
        )

    def forward(self, logits, rollout_routed_experts: torch.Tensor | None = None) -> RouterResults:
        if self.scoring_func == "sigmoid":
            scores = logits.sigmoid()
        else:
            # TODO: (yehaochen)support softmax
            raise NotImplementedError(f"insupportable scoring function for MoE gating: {self.scoring_func}")

        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)

        if os.getenv("XTUNER_ROUTER_DEBUG") == "true":
            noise = torch.randn_like(scores) * 50
            scores_for_choice = scores + noise

        if self.n_group != self.topk_group:
            assert len(logits.shape) == 2, (
                f"XTuner Internal bug, invalid logits shape: {logits.shape}, expected shape with "
                "`(seq_len, hidden_states)`"
            )
            bsz = 1
            seq_len, _ = logits.shape
            group_scores = (
                scores_for_choice.view(bsz * seq_len, self.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
            )  # [n, n_group]
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]  # [n, top_k_group]
            group_mask = torch.zeros_like(group_scores)  # [n, n_group]
            group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(
                    bsz * seq_len,
                    self.n_group,
                    self.n_routed_experts // self.n_group,
                )
                .reshape(bsz * seq_len, -1)
            )  # [n, e]
            scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)  # [n, e]

        # select top-k experts
        # (only applicable when ep_size >= 64. when ep_size=32 (4 nodes), there is no need to employ this strategy)
        _, topk_idx = torch.topk(scores_for_choice, k=self.top_k, dim=-1)

        if rollout_routed_experts is not None:
            # seq_l, expert
            topk_ids = rollout_routed_experts
            # seq_l, expert
            topk_weight = scores.gather(dim=1, index=topk_ids)
        else:
            topk_weight = scores.gather(1, topk_idx)

        # The returned `router_weights` is only used for computing balance loss
        # It should be normalized
        scores_for_choice = scores_for_choice / torch.sum(scores_for_choice, dim=-1, keepdim=True)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        topk_weight = topk_weight * self.router_scaling_factor  # must multiply the scaling factor

        # TODO: (yehaochen) `Dispatcher` calculate the distribution duplicatedly
        tokens_per_expert = torch.histc(
            topk_idx,
            bins=self.n_routed_experts,
            min=0,
            max=self.n_routed_experts,
        )  # .view(self.ep_mesh.size(), -1)

        return {
            "logits": logits,
            "router_weights": scores_for_choice,
            "topk_weights": topk_weight,
            "topk_ids": topk_idx,
            "topkens_per_expert": tokens_per_expert,
        }


class NoAuxGroupedRouter(NoAuxRouter):
    """Only works for ep_size == topk."""

    def __init__(
        self,
        *,
        n_routed_experts: int,
        num_experts_per_tok: int,
        router_scaling_factor: float,
        router_n_groups: int,
        scoring_func: Literal["sigmoid", "softmax"],
        n_group: int,
        topk_group: int,
        norm_topk_prob: bool = True,
        router_bias_update_speed: float = 0.001,
    ):
        super().__init__(
            n_routed_experts=n_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            router_scaling_factor=router_scaling_factor,
            scoring_func=scoring_func,
            n_group=n_group,
            topk_group=topk_group,
            norm_topk_prob=norm_topk_prob,
            router_bias_update_speed=router_bias_update_speed,
        )
        self.router_n_groups = router_n_groups

    def forward(self, logits, rollout_routed_experts: torch.Tensor | None = None) -> RouterResults:
        seq, ne = logits.shape
        if self.scoring_func == "sigmoid":
            scores = logits.sigmoid()
        else:
            # TODO: (yehaochen)support softmax
            raise NotImplementedError(f"insupportable scoring function for MoE gating: {self.scoring_func}")

        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)

        if os.getenv("XTUNER_ROUTER_DEBUG") == "true":
            noise = torch.randn_like(scores) * 50
            scores_for_choice = scores + noise

        # group-based selection (group size = n_routed_experts // 8) ---
        n_groups = self.router_n_groups
        assert self.n_routed_experts % n_groups == 0, f"n_routed_experts must be divisible by {n_groups}"
        assert self.top_k == 8, "top_k must be 8 for NoAuxRouterOpt"
        group_size = max(1, self.n_routed_experts // n_groups)

        scores_for_choice = scores_for_choice.view(seq, n_groups, group_size)
        group_local_max_idx = torch.topk(scores_for_choice, k=self.top_k // n_groups, dim=2)[
            1
        ]  # [seq, n_groups, top_k_per_group]
        group_offsets = (torch.arange(n_groups, device=scores_for_choice.device) * group_size).view(
            1, -1, 1
        )  # [1, n_groups, 1]
        topk_idx = (group_local_max_idx + group_offsets).to(torch.long)  # [seq, n_groups, top_k_per_group]
        scores_for_choice = scores_for_choice.view(seq, self.n_routed_experts)
        if rollout_routed_experts is not None:
            # seq_l, expert
            topk_ids = rollout_routed_experts
            # seq_l, expert
            topk_weight = scores.gather(dim=1, index=topk_ids)
        else:
            topk_idx = topk_idx.view(seq, -1)  # [seq, top_k]
            topk_weight = scores.gather(1, topk_idx)  # [seq, n_groups]
        scores_for_choice = scores_for_choice.view(seq, self.n_routed_experts)

        # The returned `router_weights` is only used for computing balance loss
        # It should be normalized
        scores_for_choice = scores_for_choice / torch.sum(scores_for_choice, dim=-1, keepdim=True)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        topk_weight = topk_weight * self.router_scaling_factor  # must multiply the scaling factor

        tokens_per_expert = torch.histc(
            topk_idx,
            bins=self.n_routed_experts,
            min=0,
            max=self.n_routed_experts,
        )  # .view(self.ep_mesh.size(), -1)

        return {
            "logits": logits,
            "router_weights": scores_for_choice,
            "topk_weights": topk_weight,
            "topk_ids": topk_idx,
            "topkens_per_expert": tokens_per_expert,
        }
