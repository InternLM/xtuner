# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Annotated, Literal

import torch
from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict
from torch import nn
from torch.nn import functional as F

from .protocol import RouterProtocol, RouterResults


class GreedyRouterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scoring_func: Annotated[Literal["sigmoid", "softmax"], Parameter(group="router")]
    router_scaling_factor: Annotated[float, Parameter(group="router")]
    norm_topk_prob: Annotated[bool, Parameter(group="router")]
    use_grouped_router: bool = False
    router_n_groups: int | None = None

    def build(
        self,
        n_routed_experts: int,
        num_experts_per_tok: int,
    ) -> "GreedyRouter":
        cfg = self.model_dump()
        use_grouped_router = cfg.pop("use_grouped_router")
        router_n_groups = cfg.pop("router_n_groups")
        if use_grouped_router:
            print("Using GreedyGroupedRouter")
            assert router_n_groups is not None, "router_n_groups must be specified for GreedyGroupedRouter"
            return GreedyGroupedRouter(
                **cfg,
                n_routed_experts=n_routed_experts,
                num_experts_per_tok=num_experts_per_tok,
                router_n_groups=router_n_groups,
            )
        else:
            return GreedyRouter(
                **cfg,
                n_routed_experts=n_routed_experts,
                num_experts_per_tok=num_experts_per_tok,
            )


class GreedyRouter(nn.Module, RouterProtocol):
    def __init__(
        self,
        *,
        n_routed_experts: int,
        num_experts_per_tok: int,
        norm_topk_prob: bool = True,
        scoring_func: Literal["sigmoid", "softmax"] = "softmax",
        router_scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.n_routed_experts = n_routed_experts
        self.top_k = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.router_scaling_factor = router_scaling_factor

    def forward(self, logits: torch.Tensor, rollout_routed_experts: torch.Tensor | None = None) -> RouterResults:
        if os.getenv("XTUNER_ROUTER_DEBUG") == "true":
            noise = torch.randn_like(logits) * 50
            logits = logits + noise

        # TODO: (yehaochen) Support sigmoid
        if self.scoring_func == "sigmoid":
            routing_weights = logits.sigmoid()
        else:
            routing_weights = F.softmax(logits, dim=1, dtype=torch.float)
        if rollout_routed_experts is not None:
            # seq_l, expert
            topk_ids = rollout_routed_experts
            # seq_l, expert
            topk_weights = routing_weights.gather(dim=1, index=topk_ids)
        else:
            topk_weights, topk_ids = torch.topk(routing_weights, self.top_k, dim=-1)

        if self.norm_topk_prob:
            topk_weights /= topk_weights.sum(dim=-1, keepdim=True)

        if self.router_scaling_factor != 1.0:
            topk_weights = topk_weights * self.router_scaling_factor

        # moe forward
        # (e, )
        tokens_per_expert = torch.histc(topk_ids, bins=self.n_routed_experts, min=0, max=self.n_routed_experts)

        return {
            "logits": logits,
            "router_weights": routing_weights,
            "topk_weights": topk_weights,
            "topk_ids": topk_ids,
            "topkens_per_expert": tokens_per_expert,
        }


class GreedyGroupedRouter(GreedyRouter):
    def __init__(
        self,
        *,
        n_routed_experts: int,
        num_experts_per_tok: int,
        router_n_groups: int,
        norm_topk_prob: bool = True,
        scoring_func: Literal["sigmoid", "softmax"] = "softmax",
        router_scaling_factor: float = 1.0,
    ):
        super().__init__(
            n_routed_experts=n_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            norm_topk_prob=norm_topk_prob,
            scoring_func=scoring_func,
            router_scaling_factor=router_scaling_factor,
        )
        self.router_n_groups = router_n_groups

    def forward(self, logits: torch.Tensor, rollout_routed_experts: torch.Tensor | None = None) -> RouterResults:
        if os.getenv("XTUNER_ROUTER_DEBUG") == "true":
            noise = torch.randn_like(logits) * 50
            logits = logits + noise

        # TODO: (yehaochen) Support sigmoid
        if self.scoring_func == "sigmoid":
            routing_weights = logits.sigmoid()
        else:
            routing_weights = F.softmax(logits, dim=1, dtype=torch.float)

        if rollout_routed_experts is not None:
            # seq_l, expert
            topk_ids = rollout_routed_experts
            # seq_l, expert
            topk_weights = routing_weights.gather(dim=1, index=topk_ids)
        else:
            # group-based selection
            n_groups = self.router_n_groups
            assert self.n_routed_experts % n_groups == 0, f"n_routed_experts must be divisible by {n_groups}"
            assert self.top_k == 8, "top_k must be 8 for NoAuxRouterOpt"
            group_size = max(1, self.n_routed_experts // n_groups)

            seq, ne = logits.shape
            scores_for_choice = routing_weights.view(seq, n_groups, group_size)
            group_local_max_idx = torch.topk(scores_for_choice, k=self.top_k // n_groups, dim=2)[
                1
            ]  # [seq, n_groups, top_k_per_group]
            group_offsets = (torch.arange(n_groups, device=scores_for_choice.device) * group_size).view(
                1, -1, 1
            )  # [1, n_groups, 1]
            topk_ids = (group_local_max_idx + group_offsets).to(torch.long)  # [seq, n_groups, top_k_per_group]
            scores_for_choice = scores_for_choice.view(seq, self.n_routed_experts)
            topk_ids = topk_ids.view(seq, -1)  # [seq, top_k]
            topk_weights = scores_for_choice.gather(1, topk_ids)  # [seq, n_groups]

        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights = topk_weights / denominator

        if self.router_scaling_factor != 1.0:
            topk_weights = topk_weights * self.router_scaling_factor

        # moe forward
        # (e, )
        tokens_per_expert = torch.histc(topk_ids, bins=self.n_routed_experts, min=0, max=self.n_routed_experts)

        return {
            "logits": logits,
            "router_weights": routing_weights,
            "topk_weights": topk_weights,
            "topk_ids": topk_ids,
            "topkens_per_expert": tokens_per_expert,
        }
