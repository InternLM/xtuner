import os
from typing import Literal

import torch
import torch.nn as nn

from xtuner.v1.config import BaseRouterConfig

from .protocol import RouterProtocol, RouterResults


class NoAuxRouterConfig(BaseRouterConfig):
    n_group: int
    topk_group: int
    router_bias_update_speed: float = 0.001

    def build(
        self,
        n_routed_experts: int,
        num_experts_per_tok: int,
    ):
        return NoAuxRouter(
            **self.model_dump(),
            n_routed_experts=n_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
        )


class NoAuxRouter(nn.Module, RouterProtocol):
    e_score_correction_bias: torch.FloatTensor

    def __init__(
        self,
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
        self.register_buffer("e_score_correction_bias", torch.empty((self.n_routed_experts), dtype=torch.float32))

    def forward(self, logits) -> RouterResults:
        if os.getenv("XTUNER_ROUTER_DEBUG") == "true":
            noise = torch.randn_like(logits) * 50
            logits = logits + noise

        if self.scoring_func == "sigmoid":
            scores = logits.sigmoid()
        else:
            # TODO: (yehaochen)support softmax
            raise NotImplementedError(f"insupportable scoring function for MoE gating: {self.scoring_func}")

        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)

        # select top-k experts
        # (only applicable when ep_size >= 64. when ep_size=32 (4 nodes), there is no need to employ this strategy)
        _, topk_idx = torch.topk(scores_for_choice, k=self.top_k, dim=-1)
        topk_weight = scores.gather(1, topk_idx)

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
            "topk_weights": topk_weight,
            "topk_ids": topk_idx,
            "topkens_per_expert": tokens_per_expert,
        }
