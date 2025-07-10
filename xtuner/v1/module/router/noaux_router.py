from typing import Literal

import torch
import torch.nn as nn

from xtuner.v1.config import BaseRouterConfig, MoEConfig

from .protocol import RouterProtocol, RouterResults


class NoAuxRouterConfig(BaseRouterConfig):
    routed_scaling_factor: float
    scoring_func: Literal["sigmoid", "softmax"]
    n_group: int
    topk_group: int
    norm_topk_prob: bool
    router_bias_update_speed: float = 0.001


class NoAuxRouter(nn.Module, RouterProtocol):
    e_score_correction_bias: torch.FloatTensor

    def __init__(self, config: MoEConfig):
        super().__init__()
        assert isinstance(config.router, NoAuxRouterConfig), "NoAuxRouter requires NoAuxRouterConfig"
        self.config = config.router
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = self.config.routed_scaling_factor
        self.scoring_func = self.config.scoring_func
        self.n_group = self.config.n_group
        self.topk_group = self.config.topk_group

        self.norm_topk_prob = self.config.norm_topk_prob
        self.register_buffer("e_score_correction_bias", torch.empty((self.n_routed_experts), dtype=torch.float32))

    def forward(self, logits) -> RouterResults:
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
        topk_weight = topk_weight * self.routed_scaling_factor  # must multiply the scaling factor

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
