# Copyright (c) OpenMMLab. All rights reserved.
from typing import Literal, TypeAlias

import torch
from pydantic import BaseModel
from torch import nn
from torch.nn import functional as F

from xtuner.v1.config import BaseRouterConfig, MoEConfig

from .protocol import RouterProtocol, RouterResults


GreeedyRouterConfig: TypeAlias = BaseRouterConfig


class GreedyRouterConfig(BaseModel):
    n_routed_experts: int = 256
    num_experts_per_tok: int = 8
    scoring_func: Literal["sigmoid", "softmax"] = "softmax"
    norm_topk_prob: bool = True


class GreedyRouter(nn.Module, RouterProtocol):
    def __init__(self, config: MoEConfig):
        super().__init__()
        assert isinstance(config.router, GreedyRouterConfig), "GreedyRouter requires GreedyRouterConfig"
        self.config = config.router
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        # topk selection algorithm
        self.norm_topk_prob = self.config.norm_topk_prob

    def forward(self, logits: torch.Tensor) -> RouterResults:
        routing_weights = F.softmax(logits, dim=1, dtype=torch.float)
        topk_weights, topk_ids = torch.topk(routing_weights, self.top_k, dim=-1)

        if self.norm_topk_prob:
            topk_weights /= topk_weights.sum(dim=-1, keepdim=True)

        # moe forward
        # (e, )
        tokens_per_expert = torch.histc(topk_ids, bins=self.n_routed_experts, min=0, max=self.n_routed_experts)

        return {
            "logits": logits,
            "topk_weights": topk_weights,
            "topk_ids": topk_ids,
            "topkens_per_expert": tokens_per_expert,
        }
