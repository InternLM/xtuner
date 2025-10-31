# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Annotated, Literal

import torch
from cyclopts import Parameter
from pydantic import BaseModel
from torch import nn
from torch.nn import functional as F

from .protocol import RouterProtocol, RouterResults


class GreedyRouterConfig(BaseModel):
    scoring_func: Annotated[Literal["sigmoid", "softmax"], Parameter(group="router")]
    router_scaling_factor: Annotated[float, Parameter(group="router")]
    norm_topk_prob: Annotated[bool, Parameter(group="router")]

    def build(
        self,
        n_routed_experts: int,
        num_experts_per_tok: int,
    ) -> "GreedyRouter":
        return GreedyRouter(
            **self.model_dump(),
            n_routed_experts=n_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
        )


class GreedyRouter(nn.Module, RouterProtocol):
    def __init__(
        self,
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

    def forward(self, logits: torch.Tensor) -> RouterResults:
        if os.getenv("XTUNER_ROUTER_DEBUG") == "true":
            noise = torch.randn_like(logits) * 50
            logits = logits + noise

        # TODO: (yehaochen) Support sigmoid
        if self.scoring_func == "sigmoid":
            routing_weights = logits.sigmoid()
        else:
            routing_weights = F.softmax(logits, dim=1, dtype=torch.float)
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
            "topk_weights": topk_weights,
            "topk_ids": topk_ids,
            "topkens_per_expert": tokens_per_expert,
        }
