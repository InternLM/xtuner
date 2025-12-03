import math

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor

from ..grouped_linear.moe_group_linear import GroupedLinear, build_grouped_linear


class LoraGroupedLinear(nn.Module):
    def __init__(
        self,
        base_layer: GroupedLinear,
        rank: int,
        alpha: int,
        lora_dropout: float = 0.0,
        init_lora_weights: bool = True,
    ):
        super().__init__()

        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.num_routed_experts = base_layer.num_routed_experts
        self.ep_mesh = base_layer.ep_mesh
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.merged = False

        self.lora_A = build_grouped_linear(self.in_features, self.rank, self.num_routed_experts, ep_mesh=self.ep_mesh)
        self.lora_B = build_grouped_linear(self.rank, self.out_features, self.num_routed_experts, ep_mesh=self.ep_mesh)

        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        for p in self.base_layer.parameters():
            p.requires_grad = False

        if init_lora_weights:
            self.reset_parameters()

    def reset_parameters(self):
        if isinstance(self.lora_A.weight, DTensor):
            # TODO: init DTensor
            raise NotImplementedError
        else:
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor, tokens_per_expert: torch.Tensor, decoding: bool = False) -> torch.Tensor:
        if self.merged:
            return self.base_layer(x, tokens_per_expert, decoding)
        original_out = self.base_layer(x, tokens_per_expert, decoding)
        # lora_out = self.lora_a_naive_forward(x, tokens_per_expert, decoding)
        lora_out = self.lora_A(x, tokens_per_expert, decoding)
        # lora_out = self.lora_b_naive_forward(lora_out, tokens_per_expert, decoding)
        lora_out = self.lora_B(lora_out, tokens_per_expert, decoding)
        return original_out + lora_out * self.scale

    def lora_a_naive_forward(
        self, x: torch.Tensor, tokens_per_expert: torch.Tensor, decoding: bool = False
    ) -> torch.Tensor:
        weight = self.lora_A.weight.view(-1, self.rank, self.in_features)
        batch_sizes = tokens_per_expert.cpu().numpy()

        out = []
        start = 0
        for i, size in enumerate(batch_sizes):
            rhs = weight[i, :, :].t()
            out.append(x[start : start + size, :] @ rhs)
            start += size
        return torch.cat(out)

    def lora_b_naive_forward(
        self, x: torch.Tensor, tokens_per_expert: torch.Tensor, decoding: bool = False
    ) -> torch.Tensor:
        weight = self.lora_B.weight.view(-1, self.out_features, self.rank)
        batch_sizes = tokens_per_expert.cpu().numpy()

        out = []
        start = 0
        for i, size in enumerate(batch_sizes):
            rhs = weight[i, :, :].t()
            out.append(x[start : start + size, :] @ rhs)
            start += size
        return torch.cat(out)

    @torch.no_grad()
    def merge_lora(self):
        raise NotImplementedError

    @torch.no_grad()
    def unmerge_lora(self):
        raise NotImplementedError

    def __repr__(self) -> str:
        return "lora." + super().__repr__()
