import math

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor

from xtuner.v1.float8.float8_gmm_tile_wise import TileWiseFloat8GroupedLinear
from xtuner.v1.utils import init_params

from ..grouped_linear.moe_group_linear import GroupedLinear, build_grouped_linear


class LoraGroupedLinear(nn.Module):
    def __init__(
        self,
        base_layer: GroupedLinear | TileWiseFloat8GroupedLinear,
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
        self.init_lora_weights = init_lora_weights

        self.lora_A = build_grouped_linear(self.in_features, self.rank, self.num_routed_experts, ep_mesh=self.ep_mesh)
        self.lora_B = build_grouped_linear(self.rank, self.out_features, self.num_routed_experts, ep_mesh=self.ep_mesh)

        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        for p in self.base_layer.parameters():
            p.requires_grad = False

        self.reset_parameters()

    def reset_parameters(self):
        if self.lora_A.weight.is_meta:
            return

        init_params(self.lora_A.weight, lambda weight: nn.init.kaiming_uniform_(weight, a=math.sqrt(5)))
        if self.init_lora_weights:
            init_params(self.lora_B.weight, nn.init.zeros_)
        else:
            init_params(self.lora_B.weight, lambda weight: nn.init.kaiming_uniform_(weight, a=math.sqrt(5)))

    def forward(self, x: torch.Tensor, tokens_per_expert: torch.Tensor, decoding: bool = False) -> torch.Tensor:
        if self.merged:
            return self.base_layer(x, tokens_per_expert, decoding)
        original_out = self.base_layer(x, tokens_per_expert, decoding)
        lora_out = self.lora_A(self.lora_dropout(x), tokens_per_expert, decoding)
        lora_out = self.lora_B(lora_out, tokens_per_expert, decoding)
        return original_out + lora_out * self.scale

    def lora_a_naive_forward(
        self, x: torch.Tensor, tokens_per_expert: torch.Tensor, decoding: bool = False
    ) -> torch.Tensor:
        weight = self.lora_A.weight.to_local() if isinstance(self.lora_A.weight, DTensor) else self.lora_A.weight
        weight = weight.view(-1, self.rank, self.in_features)
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
        weight = self.lora_B.weight.to_local() if isinstance(self.lora_B.weight, DTensor) else self.lora_B.weight
        weight = weight.view(-1, self.out_features, self.rank)
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
        """把 LoRA 权重合并进 base_layer.weight。

        对每个 expert e，计算 delta_W_e = B_e @ A_e，然后加到 base weight 上：
            base_weight_e += delta_W_e * scale
        其中 base_weight_e、A_e、B_e 分别是 base_layer、lora_A、lora_B 的第 e 个 expert 的权重。
        """
        if self.merged:
            return

        base_weight = self.base_layer.weight
        lora_a_weight = self.lora_A.weight
        lora_b_weight = self.lora_B.weight

        if isinstance(base_weight, DTensor):
            # DTensor 场景：在 local tensor 上操作，每个 rank 持有部分 expert
            base_local = base_weight.to_local()
            a_local = lora_a_weight.to_local()
            b_local = lora_b_weight.to_local()
        else:
            base_local = base_weight
            a_local = lora_a_weight
            b_local = lora_b_weight

        local_experts = a_local.shape[0] // self.rank
        expected_base_rows = local_experts * self.out_features
        if (
            a_local.shape != (local_experts * self.rank, self.in_features)
            or b_local.shape != (local_experts * self.out_features, self.rank)
            or base_local.ndim != 2
            or base_local.shape[0] < expected_base_rows
            or base_local.shape[1] != self.in_features
        ):
            raise RuntimeError("Grouped LoRA weights must contain complete local experts before merge_lora()")

        base_view = base_local[:expected_base_rows].view(local_experts, self.out_features, self.in_features)
        a_view = a_local.view(local_experts, self.rank, self.in_features)
        b_view = b_local.view(local_experts, self.out_features, self.rank)
        base_view.add_(torch.bmm(b_view, a_view), alpha=self.scale)

        self.merged = True

    @torch.no_grad()
    def unmerge_lora(self):
        """从 base_layer.weight 中还原 LoRA（如果之前 merge 过）。

        对每个 expert e，从 base weight 中减去 delta_W_e * scale：
            base_weight_e -= delta_W_e * scale
        """
        if not self.merged:
            return

        base_weight = self.base_layer.weight
        lora_a_weight = self.lora_A.weight
        lora_b_weight = self.lora_B.weight

        if isinstance(base_weight, DTensor):
            base_local = base_weight.to_local()
            a_local = lora_a_weight.to_local()
            b_local = lora_b_weight.to_local()
        else:
            base_local = base_weight
            a_local = lora_a_weight
            b_local = lora_b_weight

        local_experts = a_local.shape[0] // self.rank
        expected_base_rows = local_experts * self.out_features
        if (
            a_local.shape != (local_experts * self.rank, self.in_features)
            or b_local.shape != (local_experts * self.out_features, self.rank)
            or base_local.ndim != 2
            or base_local.shape[0] < expected_base_rows
            or base_local.shape[1] != self.in_features
        ):
            raise RuntimeError("Grouped LoRA weights must contain complete local experts before unmerge_lora()")

        base_view = base_local[:expected_base_rows].view(local_experts, self.out_features, self.in_features)
        a_view = a_local.view(local_experts, self.rank, self.in_features)
        b_view = b_local.view(local_experts, self.out_features, self.rank)
        base_view.sub_(torch.bmm(b_view, a_view), alpha=self.scale)

        self.merged = False

    def __repr__(self) -> str:
        return "lora." + super().__repr__()
