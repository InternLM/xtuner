import math

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor

from xtuner.v1.utils import init_params

from ..linear.linear import build_linear


class LoraLinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: int,
        lora_dropout: float = 0.0,
        init_lora_weights: bool = True,
    ):
        super().__init__()

        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.merged = False
        self.init_lora_weights = init_lora_weights

        weight = base_layer.weight
        dtype = weight.dtype
        device = weight.device

        # A: (in_features -> r)
        self.lora_A = build_linear(self.in_features, rank, bias=False, device=device, dtype=dtype)
        # B: (r -> out_features)
        self.lora_B = build_linear(rank, self.out_features, bias=False, device=device, dtype=dtype)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.base_layer(x)

        original_out = self.base_layer(x)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scale
        return original_out + lora_out

    @torch.no_grad()
    def merge_lora(self):
        """把 LoRA 权重合并进 base_layer.weight."""
        if self.merged:
            return

        base_weight = self.base_layer.weight
        lora_a_weight = self.lora_A.weight
        lora_b_weight = self.lora_B.weight

        base_local = base_weight.to_local() if isinstance(base_weight, DTensor) else base_weight
        a_local = lora_a_weight.to_local() if isinstance(lora_a_weight, DTensor) else lora_a_weight
        b_local = lora_b_weight.to_local() if isinstance(lora_b_weight, DTensor) else lora_b_weight

        if a_local.shape != (self.rank, self.in_features) or b_local.shape != (self.out_features, self.rank):
            raise RuntimeError("LoRA weights must be unsharded before merge_lora() is called")
        if base_local.ndim != 2 or base_local.shape[0] < self.out_features or base_local.shape[1] != self.in_features:
            raise RuntimeError("Base weight has an incompatible shape for merge_lora()")

        delta_w = torch.matmul(b_local, a_local)
        base_local[: self.out_features].add_(delta_w, alpha=self.scale)

        self.merged = True

    @torch.no_grad()
    def unmerge_lora(self):
        """从 base_layer.weight 中还原 LoRA（如果之前 merge 过）"""
        if not self.merged:
            return

        base_weight = self.base_layer.weight
        lora_a_weight = self.lora_A.weight
        lora_b_weight = self.lora_B.weight

        base_local = base_weight.to_local() if isinstance(base_weight, DTensor) else base_weight
        a_local = lora_a_weight.to_local() if isinstance(lora_a_weight, DTensor) else lora_a_weight
        b_local = lora_b_weight.to_local() if isinstance(lora_b_weight, DTensor) else lora_b_weight

        if a_local.shape != (self.rank, self.in_features) or b_local.shape != (self.out_features, self.rank):
            raise RuntimeError("LoRA weights must be unsharded before unmerge_lora() is called")
        if base_local.ndim != 2 or base_local.shape[0] < self.out_features or base_local.shape[1] != self.in_features:
            raise RuntimeError("Base weight has an incompatible shape for unmerge_lora()")

        delta_w = torch.matmul(b_local, a_local)
        base_local[: self.out_features].sub_(delta_w, alpha=self.scale)

        self.merged = False

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"r={self.rank}, "
            f"lora_alpha={self.alpha}, "
            f"scale={self.scale}, "
            f"merged={self.merged}"
        )

    def __repr__(self) -> str:
        return "lora." + super().__repr__()
