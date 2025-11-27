import math

import torch
import torch.nn as nn

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

        if init_lora_weights:
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.base_layer(x)

        original_out = self.base_layer(x)
        lora_intermediate = self.lora_A(x)
        lora_intermediate = self.lora_dropout(lora_intermediate)
        lora_out = self.lora_B(lora_intermediate) * self.scale
        return original_out + lora_out

    @torch.no_grad()
    def merge_lora(self):
        """把 LoRA 权重合并进 base_layer.weight."""
        if self.merged:
            return

        # delta_W = B @ A   shape: [out, in]
        delta_w = torch.matmul(self.lora_B.weight, self.lora_A.weight)
        self.base_layer.weight += delta_w * self.scale

        self.merged = True
        # 合并后 LoRA 参数可以不再训练
        for p in self.lora_A.parameters():
            p.requires_grad = False
        for p in self.lora_B.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def unmerge_lora(self):
        """从 base_layer.weight 中还原 LoRA（如果之前 merge 过）"""
        if not self.merged:
            return

        delta_w = torch.matmul(self.lora_B.weight, self.lora_A.weight)
        self.base_layer.weight -= delta_w * self.scale

        self.merged = False
        for p in self.lora_A.parameters():
            p.requires_grad = True
        for p in self.lora_B.parameters():
            p.requires_grad = True

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
