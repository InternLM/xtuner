# Copyright (c) OpenMMLab. All rights reserved.
"""MuonConfig 参数路由回归测试。

TestMuonParamRouting
    test_2d_matrix_routes_to_muon_regular   2D 矩阵进 Muon regular 组
    test_conv1d_weight_routes_to_adamw      3D depthwise conv1d 权重进 AdamW 组
    test_1d_params_route_to_adamw           1D 参数（norm/bias）进 AdamW 组
    test_fused_expert_weights_keep_num_experts  2D fused expert 权重仍按 num_experts 分组
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch.nn as nn

from xtuner.v1.config.optim import MuonConfig


class _MuonRoutingProbe(nn.Module):
    """Minimal stand-in exposing the interfaces MuonConfig.build() relies on."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(n_routed_experts=2)
        self.proj = nn.Linear(8, 8, bias=False)
        self.self_attn_conv1d = nn.Conv1d(8, 8, kernel_size=4, groups=8, bias=False)
        self.norm = nn.LayerNorm(8)
        self.fused_w1w3 = nn.Linear(8, 2 * 2 * 4, bias=False)
        self.fused_w2 = nn.Linear(4, 2 * 8, bias=False)

    def trainable_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return [(name, p) for name, p in self.named_parameters()]


class TestMuonParamRouting:
    def _param_to_group(self, model: nn.Module) -> dict[int, dict[str, Any]]:
        with patch("xtuner.v1.config.optim.Muon") as muon_cls, patch("xtuner.v1.config.optim.dist"):
            MuonConfig().build(model)
        assert muon_cls.call_args is not None, "MuonConfig.build() must construct the Muon optimizer"
        param_groups = muon_cls.call_args.args[0]
        return {id(p): group for group in param_groups for p in group["params"]}

    def test_2d_matrix_routes_to_muon_regular(self) -> None:
        """A plain 2D weight goes to the Muon regular group without num_experts."""
        model = _MuonRoutingProbe()
        group = self._param_to_group(model)[id(model.proj.weight)]
        assert "algorithm" not in group
        assert "num_experts" not in group

    def test_conv1d_weight_routes_to_adamw(self) -> None:
        """Depthwise conv1d weight [C, 1, K] is 3D and must go to the AdamW group."""
        model = _MuonRoutingProbe()
        group = self._param_to_group(model)[id(model.self_attn_conv1d.weight)]
        assert group.get("algorithm") == "adamw"

    def test_1d_params_route_to_adamw(self) -> None:
        """1D tensors (norm weight/bias) go to the AdamW group."""
        model = _MuonRoutingProbe()
        groups = self._param_to_group(model)
        assert groups[id(model.norm.weight)].get("algorithm") == "adamw"
        assert groups[id(model.norm.bias)].get("algorithm") == "adamw"

    def test_fused_expert_weights_keep_num_experts(self) -> None:
        """2D fused expert weights stay in Muon MoE groups with their num_experts."""
        model = _MuonRoutingProbe()
        groups = self._param_to_group(model)
        assert groups[id(model.fused_w1w3.weight)]["num_experts"] == 4
        assert groups[id(model.fused_w2.weight)]["num_experts"] == 2
