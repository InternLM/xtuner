# Copyright (c) OpenMMLab. All rights reserved.
"""MuonConfig 参数路由行为测试（真实 build，纯 CPU）。

TestMuonParamRouting                  MuonConfig.build 的分组路由行为
    test_2d_matrix_routes_to_muon_regular       2D 矩阵进 Muon regular 组
    test_conv1d_weight_routes_to_adamw          3D depthwise conv1d 权重进 AdamW 组
    test_1d_params_route_to_adamw               1D 参数（norm/bias）进 AdamW 组
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
    """路由规则：按 ndim == 2 与名字模式把参数分进 Muon / AdamW 组。"""

    def _build_param_groups(self, model: nn.Module) -> dict[int, dict[str, Any]]:
        # MuonConfig.build 的路由契约是它组装的 param_groups；真实 Muon 构造要求 DTensor 参数和
        # 分布式运行时（__init__ 会按 mesh 分组并建子通信组），对纯路由单测属于过重协作者，故在此
        # 拦截；dist 是项目外依赖，一并 patch 掉以便脱离分布式环境运行
        with patch("xtuner.v1.config.optim.Muon") as muon_cls, patch("xtuner.v1.config.optim.dist"):
            MuonConfig().build(model)
        assert muon_cls.call_args is not None, "MuonConfig.build() must construct the Muon optimizer"
        param_groups = muon_cls.call_args.args[0]
        return {id(p): group for group in param_groups for p in group["params"]}

    def test_2d_matrix_routes_to_muon_regular(self) -> None:
        # 普通 2D 权重应进 Muon regular 组，不带 num_experts
        model = _MuonRoutingProbe()
        group = self._build_param_groups(model)[id(model.proj.weight)]
        assert "algorithm" not in group
        assert "num_experts" not in group

    def test_conv1d_weight_routes_to_adamw(self) -> None:
        # 3D depthwise conv1d 权重 [C, 1, K] 应进 AdamW 组，不走 Muon
        model = _MuonRoutingProbe()
        group = self._build_param_groups(model)[id(model.self_attn_conv1d.weight)]
        assert group.get("algorithm") == "adamw"

    def test_1d_params_route_to_adamw(self) -> None:
        # 1D 参数（norm 的 weight/bias）应进 AdamW 组
        model = _MuonRoutingProbe()
        groups = self._build_param_groups(model)
        assert groups[id(model.norm.weight)].get("algorithm") == "adamw"
        assert groups[id(model.norm.bias)].get("algorithm") == "adamw"

    def test_fused_expert_weights_keep_num_experts(self) -> None:
        # 2D fused expert 权重应留在 Muon MoE 组并保留 num_experts（fused_w1w3 翻倍）
        model = _MuonRoutingProbe()
        groups = self._build_param_groups(model)
        assert groups[id(model.fused_w1w3.weight)]["num_experts"] == 4
        assert groups[id(model.fused_w2.weight)]["num_experts"] == 2
