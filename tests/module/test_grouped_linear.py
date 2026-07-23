"""GroupedLinear 的 FP8 配置选择行为测试。

TestGroupedLinearFactory
    test_grouped_gemm_switch_selects_implementation: grouped-GEMM 开关独立决定是否使用 tilewise FP8。
"""

import pytest

from xtuner.v1.float8.config import Float8Config, ScalingGranularity
from xtuner.v1.float8.float8_gmm_tile_wise import ADAPTIVEGEMM_INSTALLED, TileWiseFloat8GroupedLinear
from xtuner.v1.module.grouped_linear.moe_group_linear import GroupedLinear, build_grouped_linear


class TestGroupedLinearFactory:
    @pytest.mark.parametrize(
        ("gemm_granularity", "grouped_gemm_granularity", "expected_type"),
        [
            (ScalingGranularity.TILEWISE, None, GroupedLinear),
            pytest.param(
                None,
                ScalingGranularity.TILEWISE,
                TileWiseFloat8GroupedLinear,
                marks=pytest.mark.skipif(not ADAPTIVEGEMM_INSTALLED, reason="requires adaptive_gemm"),
            ),
        ],
    )
    def test_grouped_gemm_switch_selects_implementation(
        self,
        gemm_granularity,
        grouped_gemm_granularity,
        expected_type,
    ):
        # 验证 dense GEMM 开关不会误启用 grouped FP8，而 grouped-GEMM 开关会启用它。
        layer = build_grouped_linear(
            in_features=128,
            out_features=128,
            num_routed_experts=2,
            float8_cfg=Float8Config(
                scaling_granularity_gemm=gemm_granularity,
                scaling_granularity_grouped_gemm=grouped_gemm_granularity,
            ),
        )

        assert type(layer) is expected_type
