import pytest

from xtuner.v1.float8.config import Float8Config, ScalingGranularity
from xtuner.v1.float8.float8_gmm_tile_wise import ADAPTIVEGEMM_INSTALLED, TileWiseFloat8GroupedLinear
from xtuner.v1.module.grouped_linear.moe_group_linear import GroupedLinear, build_grouped_linear


def test_build_grouped_linear_ignores_dense_gemm_float8_switch():
    layer = build_grouped_linear(
        in_features=128,
        out_features=128,
        num_routed_experts=2,
        float8_cfg=Float8Config(
            scaling_granularity_gemm=ScalingGranularity.TILEWISE,
            scaling_granularity_grouped_gemm=None,
        ),
    )

    assert isinstance(layer, GroupedLinear)
    assert not isinstance(layer, TileWiseFloat8GroupedLinear)


@pytest.mark.skipif(not ADAPTIVEGEMM_INSTALLED, reason="TileWiseFloat8GroupedLinear requires adaptive_gemm")
def test_build_grouped_linear_uses_grouped_gemm_float8_switch():
    layer = build_grouped_linear(
        in_features=128,
        out_features=128,
        num_routed_experts=2,
        float8_cfg=Float8Config(
            scaling_granularity_gemm=None,
            scaling_granularity_grouped_gemm=ScalingGranularity.TILEWISE,
        ),
    )

    assert isinstance(layer, TileWiseFloat8GroupedLinear)
