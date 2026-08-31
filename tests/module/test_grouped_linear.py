"""GroupedLinear 的 FP8 配置选择行为测试。

TestGroupedLinearFactory
    test_grouped_gemm_switch_selects_implementation: grouped-GEMM 开关独立决定是否使用 tilewise FP8。
"""

import pytest
import torch

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_grouped_linear_returns_natural_gradient_for_call_local_weight() -> None:
    layer = GroupedLinear(in_features=128, out_features=128, num_routed_experts=2).cuda().bfloat16()
    original_parameter = layer.weight
    override = torch.randn(2, 128, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    hidden_states = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    counts = torch.tensor([2, 2], device="cuda", dtype=torch.int32)
    override_ref = override.detach().clone().requires_grad_()
    hidden_states_ref = hidden_states.detach().clone().requires_grad_()

    output = layer(hidden_states, counts, trainable_weight=override)
    expected = torch.cat(
        (
            hidden_states_ref[:2] @ override_ref[0].T,
            hidden_states_ref[2:] @ override_ref[1].T,
        )
    )
    grad_output = torch.randn_like(output)
    output.backward(grad_output)
    expected.backward(grad_output)

    assert layer.weight is original_parameter
    assert original_parameter.grad is None
    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(hidden_states.grad, hidden_states_ref.grad, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(override.grad, override_ref.grad, rtol=2e-2, atol=2e-2)
