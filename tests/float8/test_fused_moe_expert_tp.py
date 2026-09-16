# Copyright (c) OpenMMLab. All rights reserved.

from unittest import mock

import pytest
import torch

from xtuner.v1.float8.config import Float8Config, ScalingGranularity
from xtuner.v1.float8.float8_gmm_tile_wise import TileWiseFloat8GroupedLinear
from xtuner.v1.module.grouped_linear.moe_group_linear import build_grouped_linear


class TestFusedMoEExpertTP:
    @pytest.mark.parametrize("enable_fused_moe_activation", [False, True])
    def test_builder_preserves_expert_tp_and_fusion_options(self, enable_fused_moe_activation: bool) -> None:
        ep_mesh, expert_tp_mesh, ep_tp_mesh = mock.Mock(), mock.Mock(), mock.Mock()
        float8_cfg = Float8Config(
            scaling_granularity_grouped_gemm=ScalingGranularity.TILEWISE,
            enable_fused_moe_activation=enable_fused_moe_activation,
        )
        with mock.patch(
            "xtuner.v1.module.grouped_linear.moe_group_linear.TileWiseFloat8GroupedLinear"
        ) as factory:
            layer = build_grouped_linear(
                128,
                256,
                2,
                ep_mesh=ep_mesh,
                expert_tp_mesh=expert_tp_mesh,
                parallel_style="column",
                float8_cfg=float8_cfg,
                ep_tp_mesh=ep_tp_mesh,
                num_fused_projections=2,
            )

        assert layer is factory.return_value
        factory.assert_called_once_with(
            128,
            256,
            2,
            moe_bias=False,
            ep_mesh=ep_mesh,
            expert_tp_mesh=expert_tp_mesh,
            parallel_style="column",
            ep_tp_mesh=ep_tp_mesh,
            num_fused_projections=2,
            enable_fused_moe_activation=enable_fused_moe_activation,
        )

    @pytest.mark.parametrize("fused_swiglu", [False, True])
    def test_forward_uses_local_output_features(self, fused_swiglu: bool) -> None:
        # Exercise the TP-local output reshape without allocating weights or running CUDA kernels.
        layer = TileWiseFloat8GroupedLinear.__new__(TileWiseFloat8GroupedLinear)
        torch.nn.Module.__init__(layer)
        layer.out_features = 256
        layer.local_out_features = 128
        layer.enable_fused_moe_activation = True
        input_tensor = torch.empty(2, 3, 256 if fused_swiglu else 128)
        tokens_per_expert = torch.tensor([3, 3])
        output = torch.empty(6, 128)
        weight_fp8 = mock.sentinel.weight_fp8
        kernel_name = (
            "fp8_gmm_weight_per_block_act_per_tile_fused_swiglu"
            if fused_swiglu
            else "fp8_gmm_weight_per_block_act_per_tile"
        )

        with mock.patch.object(layer, "_prepare_weight_fp8", return_value=weight_fp8), mock.patch(
            f"xtuner.v1.float8.float8_gmm_tile_wise.{kernel_name}.apply", return_value=output
        ) as gemm:
            if fused_swiglu:
                actual = layer.forward_fused_moe_act(input_tensor, tokens_per_expert)
            else:
                actual = layer(input_tensor, tokens_per_expert)

        assert actual.shape == (2, 3, 128)
        assert gemm.call_args.args[0].shape == (6, input_tensor.shape[-1])
        assert gemm.call_args.args[1] is weight_fp8
        assert gemm.call_args.args[2] is tokens_per_expert
        if not fused_swiglu:
            assert gemm.call_args.args[3] is True
