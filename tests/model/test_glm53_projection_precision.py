# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

from xtuner.v1.float8.config import Float8Config, ScalingGranularity
from xtuner.v1.model.moe.glm53.glm53 import Glm53TextMoEConfig
from xtuner.v1.module.attention.kda import KDAConfig
from xtuner.v1.module.decoder_layer.mhc import MHCConfig, hc_pre
from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEGate
from xtuner.v1.module.mtp import MTPConfig
from xtuner.v1.module.router import NoAuxRouterConfig


class TestProjectionPrecision:
    @pytest.mark.parametrize("sp_impl", ["ulysses", "fla"])
    def test_glm53_builder_wires_projection_blocks_without_changing_sp(self, sp_impl):
        cfg = Glm53TextMoEConfig(
            compile_cfg=False,
            ep_size=1,
            dispatcher=None,
            num_hidden_layers=1,
            first_k_dense_replace=0,
            glm53_layer_types=["linear_attention"],
            linear_attention=KDAConfig(num_heads=2, head_dim=16, sp_impl=sp_impl, gate_projection_block_size=64),
            mhc=MHCConfig(hc_mult=4, projection_block_size=64),
            router_projection_block_size=64,
            mtp_config=MTPConfig(num_layers=7, share_weights=True),
        )
        # Only the CUDA offload stream is mocked; construct the actual model on meta.
        with patch("torch.cuda.Stream"), torch.device("meta"):
            model = cfg.build()
        layer = model.layers["0"]
        assert layer.self_attn.sp_impl == sp_impl
        assert layer.self_attn.gate_projection_block_size == 64
        assert layer.mhc_cfg.projection_block_size == 64
        assert layer.gate.router_projection_block_size == 64
        assert len(model.mtp_block.layers) == 1
        assert model.mtp_block.layers[0].decoder_layer.gate.router_projection_block_size == 64

    @pytest.mark.gpu
    @pytest.mark.parametrize("full_rank", [False, True])
    def test_kda_excludes_only_sensitive_gates_from_fp8(self, full_rank):
        cfg = KDAConfig(num_heads=4, head_dim=16, use_full_rank_gate=full_rank, gate_projection_block_size=64)
        module = (
            cfg.build(hidden_size=64, float8_cfg=Float8Config(scaling_granularity_gemm=ScalingGranularity.TILEWISE))
            .cuda()
            .to(torch.bfloat16)
        )
        assert "Float8" in type(module.q_proj).__name__
        assert "Float8" not in type(module.f_a_proj).__name__
        assert "Float8" not in type(module.f_b_proj).__name__
        assert "Float8" in type(module.b_proj).__name__
        assert "Float8" in type(module.o_proj).__name__
        x = torch.randn(1, 131, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        y = module._gate_output(x)
        if full_rank:
            expected = F.linear(x.double(), module.g_proj.weight.double())
        else:
            expected = F.linear(F.linear(x.double(), module.g_a_proj.weight.double()), module.g_b_proj.weight.double())
        assert y.dtype == torch.float32
        torch.testing.assert_close(y.double(), expected, atol=2e-5, rtol=2e-5)
        y.square().mean().backward()
        for name, parameter in module.named_parameters():
            if name.startswith("g_"):
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()

    def test_mhc_blocked_projection_matches_native_math(self):
        torch.manual_seed(82)
        x = torch.randn(2, 131, 4, 17, requires_grad=True)
        fn = torch.randn(24, 68, requires_grad=True)
        scale, base = torch.rand(3), torch.rand(24)
        native = hc_pre(x, fn, scale, base, 4, 20, 1e-6)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            blocked = hc_pre(x, fn, scale, base, 4, 20, 1e-6, projection_block_size=64)
        for a, b in zip(native, blocked):
            torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
        g1 = torch.autograd.grad(native[0].square().sum(), (x, fn), retain_graph=True)
        g2 = torch.autograd.grad(blocked[0].square().sum(), (x, fn))
        for a, b in zip(g1, g2):
            torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)

    @pytest.mark.gpu
    def test_router_preserves_rollout_routing(self):
        router = NoAuxRouterConfig(
            n_group=1, topk_group=1, scoring_func="sigmoid", norm_topk_prob=True, router_scaling_factor=2.5
        )
        gate = MoEGate(
            hidden_size=17,
            n_routed_experts=8,
            num_experts_per_tok=2,
            router_config=router,
            router_projection_block_size=64,
        )
        gate = gate.cuda()
        torch.nn.init.normal_(gate.weight)
        hidden = torch.randn(1, 131, 17, device="cuda")
        experts = torch.tensor([[2, 5]], device="cuda").expand(131, -1)
        output = gate(hidden, experts)
        assert torch.equal(output["topk_ids"], experts)
        expected = F.linear(hidden.reshape(-1, 17).double(), gate.weight.double())
        torch.testing.assert_close(output["logits"].double(), expected, atol=2e-5, rtol=2e-5)
