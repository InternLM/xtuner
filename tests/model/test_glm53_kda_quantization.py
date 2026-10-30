"""CPU/meta regression tests for the GLM-5.3 mixed-FP8 projection policy."""

from pathlib import Path

import pytest
import torch

from xtuner.v1.float8 import TileWiseFloat8Linear
from xtuner.v1.float8.config import Float8Config
from xtuner.v1.model.moe.glm53.glm53 import Glm53TextMoEConfig, HFGlm5NextTextConfig
from xtuner.v1.module.attention.kda import KDAConfig
from xtuner.v1.module.linear import build_linear


class TestKDAProjectionQuantization:
    @pytest.mark.parametrize("quantize", [True, False])
    @pytest.mark.parametrize("full_rank_gate", [True, False])
    def test_build_respects_projection_policy(self, quantize: bool, full_rank_gate: bool) -> None:
        cfg = KDAConfig(num_heads=1, head_dim=128, quantize_projections=quantize, use_full_rank_gate=full_rank_gate)
        float8_cfg = Float8Config(scaling_granularity_gemm="tilewise")
        with torch.device("meta"):
            module = cfg.build(hidden_size=128, float8_cfg=float8_cfg)
            unrelated_linear = build_linear(128, 128, bias=False, float8_cfg=float8_cfg)
        names = ["q_proj", "k_proj", "v_proj", "o_proj", "b_proj", "f_a_proj", "f_b_proj"]
        names += ["g_proj"] if full_rank_gate else ["g_a_proj", "g_b_proj"]
        for name in names:
            assert isinstance(getattr(module, name), TileWiseFloat8Linear) is quantize, name
        # Disabling KDA quantization must not mutate the shared FP8 policy.
        assert isinstance(unrelated_linear, TileWiseFloat8Linear)

    def test_generic_default_preserves_fp8(self) -> None:
        cfg = KDAConfig(num_heads=1, head_dim=128)
        assert cfg.quantize_projections
        with torch.device("meta"):
            module = cfg.build(hidden_size=128, float8_cfg=Float8Config(scaling_granularity_gemm="tilewise"))
        assert isinstance(module.q_proj, TileWiseFloat8Linear)

    @pytest.mark.parametrize("quantize", [True, False])
    def test_without_fp8_config_uses_regular_linear(self, quantize: bool) -> None:
        with torch.device("meta"):
            module = KDAConfig(num_heads=1, head_dim=128, quantize_projections=quantize).build(hidden_size=128)
        assert not isinstance(module.q_proj, TileWiseFloat8Linear)

    def test_glm53_default_excludes_kda_from_fp8(self) -> None:
        cfg = Glm53TextMoEConfig()
        assert cfg.linear_attention is not None
        assert not cfg.linear_attention.quantize_projections
        restored = KDAConfig.model_validate(cfg.linear_attention.model_dump())
        assert not restored.quantize_projections
        with torch.device("meta"):
            module = cfg.linear_attention.build(
                hidden_size=cfg.hidden_size, float8_cfg=Float8Config(scaling_granularity_gemm="tilewise")
            )
        assert not isinstance(module.q_proj, TileWiseFloat8Linear)
        assert not isinstance(module.o_proj, TileWiseFloat8Linear)

    @pytest.mark.skipif(HFGlm5NextTextConfig is None, reason="requires Transformers with glm5_next support")
    def test_hf_conversion_excludes_kda_from_fp8(self, tmp_path: Path) -> None:
        assert HFGlm5NextTextConfig is not None
        # Published checkpoints retain these fields in addition to HF's native defaults.
        HFGlm5NextTextConfig(
            first_k_dense_replace=3,
            eos_token_id=[154820, 154827, 154829],
            linear_attn_config={
                "num_heads": 64,
                "head_dim": 128,
                "short_conv_kernel_size": 4,
                "gate_lower_bound": -5.0,
            },
            num_nextn_predict_layers=1,
            scoring_func="sigmoid",
        ).save_pretrained(tmp_path)
        cfg = Glm53TextMoEConfig.from_hf(tmp_path)
        assert cfg.linear_attention is not None
        assert not cfg.linear_attention.quantize_projections
