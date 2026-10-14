"""GLM text normalization must use the checkpoint's HF rounding boundaries."""

from unittest.mock import patch

import pytest
import torch

from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextRMSNorm
from xtuner.v1.model.moe.glm53.glm53 import Glm53TextMoEConfig, _glm53_rms_norm
from xtuner.v1.module.attention.kda import KDAConfig
from xtuner.v1.module.mtp import MTPConfig
from xtuner.v1.module.rms_norm import RMSNorm
from xtuner.v1.ops.rms_norm import _triton_rms_norm, native_rms_norm, npu_rms_norm
from xtuner.v1.ops.rms_norm.protocol import RMSNormProtocol


class TestGlm53RMSNorm:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
    @pytest.mark.parametrize("shape", [(37, 64), (1, 37, 64), (2, 7, 64)])
    def test_hf_forward_and_backward(self, dtype: torch.dtype, device: str, shape: tuple[int, ...]) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA required")
        torch.manual_seed(41)
        xt = RMSNorm(64, eps=1e-5).to(device=device, dtype=dtype)
        hf = Glm5NextTextRMSNorm(64, eps=1e-5).to(device=device, dtype=dtype)
        with torch.no_grad():
            xt.weight.uniform_(0.5, 1.5)
            hf.weight.copy_(xt.weight)
        x = torch.randn(shape, device=device, dtype=dtype)
        if dtype == torch.bfloat16:
            assert not torch.equal(xt(x), hf(x)), "Fixture must expose the original rounding error"
        xt.rms_norm_fn = _glm53_rms_norm
        a, b = (x.clone().requires_grad_() for _ in range(2))
        ya, yb = xt(a), hf(b)
        torch.testing.assert_close(ya, yb, atol=0, rtol=0)
        cotangent = torch.randn_like(ya)
        ya.backward(cotangent)
        yb.backward(cotangent)
        torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)
        torch.testing.assert_close(xt.weight.grad, hf.weight.grad, atol=0, rtol=0)

    @pytest.mark.parametrize("backend", [native_rms_norm, _triton_rms_norm, npu_rms_norm])
    def test_glm_builder_preserves_backend_selection(self, backend: RMSNormProtocol) -> None:
        cfg = Glm53TextMoEConfig(
            compile_cfg=False,
            ep_size=1,
            dispatcher=None,
            num_hidden_layers=2,
            first_k_dense_replace=1,
            glm53_layer_types=["linear_attention", "deepseek_sparse_attention"],
            linear_attention=KDAConfig(num_heads=2, head_dim=16),
            mtp_config=MTPConfig(num_layers=1, share_weights=True),
        )
        other = RMSNorm(32)
        original_fn = other.rms_norm_fn
        with (
            patch("xtuner.v1.module.rms_norm.rms_norm.rms_norm", backend),
            patch("torch.cuda.Stream"),
            torch.device("meta"),
        ):
            model = cfg.build()
        selected = {name for name, module in model.named_modules() if isinstance(module, RMSNorm)}
        assert {
            "norm",
            "layers.0.input_layernorm",
            "layers.0.post_attention_layernorm",
            "layers.1.input_layernorm",
            "layers.1.post_attention_layernorm",
            "layers.1.self_attn.q_a_layernorm",
            "layers.1.self_attn.kv_a_layernorm",
        } <= selected
        assert any(name.startswith("mtp_block.") for name in selected)
        expected_fn = _glm53_rms_norm if backend is native_rms_norm else backend
        for name in selected:
            assert model.get_submodule(name).rms_norm_fn is expected_fn, name
        assert other.rms_norm_fn is original_fn
