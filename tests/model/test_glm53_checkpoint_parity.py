"""GLM53 checkpoint loading and text-logit parity with Hugging Face."""

from pathlib import Path

import pytest
import torch

from transformers import Glm5NextConfig, Glm5NextForConditionalGeneration, Glm5NextTextConfig, Glm5NextVisionConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model.compose.glm53 import Glm53BaseConfig


class TestGlm53CheckpointParity:
    @pytest.mark.gpu
    def test_checkpoint_loaded_text_logits_match_hf(self, tmp_path: Path) -> None:
        """Load the same checkpoint through both public paths without overriding normalization."""
        torch.manual_seed(1)
        # Small weights expose normalization errors in the final logits without changing eps.
        # Keep 64 heads for the production config validator and use eager attention below.
        text_config = Glm5NextTextConfig(
            vocab_size=64,
            initializer_range=0.005,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=64,
            num_key_value_heads=64,
            q_lora_rank=16,
            kv_lora_rank=24,
            qk_nope_head_dim=8,
            v_head_dim=8,
            layer_types=["full_attention", "full_attention"],
            mlp_layer_types=["dense", "sparse"],
            first_k_dense_replace=1,
            index_topk=16,  # Cover this short sequence to avoid near-tied pool selection.
            index_head_dim=8,
            index_n_heads=2,
            index_kpool=2,
            n_routed_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=16,
            scoring_func="sigmoid",
            num_nextn_predict_layers=0,
            linear_attn_config={"num_heads": 2, "head_dim": 16, "short_conv_kernel_size": 4, "gate_lower_bound": -5.0},
            pad_token_id=0,
            eos_token_id=1,
        )
        vision_config = Glm5NextVisionConfig(
            depth=1,
            hidden_size=32,
            num_heads=4,
            intermediate_size=64,
            out_hidden_size=32,
            projection_intermediate_size=64,
        )
        config = Glm5NextConfig(text_config=text_config, vision_config=vision_config)
        Glm5NextForConditionalGeneration(config).save_pretrained(tmp_path)

        hf_model = Glm5NextForConditionalGeneration.from_pretrained(
            tmp_path, dtype=torch.float16, device_map="cuda", attn_implementation="eager"
        ).eval()
        cfg = Glm53BaseConfig.from_hf(tmp_path)
        cfg.compile_cfg = False
        cfg.text_config.compile_cfg = False
        cfg.vision_config.attn_impl = "eager_attention"
        cfg.text_config.attention.sparse_mla_backend = "torch"
        cfg.text_config.attention.indexer_backend = "torch"
        model = cfg.build().to(device="cuda", dtype=torch.float16).eval()
        model.from_hf(tmp_path, strict=True)

        tokens = torch.tensor([[2, 7, 11, 5, 19, 23, 3, 17, 29, 13, 1]], device="cuda")
        seq_ctx = SequenceContext.from_input_ids((tokens,), device="cuda")
        with torch.no_grad():
            expected = hf_model(input_ids=tokens, use_cache=False).logits
            actual = model(seq_ctx=seq_ctx, loss_ctx=None).logits
        torch.testing.assert_close(actual.float(), expected.float(), atol=1e-4, rtol=1e-3)
