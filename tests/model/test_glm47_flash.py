from unittest import mock

import torch

from xtuner.v1.model import Glm47FlashConfig, get_model_config
from xtuner.v1.module.attention import MLAConfig
from xtuner.v1.module.attention.mla import (
    mla_apply_rotary_pos_emb,
    mla_apply_rotary_pos_emb_non_interleaved,
)
from xtuner.v1.module.router.noaux_router import NoAuxRouterConfig


def _tiny_glm47_flash_config() -> Glm47FlashConfig:
    return Glm47FlashConfig(
        vocab_size=32,
        max_position_embeddings=64,
        pad_token_id=0,
        eos_token_id=1,
        hf_eos_token_id=[1, 2],
        num_hidden_layers=3,
        first_k_dense_replace=1,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=4,
        attention=MLAConfig(
            num_attention_heads=2,
            head_dim=4,
            kv_lora_rank=4,
            q_lora_rank=8,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=4,
            rope_interleave=True,
        ),
        hf_head_dim=4,
        qk_head_dim=8,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        router=NoAuxRouterConfig(
            n_group=1,
            topk_group=1,
            scoring_func="sigmoid",
            norm_topk_prob=True,
            router_scaling_factor=1.8,
        ),
        mlp_layer_types=["dense", "sparse", "sparse"],
        num_nextn_predict_layers=0,
        mtp_config=None,
        compile_cfg=False,
    )


class TestGlm47FlashConfig:
    def test_alias_and_default_architecture(self):
        config = get_model_config("glm-4.7-flash")

        assert isinstance(config, Glm47FlashConfig)
        assert config.model_type == "glm4_moe_lite"
        assert config.num_hidden_layers == 47
        assert config.first_k_dense_replace == 1
        assert config.attention.rope_interleave
        assert config.n_routed_experts == 64
        assert config.num_experts_per_tok == 4

    def test_hf_key_mapping_uses_packed_experts_and_mtp_layer(self):
        config = _tiny_glm47_flash_config()
        with mock.patch("torch.cuda.Stream"):
            model = config.build()

        assert model.to_hf_key_list("layers.1.experts.fused_w1w3.weight") == [
            "model.layers.1.mlp.experts.gate_up_proj"
        ]
        assert model.to_hf_key_list("layers.1.experts.fused_w2.weight") == [
            "model.layers.1.mlp.experts.down_proj"
        ]
        assert model.to_hf_key_list("layers.1.gate.router.e_score_correction_bias") == [
            "model.layers.1.mlp.gate.e_score_correction_bias"
        ]
        assert model.to_hf_key_list("mtp_block.layers.0.final_layernorm.weight") == [
            "model.layers.3.shared_head.norm.weight"
        ]

    def test_packed_expert_layout_round_trip(self):
        config = _tiny_glm47_flash_config()
        with mock.patch("torch.cuda.Stream"):
            model = config.build()

        gate_up_hf = torch.arange(4 * 8 * 16, dtype=torch.float32).reshape(4, 8, 16)
        gate_up_local = torch.empty(32, 16)
        model.safetensors_to_params(
            [gate_up_hf], gate_up_local, "layers.1.experts.fused_w1w3.weight", None, None, None
        )
        torch.testing.assert_close(gate_up_local, gate_up_hf.flatten(0, 1))
        torch.testing.assert_close(
            model.param_to_safetensor(gate_up_local, "model.layers.1.mlp.experts.gate_up_proj"), gate_up_hf
        )

        down_hf = torch.arange(4 * 16 * 4, dtype=torch.float32).reshape(4, 16, 4)
        down_local = torch.empty(64, 4)
        model.safetensors_to_params([down_hf], down_local, "layers.1.experts.fused_w2.weight", None, None, None)
        torch.testing.assert_close(down_local, down_hf.flatten(0, 1))
        torch.testing.assert_close(
            model.param_to_safetensor(down_local, "model.layers.1.mlp.experts.down_proj"), down_hf
        )


class TestGlm47FlashRope:
    def test_interleaved_and_half_split_layouts(self):
        q = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
        k = q + 1
        cos = torch.tensor([[[0.8, 0.6, 0.8, 0.6]]])
        sin = torch.tensor([[[0.6, 0.8, 0.6, 0.8]]])

        q_interleaved, k_interleaved = mla_apply_rotary_pos_emb(q, k, cos, sin)
        q_half, k_half = mla_apply_rotary_pos_emb_non_interleaved(q, k, cos, sin)

        expected_q_interleaved = torch.tensor([[[[-0.4, -1.4, 2.2, 4.8]]]])
        expected_q_half = torch.tensor([[[[-1.0, -2.0, 3.0, 4.0]]]])
        torch.testing.assert_close(q_interleaved, expected_q_interleaved)
        torch.testing.assert_close(q_half, expected_q_half)
        assert not torch.equal(k_interleaved, k_half)
