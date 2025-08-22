from unittest import TestCase

from xtuner.v1.module.attention import MHAConfig
from xtuner.v1.module.router import GreedyRouterConfig
from xtuner.v1.model.moe.qwen3 import Qwen3MoE


class TestBuildModel(TestCase):
    def test_build_moe(self):
        from xtuner.v1.model import Qwen3MoEConfig

        router_config = GreedyRouterConfig(
            scoring_func="sigmoid",
            norm_topk_prob=True,
            router_scaling_factor=1.0,
        )
        attention_config = MHAConfig(
            num_attention_heads=32,
            num_key_value_heads=4,
            head_dim=128,
            qk_norm=True
        )
        config = Qwen3MoEConfig(
            vocab_size=151936,
            max_position_embeddings=4096,
            padding_idx=0,
            num_hidden_layers=48,
            hidden_size=2048,
            intermediate_size=6144,
            rms_norm_eps=1e-6,
            rope_theta=1000000.0,
            hidden_act="silu",
            attention=attention_config,
            tie_word_embeddings=False,
            n_routed_experts=128,
            n_shared_experts=0,
            num_experts_per_tok=8,
            first_k_dense_replace=0,
            hidden_factor=1.0,
            moe_intermediate_size=768,
            router=router_config,
        )
        model = config.build()
        self.assertIsInstance(model, Qwen3MoE)
