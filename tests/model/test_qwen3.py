import os

import parametrize
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_distributed import DistributedTestBase
from transformers import AutoModelForCausalLM

from xtuner.v1.model.moe.moe import MoEConfig
from xtuner.v1.model.moe.qwen3 import Qwen3MoE
from xtuner.v1.module.attention import MHAConfig
from xtuner.v1.module.router import GreedyRouterConfig

# Qwen3 30B A3
QWEN3_MOE_PATH = os.environ["QWEN3_MOE_PATH"]


class TestQwen3MoE(DistributedTestBase):
    @parametrize.parametrize("device", [("cuda",)])
    def test_qwen3_moe(self, device):
        self.create_pg("cuda")
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
        config = MoEConfig(
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
            training_dtype="bf16",
            chunked_loss=False,
            n_routed_experts=128,
            n_shared_experts=0,
            num_experts_per_tok=8,
            first_k_dense_replace=0,
            hidden_factor=1.0,
            moe_intermediate_size=768,
            dispatcher="naive",
            router=router_config,
        )
        with torch.device("meta"):
            qwen_model = Qwen3MoE(config).to(torch.bfloat16)

        qwen_model.from_hf(QWEN3_MOE_PATH)

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "8"))
