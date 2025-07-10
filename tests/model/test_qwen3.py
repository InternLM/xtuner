import os

import parametrize
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_distributed import DistributedTestBase
from transformers import AutoModelForCausalLM, AutoTokenizer

from xtuner.v1.model.moe.moe import MoEConfig, SequenceContext
from xtuner.v1.model.moe.qwen3 import Qwen3MoE
from xtuner.v1.module.attention import MHAConfig
from xtuner.v1.module.router import GreedyRouterConfig

# Qwen3 30B A3
QWEN3_MOE_PATH = os.environ["QWEN3_MOE_PATH"]


class TestQwen3MoE(DistributedTestBase):
    @parametrize.parametrize(
        "device,dispatcher,ep_size",
        [
            ("cuda", "deepep", 8),
            ("cuda", "all2all", 8),
            ("cuda", None, 1),
        ],
    )
    def test_qwen3_moe_run(self, device, dispatcher, ep_size):
        self.create_pg(device)

        hf_model = AutoModelForCausalLM.from_pretrained(
            QWEN3_MOE_PATH,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="cuda"
        )
        tokenizer = AutoTokenizer.from_pretrained(QWEN3_MOE_PATH, trust_remote_code=True)
        input_ids = tokenizer("吃葡萄不吐葡萄皮", return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            output = hf_model(
                input_ids=input_ids,
                labels=input_ids.clone(),
            )
        expected_loss = output.loss

        del hf_model
        torch.cuda.empty_cache()

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
            dispatcher=dispatcher,
            router=router_config,
        )

        device_mesh = init_device_mesh(
            device_type=device,
            mesh_shape=(self.world_size // ep_size, ep_size),
            mesh_dim_names=("dp", "ep"),
        )
        ep_mesh = device_mesh["ep"]
        with torch.device("meta"):
            qwen_model = Qwen3MoE(config, ep_mesh=ep_mesh).to(torch.bfloat16)

        seq_ctx = SequenceContext.from_input_ids(input_ids=(input_ids, ))
        seq_ctx, shifted_labels = seq_ctx.shift_with_labels(labels=input_ids)

        qwen_model.from_hf(QWEN3_MOE_PATH)

        with torch.no_grad():
            output = qwen_model(
                seq_ctx=seq_ctx,
                labels=shifted_labels,
            )
        loss = output["loss"]
        self.assertTrue(torch.allclose(loss, expected_loss.to(loss.dtype), atol=1e-2, rtol=1e-2))


    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "8"))
