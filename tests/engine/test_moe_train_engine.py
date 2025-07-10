import os
import tempfile
import shutil
import time
import copy

import parametrize
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_distributed import DistributedTestBase
from transformers import AutoModelForCausalLM, AutoTokenizer

from xtuner.v1.model.moe.moe import MoEConfig, SequenceContext
from xtuner.v1.model.moe.qwen3 import Qwen3MoE
from xtuner.v1.module.attention import MHAConfig
from xtuner.v1.module.router import GreedyRouterConfig
from xtuner.v1.config import FSDPConfig, LRConfig, MoEConfig, MoELossConfig, OptimConfig, AdamWConfig
from xtuner.v1.engine.moe_train_engine import MoETrainEngine


# Qwen3 30B A3
QWEN3_MOE_PATH = os.environ["QWEN3_MOE_PATH"]


class TestMoEEngine(DistributedTestBase):
    @parametrize.parametrize(
        "device,ep_size,hsdp_sharding_size",
        [
            ("cuda", 1, 8),  # todo: test ep8 and hsdp, OOM in 8 gpus
        ],
    )
    def test_moe_engine_train(self, device, ep_size, hsdp_sharding_size):
        self.create_pg(device)

        router_config = GreedyRouterConfig(
            scoring_func="softmax",
            norm_topk_prob=True,
            router_scaling_factor=1.0,
        )
        attention_config = MHAConfig(num_attention_heads=32, num_key_value_heads=4, head_dim=128, qk_norm=True)
        moe_cfg = MoEConfig(
            model_path=QWEN3_MOE_PATH,
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
            router=router_config,
            model_type="qwen",
            dispatcher="all2all",
        )

        moe_loss_cfg = MoELossConfig(
            balancing_loss_type="softmax",
            balancing_loss_alpha=0.001,
            balancing_loss_global_average=True,
            z_loss_alpha=0.001,
            z_loss_global_average=True,
        )
        optim_cfg: AdamWConfig = AdamWConfig()
        lr_cfg: LRConfig = LRConfig(total_steps=1000)
        fsdp_cfg: FSDPConfig = FSDPConfig(
            torch_compile=False,
            cpu_offload=False,
            ep_size=ep_size,
            max_length=8192,
            hsdp_sharding_size=hsdp_sharding_size,
        )
        engine = MoETrainEngine(
            model_cfg=moe_cfg, moe_loss_cfg=moe_loss_cfg, optim_cfg=optim_cfg, lr_cfg=lr_cfg, fsdp_cfg=fsdp_cfg
        )

        tok = AutoTokenizer.from_pretrained(
            QWEN3_MOE_PATH
        )
        txt = "根据国际地球自转和参考系服务机构的数据，今年夏天是自2020年以来第六次地球自转加速。7月9日将成为有史以来最短的一天，比平时短1.3到1.6毫秒。 "
        input_ids = tok.encode(txt, return_tensors="pt").view(1, -1)
        labels = input_ids.clone()
        input_ids = input_ids[:, :-1]
        labels = labels[:, 1:]
        data_batch = {
            "input_ids": input_ids,
            "labels": labels,
            "num_tokens": torch.tensor([input_ids.shape[1]], device=input_ids.device, dtype=torch.int32),
        }
        losses = []
        for _ in range(10):
            log = engine.train_step([data_batch], intra_layer_micro_batch=1)
            losses.append(log["reduced_llm_loss"])
        losses_ref = [2.44, 2.44, 2.42, 2.41, 2.34, 2.33, 2.16, 2.13, 1.71, 1.55]
        for loss, loss_ref in zip(losses, losses_ref):
            self.assertTrue(abs(loss - loss_ref) < 0.05)

    @parametrize.parametrize(
        "device,ep_size,hsdp_sharding_size",
        [
            ("cuda", 1, 8),  # todo: test ep8 and hsdp, OOM in 8 gpus
        ],
    )
    def test_save_and_load(self, device, ep_size, hsdp_sharding_size):
        self.create_pg(device)

        temp_dir = tempfile.mkdtemp()
        if dist.get_rank() == 0:
            temp_dir = [temp_dir]
        else:
            temp_dir = [None]
        dist.broadcast_object_list(temp_dir, src=0)
        temp_dir = temp_dir[0]
        router_config = GreedyRouterConfig(
            scoring_func="softmax",
            norm_topk_prob=True,
            router_scaling_factor=1.0,
        )
        attention_config = MHAConfig(num_attention_heads=32, num_key_value_heads=4, head_dim=128, qk_norm=True)
        moe_cfg = MoEConfig(
            vocab_size=151936,
            max_position_embeddings=4096,
            padding_idx=0,
            num_hidden_layers=4,
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
            router=router_config,
            model_type="qwen",
            dispatcher="all2all",
        )

        moe_loss_cfg = MoELossConfig(
            balancing_loss_type="softmax",
            balancing_loss_alpha=0.001,
            balancing_loss_global_average=True,
            z_loss_alpha=0.001,
            z_loss_global_average=True,
        )
        optim_cfg: AdamWConfig = AdamWConfig()
        lr_cfg: LRConfig = LRConfig(total_steps=1000)
        fsdp_cfg: FSDPConfig = FSDPConfig(
            torch_compile=True,
            cpu_offload=False,
            ep_size=ep_size,
            max_length=8192,
            hsdp_sharding_size=hsdp_sharding_size,
        )
        engine = MoETrainEngine(
            model_cfg=moe_cfg,
            moe_loss_cfg=moe_loss_cfg,
            optim_cfg=optim_cfg,
            lr_cfg=lr_cfg,
            fsdp_cfg=fsdp_cfg,
        )

        engine.init_model()
        engine.save_hf(
            hf_dir=temp_dir,
            save_dtype=torch.bfloat16,
        )

        dist.barrier()
        time.sleep(1)

        moe_cfg2 = copy.deepcopy(moe_cfg)
        moe_cfg2.model_path = temp_dir
        engine2 = MoETrainEngine(
            model_cfg=moe_cfg2,
            moe_loss_cfg=moe_loss_cfg,
            optim_cfg=optim_cfg,
            lr_cfg=lr_cfg,
            fsdp_cfg=fsdp_cfg,
        )
        state_dict = engine.model.state_dict()
        state_dict2 = engine2.model.state_dict()
        for key, val in state_dict.items():
            val2 = state_dict2[key]
            val = val.full_tensor().bfloat16()
            val2 = val2.full_tensor().bfloat16()
            self.assertTrue(torch.equal(val, val2[:val.shape[0]]), f"Mismatch in {key} between bf16 and fp8, {val} and {val2[:val.shape[0]]}")

        if dist.get_rank() == 0:
            shutil.rmtree(temp_dir)

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "8"))
