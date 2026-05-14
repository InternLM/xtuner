import os
import unittest

import torch
import torch.distributed as dist

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.module.attention import MHAConfig
from xtuner.v1.module.dispatcher.base import NaiveDispatcher
from xtuner.v1.module.router.greedy import GreedyRouterConfig
from xtuner.v1.model.moe.qwen3 import Qwen3MoEConfig


def _tiny_moe_cfg() -> Qwen3MoEConfig:
    return Qwen3MoEConfig(
        vocab_size=32,
        max_position_embeddings=32,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        num_hidden_layers=1,
        hidden_size=16,
        intermediate_size=32,
        rms_norm_eps=1e-6,
        rope_theta=1e6,
        hidden_act="silu",
        attention=MHAConfig(num_attention_heads=2, num_key_value_heads=1, head_dim=8, qk_norm=True),
        tie_word_embeddings=False,
        n_routed_experts=4,
        n_shared_experts=0,
        num_experts_per_tok=2,
        first_k_dense_replace=0,
        hidden_factor=1.0,
        moe_intermediate_size=8,
        router=GreedyRouterConfig(scoring_func="softmax", norm_topk_prob=True, router_scaling_factor=1.0),
        ep_size=1,
        expert_tp_size=2,
        dispatcher=None,
        compile_cfg=False,
        balancing_loss_cfg=None,
        z_loss_cfg=None,
    )


@unittest.skipIf(not torch.cuda.is_available(), "CUDA/NCCL is required for real ExpertTP mesh validation.")
class TestMoEExpertTPWithoutEP(DeterministicDDPTestCase):
    def test_builds_real_ep_ownership_mesh_for_expert_tp_without_ep(self) -> None:
        pg = self.create_pg("cuda")
        rank = dist.get_rank()
        torch.cuda.set_device(rank % torch.cuda.device_count())

        model = _tiny_moe_cfg().build()
        layer = model.layers["0"]

        # 中文注释：不开 EP 但开启 expert TP 时，EP ownership 维度仍然真实存在，只是 size=1。
        assert model.ep_mesh is not None
        assert model.tp_mesh is not None
        assert model.ep_mesh.size() == 1
        assert model.tp_mesh.size() == 2
        assert layer.experts.fused_w1w3.ep_size == 1
        assert layer.experts.fused_w1w3.tp_size == 2
        assert isinstance(layer.dispatcher, NaiveDispatcher)

        dist.barrier()
        dist.destroy_process_group(pg)

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "2"))
