"""GLM-5.2 配置、HF 转换、路由、checkpoint 与并行数值行为测试。

TestGlm52Config
    test_from_hf_preserves_glm_specific_behavior: HF 配置转换保留 DSA、router 与 MTP 语义。
    test_rejects_shared_physical_mtp_indexer: 非法的 physical MTP indexer 计划会被拒绝。
TestGlm52CheckpointConversion
    test_tiny_model_round_trips_through_hf: tiny 主干与 MTP 参数可经公共 HF API 无损往返。
TestGlm52RouterBias
    test_scratch_init_zeroes_main_and_mtp_biases: 从头初始化清零主干与 MTP router bias。
    test_update_bias_handles_main_and_shared_mtp_loads: bias 更新覆盖主干并聚合共享 MTP 深度。
TestGlm52SequenceParallel
    test_mtp_loss_and_gradients_match_full_sequence: SP2 的 MTP loss 与梯度匹配完整序列。
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pytest
import torch
import torch.distributed as dist

from transformers.models.glm_moe_dsa import GlmMoeDsaConfig as HFGlmMoeDsaConfig
from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model import Glm52MoEConfig, get_model_config, get_model_config_from_hf
from xtuner.v1.module.attention import DSAMLAConfig
from xtuner.v1.module.mtp import MTPConfig
from xtuner.v1.module.router.noaux_router import NoAuxRouterConfig
from xtuner.v1.utils.test_utils import init_data_mesh


GLM5_2_MOE_PATH = Path(os.environ["GLM5_2_MOE_PATH"])


def _tiny_glm52_config() -> Glm52MoEConfig:
    return Glm52MoEConfig(
        vocab_size=32,
        max_position_embeddings=64,
        pad_token_id=0,
        eos_token_id=1,
        hf_eos_token_id=[1, 2],
        num_hidden_layers=3,
        first_k_dense_replace=1,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        attention=DSAMLAConfig(
            num_attention_heads=2,
            head_dim=4,
            kv_lora_rank=4,
            q_lora_rank=8,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=4,
            index_topk=4,
            index_head_dim=4,
            index_n_heads=2,
            indexer_types=["full", "shared", "shared", "full"],
            sparse_mla_backend="torch",
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
            router_scaling_factor=2.5,
        ),
        mlp_layer_types=["dense", "sparse", "sparse"],
        num_nextn_predict_layers=1,
        compile_cfg=False,
    )


class TestGlm52Config:
    def test_from_hf_preserves_glm_specific_behavior(self):
        # 验证公共 HF 配置转换保留 GLM-5.2 的 DSA、router、MTP 及回写语义。
        hf_config = HFGlmMoeDsaConfig.from_pretrained(GLM5_2_MOE_PATH)

        config = get_model_config_from_hf(GLM5_2_MOE_PATH)

        assert isinstance(config, Glm52MoEConfig)
        assert isinstance(get_model_config("glm-5.2"), Glm52MoEConfig)
        assert (config.model_type, config.num_hidden_layers, config.hidden_size) == (
            hf_config.model_type,
            hf_config.num_hidden_layers,
            hf_config.hidden_size,
        )
        assert (
            config.attention.index_topk,
            config.attention.index_head_dim,
            config.attention.index_n_heads,
        ) == (
            hf_config.index_topk,
            hf_config.index_head_dim,
            hf_config.index_n_heads,
        )
        assert config.attention.indexer_types == [*hf_config.indexer_types, "full"]
        assert isinstance(config.router, NoAuxRouterConfig)
        assert (
            config.router.scoring_func,
            config.router.n_group,
            config.router.topk_group,
            config.router.router_scaling_factor,
        ) == (
            hf_config.scoring_func,
            hf_config.n_group,
            hf_config.topk_group,
            hf_config.routed_scaling_factor,
        )
        assert config.mtp_config == MTPConfig(
            num_layers=hf_config.num_nextn_predict_layers,
            share_weights=True,
        )
        assert config.hf_config.indexer_types == hf_config.indexer_types

    def test_rejects_shared_physical_mtp_indexer(self):
        # 验证 physical MTP 首层配置为 shared 时通过公共 build API 明确报错。
        config = _tiny_glm52_config()
        config.attention.indexer_types = ["full", "full", "full", "shared"]
        config.mtp_config = MTPConfig(num_layers=1, share_weights=True)

        with pytest.raises(ValueError, match="physical MTP indexer_types"):
            config.build()


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestGlm52CheckpointConversion(DeterministicDDPTestCase):
    def test_tiny_model_round_trips_through_hf(self):
        # 验证 tiny 主干、MoE 与 MTP 参数通过 save_hf/from_hf 公共 API 后逐参数保持一致。
        self.create_pg("cuda")
        config = _tiny_glm52_config()
        config.mtp_config = MTPConfig(num_layers=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            model = config.build().to(device="cuda", dtype=torch.bfloat16)
            model.init_weights()
            model.save_hf(tmpdir)

            restored = config.build().to(device="cuda", dtype=torch.bfloat16)
            restored.from_hf(tmpdir)

            expected_parameters = dict(model.named_parameters())
            actual_parameters = dict(restored.named_parameters())
            assert actual_parameters.keys() == expected_parameters.keys()
            for name, expected in expected_parameters.items():
                torch.testing.assert_close(actual_parameters[name], expected, rtol=0, atol=0)

    @property
    def world_size(self) -> int:
        return 1


class TestGlm52RouterBias:
    def test_scratch_init_zeroes_main_and_mtp_biases(self):
        # 验证 init_weights 会覆盖脏值并清零主干与 MTP 的 NoAux router bias。
        config = _tiny_glm52_config()
        config.mtp_config = MTPConfig(num_layers=1)
        with mock.patch("torch.cuda.Stream"):
            model = config.build()
        routers = [
            model.layers["1"].gate.router,
            model.layers["2"].gate.router,
            model.mtp_block.layers[0].decoder_layer.gate.router,  # type: ignore[union-attr]
        ]
        for router in routers:
            router.e_score_correction_bias.fill_(1.0)

        model.init_weights()

        for router in routers:
            torch.testing.assert_close(
                router.e_score_correction_bias,
                torch.zeros_like(router.e_score_correction_bias),
            )

    def test_update_bias_handles_main_and_shared_mtp_loads(self):
        # 验证主干 router 分别更新，而共享权重的多个 MTP 深度先聚合再只更新一次。
        config = _tiny_glm52_config()
        config.mtp_config = MTPConfig(num_layers=2, share_weights=True)
        with mock.patch("torch.cuda.Stream"):
            model = config.build()
        main_routers = [model.layers[str(idx)].gate.router for idx in (1, 2)]
        mtp_router = model.mtp_block.layers[0].decoder_layer.gate.router  # type: ignore[union-attr]
        for router in [*main_routers, mtp_router]:
            router.e_score_correction_bias.zero_()
        device = mtp_router.e_score_correction_bias.device
        expert_counts = torch.tensor(
            [
                [2, 0, 1, 1],
                [0, 2, 1, 1],
                [4, 0, 0, 0],
                [0, 3, 1, 0],
            ],
            device=device,
        )

        model.update_bias(expert_counts, expert_counts.float().mean(dim=1))

        update_speed = config.router.router_bias_update_speed
        torch.testing.assert_close(
            main_routers[0].e_score_correction_bias,
            torch.tensor([-update_speed, update_speed, 0.0, 0.0], device=device),
        )
        torch.testing.assert_close(
            main_routers[1].e_score_correction_bias,
            torch.tensor([update_speed, -update_speed, 0.0, 0.0], device=device),
        )
        torch.testing.assert_close(
            mtp_router.e_score_correction_bias,
            torch.tensor(
                [-update_speed, -update_speed, update_speed, update_speed],
                device=device,
            ),
        )


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires 2 CUDA devices")
class TestGlm52SequenceParallel(DeterministicDDPTestCase):
    def test_mtp_loss_and_gradients_match_full_sequence(self):
        # 验证 packed 序列经 SP2 后的 LM/MTP loss 与全局参数梯度匹配未切分基线。
        self.create_pg("cuda")
        config = _tiny_glm52_config()
        config.hidden_size = 128
        config.intermediate_size = 128
        config.moe_intermediate_size = 128
        config.mtp_config = MTPConfig(num_layers=2, share_weights=True)
        config.lm_loss_cfg = CELossConfig(mode="eager")
        config.dispatcher = None
        config.ep_size = 1

        torch.manual_seed(17)
        baseline_model = config.build().to(device="cuda", dtype=torch.bfloat16)
        baseline_model.init_weights()
        sp_model = config.build().to(device="cuda", dtype=torch.bfloat16)
        sp_model.load_state_dict(baseline_model.state_dict())
        baseline_model.train()
        sp_model.train()

        sequence_0 = torch.tensor([[2, 3, 4, 5, 6, 7]], device="cuda")
        sequence_1 = torch.tensor([[8, 9, 10, 11]], device="cuda")
        packed_inputs = (sequence_0[:, :-1], sequence_1[:, :-1])
        shifted_labels = torch.cat((sequence_0[:, 1:], sequence_1[:, 1:]), dim=1)

        baseline_seq_ctx = SequenceContext.from_input_ids(packed_inputs, device="cuda")
        baseline_data = {"seq_ctx": baseline_seq_ctx, "shifted_labels": shifted_labels}
        baseline_loss_ctx = baseline_model.build_loss_ctx_batch([baseline_data], sp_mesh=None)[0]
        baseline_output = baseline_model(seq_ctx=baseline_seq_ctx, loss_ctx=baseline_loss_ctx)
        (baseline_output["loss"] + baseline_output["mtp_loss"]).backward()

        baseline_gradients = {}
        for name, parameter in baseline_model.named_parameters():
            if parameter.grad is not None:
                gradient = parameter.grad.detach().float().clone()
                dist.all_reduce(gradient)
                baseline_gradients[name] = gradient / dist.get_world_size()

        sp_mesh = init_data_mesh("cuda", sp_size=2)["sp"]
        full_sp_seq_ctx = SequenceContext.from_input_ids(packed_inputs, device="cuda")
        sp_data = {"seq_ctx": full_sp_seq_ctx, "shifted_labels": shifted_labels}
        sp_loss_ctx = sp_model.build_loss_ctx_batch([sp_data], sp_mesh=sp_mesh)[0]
        sp_seq_ctx = full_sp_seq_ctx.split(sp_mesh)
        sp_output = sp_model(seq_ctx=sp_seq_ctx, loss_ctx=sp_loss_ctx)
        (sp_output["loss"] + sp_output["mtp_loss"]).backward()

        sp_gradients = {}
        for name, parameter in sp_model.named_parameters():
            if parameter.grad is not None:
                gradient = parameter.grad.detach().float().clone()
                dist.all_reduce(gradient)
                sp_gradients[name] = gradient / dist.get_world_size()

        torch.testing.assert_close(sp_output["loss"], baseline_output["loss"])
        torch.testing.assert_close(sp_output["mtp_loss"], baseline_output["mtp_loss"])
        self.assertEqual(sp_gradients.keys(), baseline_gradients.keys())
        max_relative_error = max(
            float((sp_gradients[name] - expected).norm() / expected.norm().clamp_min(1e-12))
            for name, expected in baseline_gradients.items()
        )
        min_cosine_similarity = min(
            float(
                torch.nn.functional.cosine_similarity(
                    sp_gradients[name].flatten(),
                    expected.flatten(),
                    dim=0,
                )
            )
            for name, expected in baseline_gradients.items()
        )
        self.assertLess(max_relative_error, 2e-2)
        self.assertGreater(min_cosine_similarity, 0.9999)

    @property
    def world_size(self) -> int:
        return 2
