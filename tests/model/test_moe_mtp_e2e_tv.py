"""MoE MTP e2e TV loss 的前向回归测试。

TestMoEMTPE2ETVLoss
    test_tv_target_uses_normalized_main_hidden_states: 单 micro-batch 路径中 TV target 必须是 final norm 之后的主干状态。
"""

import unittest

import torch

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.loss import MTPE2ETVLossContext
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model.moe.glm52 import DSAMLAConfig, Glm52MoEConfig
from xtuner.v1.module.mtp import MTPConfig
from xtuner.v1.module.router.noaux_router import NoAuxRouterConfig


def _tiny_e2e_tv_config() -> Glm52MoEConfig:
    return Glm52MoEConfig(
        vocab_size=32,
        max_position_embeddings=64,
        pad_token_id=0,
        eos_token_id=1,
        hf_eos_token_id=[1],
        num_hidden_layers=2,
        first_k_dense_replace=0,
        hidden_size=128,
        intermediate_size=128,
        moe_intermediate_size=128,
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
            indexer_types=["full", "shared", "full"],
        ),
        hf_head_dim=4,
        qk_head_dim=8,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        router=NoAuxRouterConfig(
            n_group=1,
            topk_group=1,
            scoring_func="sigmoid",
            norm_topk_prob=True,
            router_scaling_factor=2.5,
        ),
        mlp_layer_types=["sparse", "sparse"],
        mtp_config=MTPConfig(num_layers=2, share_weights=True, loss_type="e2e_tv"),
        lm_loss_cfg=CELossConfig(mode="eager"),
        compile_cfg=False,
    )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestMoEMTPE2ETVLoss(DeterministicDDPTestCase):
    def test_tv_target_uses_normalized_main_hidden_states(self):
        # 主 LM 分支（norm + lm_head）在 MTP 之后执行，TV target 仍须与主 LM 分支看到同一份 normalized 状态。
        self.create_pg("cuda")
        cfg = _tiny_e2e_tv_config()
        with torch.device("cuda"):
            model = cfg.build().to(torch.bfloat16)
        model.init_weights()
        with torch.no_grad():
            # 非平凡的 norm 权重让遗漏 normalization 在 loss 上可见。
            model.norm.weight.uniform_(0.5, 2.0)

        mtp_io: dict[str, object] = {}

        def capture_mtp_io(_module, args, output):
            mtp_io["main_hidden_states"] = args[0]
            mtp_io["draft_hidden_states"] = [depth_output["hidden_states"] for depth_output in output]

        input_ids = (torch.arange(2, 19) * 7 % cfg.vocab_size).view(1, -1)
        seq_ctx = SequenceContext.from_input_ids((input_ids[:, :-1],), device="cuda")
        data = {"seq_ctx": seq_ctx, "shifted_labels": input_ids[:, 1:]}
        loss_ctx = model.build_loss_ctx_batch([data], sp_mesh=None)[0]
        tv_loss_ctx = loss_ctx["mtp_e2e_tv"]
        assert isinstance(tv_loss_ctx, MTPE2ETVLossContext)

        hook = model.mtp_block.register_forward_hook(capture_mtp_io)
        try:
            output = model(seq_ctx=seq_ctx, loss_ctx=loss_ctx)
        finally:
            hook.remove()

        with torch.no_grad():
            expected_tv_loss, _ = model.lm_head(
                (model.norm(mtp_io["main_hidden_states"]), mtp_io["draft_hidden_states"]),
                tv_loss_ctx,
            )
        expected_mtp_loss = expected_tv_loss * cfg.mtp_config.loss_scaling_factor
        torch.testing.assert_close(output["mtp_loss"].detach().float(), expected_mtp_loss.float())

    @property
    def world_size(self) -> int:
        return 1
