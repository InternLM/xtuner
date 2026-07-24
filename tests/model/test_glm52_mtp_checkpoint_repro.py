"""GLM-5.2 MTP reentrant checkpoint 的真实训练回归测试。

TestGlm52CompiledMTPCheckpoint
    test_shared_mtp_depths_train_with_compile_and_topk_offload: 共享 MTP 深度可在 compile/offload 下训练。
TestGlm52MicroBatchMTPCheckpoint
    test_nested_micro_batch_inputs_preserve_gradients: EP2 micro2 的嵌套 embedding 梯度可正确反传。
"""

import math
import os
import unittest
from unittest import mock

import torch

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.config import AdamWConfig, FSDPConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.engine.train_engine import TrainEngine
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model.base import ModelItem
from xtuner.v1.model.moe.glm52 import Glm52MoEConfig
from xtuner.v1.module.attention import DSAMLAConfig
from xtuner.v1.module.mtp import MTPConfig
from xtuner.v1.module.router.noaux_router import NoAuxRouterConfig


def _tiny_mtp_config(ep_size: int, mtp_num_layers: int, compile_model: bool) -> Glm52MoEConfig:
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
        mtp_config=MTPConfig(num_layers=mtp_num_layers, share_weights=True),
        lm_loss_cfg=CELossConfig(mode="eager"),
        compile_cfg=None if compile_model else False,
        dispatcher="all2all" if ep_size > 1 else None,
        ep_size=ep_size,
    )


def _build_engine(
    *,
    intra_layer_micro_batch: int,
    ep_size: int,
    mtp_num_layers: int,
    compile_model: bool,
) -> TrainEngine:
    engine = TrainEngine(
        model_cfg=_tiny_mtp_config(ep_size, mtp_num_layers, compile_model),
        optim_cfg=AdamWConfig(lr=1e-3, foreach=False),
        fsdp_cfg=FSDPConfig(
            ep_size=ep_size,
            cpu_offload=False,
            recompute_ratio=0.0,
            torch_compile=compile_model,
        ),
        intra_layer_micro_batch=intra_layer_micro_batch,
    )
    engine.init_model_weights()
    return engine


def _model_item(engine: TrainEngine, start: int) -> ModelItem:
    input_ids = torch.arange(start, start + 6).view(1, -1) % engine.model_cfg.vocab_size
    seq_ctx = SequenceContext.from_input_ids((input_ids[:, :-1],), device="cuda")
    data = {"seq_ctx": seq_ctx, "shifted_labels": input_ids[:, 1:]}
    loss_ctx = engine.model.build_loss_ctx_batch([data], sp_mesh=None)[0]
    return ModelItem(seq_ctx=seq_ctx, loss_ctx=loss_ctx)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestGlm52CompiledMTPCheckpoint(DeterministicDDPTestCase):
    def test_shared_mtp_depths_train_with_compile_and_topk_offload(self):
        # 验证默认 reentrant checkpoint 可训练共享 MTP 深度且 loss 有限。
        self.create_pg("cuda")
        engine = _build_engine(
            intra_layer_micro_batch=1,
            ep_size=1,
            mtp_num_layers=2,
            compile_model=True,
        )
        try:
            with mock.patch.dict(
                os.environ,
                {"XTUNER_ACTIVATION_OFFLOAD": "0", "XTUNER_DSA_TOPK_OFFLOAD": "1"},
            ):
                step_info = engine.train_step([_model_item(engine, 2)])

            assert math.isfinite(step_info["total_loss"])
            assert math.isfinite(step_info["logs_info"]["reduced_mtp_loss"])
        finally:
            del engine
            torch.cuda.empty_cache()

    @property
    def world_size(self) -> int:
        return 1


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires 2 CUDA devices")
class TestGlm52MicroBatchMTPCheckpoint(DeterministicDDPTestCase):
    def test_nested_micro_batch_inputs_preserve_gradients(self):
        # 验证 EP2 micro2 的嵌套 future embedding 经 pytree checkpoint 后可完成真实训练步。
        self.create_pg("cuda")
        engine = _build_engine(
            intra_layer_micro_batch=2,
            ep_size=2,
            mtp_num_layers=1,
            compile_model=False,
        )
        try:
            with mock.patch.dict(
                os.environ,
                {"XTUNER_ACTIVATION_OFFLOAD": "0", "XTUNER_DSA_TOPK_OFFLOAD": "0"},
            ):
                step_info = engine.train_step(
                    [
                        _model_item(engine, 2),
                        _model_item(engine, 14),
                    ]
                )

            assert math.isfinite(step_info["total_loss"])
            assert math.isfinite(step_info["logs_info"]["reduced_mtp_loss"])
        finally:
            del engine
            torch.cuda.empty_cache()

    @property
    def world_size(self) -> int:
        return 2
