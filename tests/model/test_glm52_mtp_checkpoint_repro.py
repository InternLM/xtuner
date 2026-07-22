"""MTP 两种 checkpoint 反例及无 MTP 边界的最小真实训练测试。"""

import math
import os
import unittest
from unittest import mock

import pytest
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.config import AdamWConfig, FSDPConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.engine.train_engine import TrainEngine
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model.base import ModelItem
from xtuner.v1.model.moe.glm52 import Glm52MoEConfig
from xtuner.v1.model.utils import checkpoint_wrapper
from xtuner.v1.module.attention import DSAMLAConfig
from xtuner.v1.module.mtp import MTPConfig
from xtuner.v1.module.router.noaux_router import NoAuxRouter, NoAuxRouterConfig


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
    use_reentrant: bool | None = None,
) -> TrainEngine:
    checkpoint_config = {}
    if use_reentrant is not None:
        checkpoint_config["mtp_checkpoint_use_reentrant"] = use_reentrant

    engine = TrainEngine(
        model_cfg=_tiny_mtp_config(ep_size, mtp_num_layers, compile_model),
        optim_cfg=AdamWConfig(lr=1e-3, foreach=False),
        fsdp_cfg=FSDPConfig(
            ep_size=ep_size,
            cpu_offload=False,
            recompute_ratio=0.0,
            torch_compile=compile_model,
            **checkpoint_config,
        ),
        intra_layer_micro_batch=intra_layer_micro_batch,
    )
    engine.init_model_weights()
    with torch.no_grad():
        for module in engine.model.modules():
            if isinstance(module, NoAuxRouter):
                module.e_score_correction_bias.zero_()
    return engine


def _model_item(engine: TrainEngine, start: int) -> ModelItem:
    input_ids = torch.arange(start, start + 6).view(1, -1) % engine.model_cfg.vocab_size
    seq_ctx = SequenceContext.from_input_ids((input_ids[:, :-1],), device="cuda")
    data = {"seq_ctx": seq_ctx, "shifted_labels": input_ids[:, 1:]}
    loss_ctx = engine.model.build_loss_ctx_batch([data], sp_mesh=None)[0]
    return ModelItem(seq_ctx=seq_ctx, loss_ctx=loss_ctx)


def test_mtp_checkpoint_defaults_to_reentrant():
    assert FSDPConfig().mtp_checkpoint_use_reentrant is True


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestGlm52MTPCheckpointRepro(DeterministicDDPTestCase):
    def test_non_reentrant_checkpoint_fails_for_compiled_shared_mtp_depths(self):
        self.create_pg("cuda")
        # Case 1：compile, topk offload, MTP share weights and depth > 1
        # 两个 logical depth 共用一个物理 MTP layer。depth1 original 在
        # cache 为空时计算 indexer；由于 non-reentrant 无法正确更新 cache 计数，
        # depth1 replay 却误用了 depth2 留下的 cache，即 original=COMPUTE、
        # replay=REUSE。
        #
        # indexer 始终 no_grad，所以不开 compile 时可以简化为：
        #   original: indexer -> SparseMLA 保存 [A, B, C]
        #   replay:   cache   -> SparseMLA 保存 [A, B, C]
        # 两张清单仍能对齐。compile + top-k offload 会让两条路径经过不同的 graph
        # break 和 compiled block，例如保存 [A, B, C, D] 与 [A, X, C, D]；
        # checkpoint 按槽位比较 metadata，发现 B 与 X 不同后才抛出错误。
        # 字母仅表示保存槽位，不代表真实 Tensor，也不表示 checkpoint 比较数值。
        engine = _build_engine(
            intra_layer_micro_batch=1,
            use_reentrant=False,
            ep_size=1,
            mtp_num_layers=2,
            compile_model=True,
        )
        try:
            with mock.patch.dict(
                os.environ,
                {"XTUNER_ACTIVATION_OFFLOAD": "0", "XTUNER_DSA_TOPK_OFFLOAD": "1"},
            ):
                with pytest.raises(torch.utils.checkpoint.CheckpointError, match="different metadata"):
                    engine.train_step([_model_item(engine, 2)])
        finally:
            del engine
            torch.cuda.empty_cache()

    def test_reentrant_checkpoint_supports_compiled_shared_mtp_depths(self):
        self.create_pg("cuda")
        # Case 1：compile, topk offload, MTP share weights and depth > 1
        # 使用默认 reentrant 重跑相同配置。original 不建立内部 autograd graph，
        # 因而不需要和 replay 核对保存清单；replay 可以复用 original 的离散 top-k。
        # DSA 又能正确统计两次 replay，并在最后一次使用后释放 cache，所以训练应
        # 正常完成。
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


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestGlm52DecoderCheckpointRepro(DeterministicDDPTestCase):
    def test_non_reentrant_checkpoint_fails_without_mtp_for_shared_dsa_cache(self):
        self.create_pg("cuda")
        # Case 1 的根因不依赖 MTP：普通 full decoder 生产 top-k、后续 shared
        # decoder 消费同一 cache，只要二者都用 non-reentrant checkpoint，也能构造
        # original=COMPUTE、replay=REUSE。compile + offload 再把这个状态错误暴露为
        # compiled block 的保存槽位错位。
        config = _tiny_mtp_config(ep_size=1, mtp_num_layers=1, compile_model=True)
        config.first_k_dense_replace = 2
        config.mlp_layer_types = ["dense", "dense"]
        config.attention.indexer_types = ["full", "shared"]
        config.mtp_config = None

        with torch.device("meta"):
            model = config.build()
        for layer_idx, layer in model.layers.items():
            model.layers[layer_idx] = checkpoint_wrapper(
                layer,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )
        model.fully_shard(
            FSDPConfig(
                ep_size=1,
                cpu_offload=False,
                recompute_ratio=0.0,
                torch_compile=True,
            )
        )
        model.init_weights()
        model.train()

        try:
            input_ids = torch.arange(2, 7).view(1, -1)
            seq_ctx = SequenceContext.from_input_ids((input_ids,), device="cuda")
            hidden_states = model.embed_tokens(seq_ctx.input_ids)
            position_embeddings = model.rotary_emb(hidden_states, seq_ctx.position_ids)

            with mock.patch.dict(
                os.environ,
                {"XTUNER_ACTIVATION_OFFLOAD": "0", "XTUNER_DSA_TOPK_OFFLOAD": "1"},
            ):
                for decoder_layer in model.layers.values():
                    hidden_states = decoder_layer(
                        hidden_states,
                        position_embeddings=position_embeddings,
                        seq_ctx=seq_ctx,
                    )
                with pytest.raises(torch.utils.checkpoint.CheckpointError, match="different metadata"):
                    hidden_states.float().square().mean().backward()
        finally:
            del model
            torch.cuda.empty_cache()

    @property
    def world_size(self) -> int:
        return 1


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires 2 CUDA devices")
class TestGlm52MTPMicroBatchCheckpointRepro(DeterministicDDPTestCase):
    def test_reentrant_checkpoint_supports_two_mtp_micro_batches(self):
        self.create_pg("cuda")
        # Case 2：ep>1, intra layer micro batch > 1 (eg. = 2)
        # micro2 传入 [embedding_0, embedding_1]。修复前 checkpoint 只看到
        # 外层 list，看不到里面两个 Tensor，所以 replay 前不会 detach 它们。replay
        # 的内部 backward 先使用一次原 embedding graph；外层 backward 再使用一次，
        # 就会报 backward through graph twice。
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
                # pytree wrapper 先展开成两个顶层 Tensor，checkpoint 因而能逐个
                # detach。replay 不再直接进入旧 graph；算出的梯度由 checkpoint
                # backward 返回，并与 main LM/MTP 的梯度正常汇合。
                step_info = engine.train_step([_model_item(engine, 2), _model_item(engine, 14)])
            assert math.isfinite(step_info["total_loss"])
            assert math.isfinite(step_info["logs_info"]["reduced_mtp_loss"])
        finally:
            del engine
            torch.cuda.empty_cache()

    @property
    def world_size(self) -> int:
        return 2
