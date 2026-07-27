"""GLM-5.2 TrainEngine 的训练、优化组合与 DCP 持久化行为测试。

TestGlm52OptimizedEngine
    test_sp2_ep4_micro2_compile_offload_train_step: SP2、EP4、micro2、compile 与双 offload 可联合训练。
TestGlm52PretrainedEngine
    test_ep8_loss_curve_matches_reference: 预训练权重的 EP8 优化轨迹匹配数值基线。
    test_tilewise_fp8_loss_curve_matches_bf16: tilewise FP8 训练轨迹接近 BF16。
    test_tilewise_fp8_ep4_train_step: tilewise FP8 与 EP4 联合训练产生有限 loss。
TestGlm52CheckpointEngine
    test_dcp_round_trip_preserves_model_and_optimizer: DCP 往返保留模型与优化器状态。
"""

import math
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.optim.lr_scheduler import LambdaLR

from transformers import AutoTokenizer
from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.engine.train_engine import TrainEngine
from xtuner.v1.float8.config import Float8Config, ScalingGranularity
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model import get_model_config_from_hf
from xtuner.v1.model.base import ModelItem
from xtuner.v1.model.moe.glm52 import Glm52MoEConfig
from xtuner.v1.module.attention import DSAMLAConfig
from xtuner.v1.module.mtp import MTPConfig
from xtuner.v1.module.router.noaux_router import NoAuxRouter, NoAuxRouterConfig
from xtuner.v1.utils import pad_to_max_length
from xtuner.v1.utils.device import get_device
from xtuner.v1.utils.test_utils import init_data_mesh


GLM5_2_TINY_MOE_PATH = Path(os.environ["GLM5_2_TINY_MOE_PATH"])
DEVICE = get_device()


def _pretrained_config(
    dispatcher: str | None,
    ep_size: int,
    float8_cfg: Float8Config | None = None,
) -> Glm52MoEConfig:
    config: Glm52MoEConfig = get_model_config_from_hf(GLM5_2_TINY_MOE_PATH)
    config.mtp_config = None
    config.compile_cfg = False
    config.dispatcher = dispatcher
    config.ep_size = ep_size
    config.float8_cfg = float8_cfg
    return config


def _tiny_checkpoint_config(dispatcher: str | None, ep_size: int) -> Glm52MoEConfig:
    return Glm52MoEConfig(
        vocab_size=64,
        max_position_embeddings=128,
        pad_token_id=0,
        eos_token_id=1,
        hf_eos_token_id=[1],
        num_hidden_layers=2,
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
            indexer_types=["full", "shared"],
            sparse_mla_backend="torch",
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
        mlp_layer_types=["dense", "sparse"],
        compile_cfg=False,
        dispatcher=dispatcher,
        ep_size=ep_size,
    )


def _tiny_sp_mtp_config() -> Glm52MoEConfig:
    config = _tiny_checkpoint_config(dispatcher="all2all", ep_size=4)
    config.hidden_size = 128
    config.intermediate_size = 128
    config.moe_intermediate_size = 128
    config.attention.indexer_types = ["full", "shared", "full"]
    config.num_nextn_predict_layers = 2
    config.mtp_config = MTPConfig(num_layers=2, share_weights=True)
    config.lm_loss_cfg = CELossConfig(mode="eager")
    config.compile_cfg = None
    return config


def _tilewise_float8_config() -> Float8Config:
    return Float8Config(
        scaling_granularity_gemm=ScalingGranularity.TILEWISE,
        scaling_granularity_grouped_gemm=ScalingGranularity.TILEWISE,
    )


def _build_pretrained_engine(
    dispatcher: str | None,
    ep_size: int,
    float8_cfg: Float8Config | None = None,
) -> TrainEngine:
    return TrainEngine(
        model_cfg=_pretrained_config(dispatcher, ep_size, float8_cfg),
        optim_cfg=AdamWConfig(),
        fsdp_cfg=FSDPConfig(cpu_offload=False, ep_size=ep_size),
    )


def _build_train_input(tokenizer: AutoTokenizer, loss_cfg: CELossConfig) -> list[ModelItem]:
    input_ids = tokenizer.encode(
        "吃葡萄不吐葡萄皮。GLM-5.2 engine smoke.",
        return_tensors="pt",
    ).view(1, -1)
    labels = input_ids[:, 1:]
    input_ids = input_ids[:, :-1]
    pack_len = 256 - input_ids.shape[1]
    input_ids = pad_to_max_length(input_ids, 0, max_length=256)
    labels = pad_to_max_length(labels, -100, max_length=256)

    seq_ctx = SequenceContext.from_input_ids((input_ids,), device=DEVICE)
    seq_ctx.num_padding = pack_len
    loss_ctx = loss_cfg.build(data={"shifted_labels": labels.to(DEVICE)}, sp_mesh=None)
    assert loss_ctx is not None
    loss_ctx = loss_cfg.loss_ctx_cls.build_batches([loss_ctx])[0]
    return [ModelItem(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx})]


def _run_loss_curve(
    tokenizer: AutoTokenizer,
    dispatcher: str | None,
    ep_size: int,
    float8_cfg: Float8Config | None,
    num_steps: int,
) -> torch.Tensor:
    engine = _build_pretrained_engine(dispatcher, ep_size, float8_cfg)
    engine.from_hf(GLM5_2_TINY_MOE_PATH, strict=False)
    loss_cfg = CELossConfig()
    warmup_steps = 1000 * LRConfig().warmup_ratio
    lr_scheduler = LambdaLR(
        engine.optimizer,
        lambda step: step / warmup_steps if step < warmup_steps else 1,
    )
    losses = []
    try:
        for _ in range(num_steps):
            logs = engine.train_step(_build_train_input(tokenizer, loss_cfg))["logs_info"]
            grad_norm = engine.clip_grad_norm()
            engine.step_optimizer(grad_norm)
            lr_scheduler.step()
            losses.append(logs["reduced_llm_loss"])
        return torch.tensor(losses)
    finally:
        del engine
        torch.cuda.empty_cache()


@unittest.skipUnless(torch.cuda.device_count() >= 8, "requires 8 CUDA devices")
class TestGlm52OptimizedEngine(DeterministicDDPTestCase):
    def test_sp2_ep4_micro2_compile_offload_train_step(self):
        # 验证生产优化组合经两次梯度累积后 loss、梯度与优化器状态均有效。
        self.create_pg("cuda")
        model_cfg = _tiny_sp_mtp_config()
        engine = TrainEngine(
            model_cfg=model_cfg,
            optim_cfg=AdamWConfig(lr=1e-3, foreach=False),
            fsdp_cfg=FSDPConfig(
                ep_size=4,
                cpu_offload=False,
                recompute_ratio=1.0,
                torch_compile=True,
            ),
            intra_layer_micro_batch=2,
        )
        engine.init_model_weights()
        sp_mesh = init_data_mesh(str(DEVICE), sp_size=2)["sp"]
        data_batches = []
        seq_ctx_list = []

        try:
            for micro_batch_idx in range(4):
                start = 2 + micro_batch_idx * 12
                input_ids = torch.arange(start, start + 10).view(1, -1) % model_cfg.vocab_size
                full_seq_ctx = SequenceContext.from_input_ids((input_ids[:, :-1],), device=DEVICE)
                data = {"seq_ctx": full_seq_ctx, "shifted_labels": input_ids[:, 1:]}
                loss_ctx = engine.model.build_loss_ctx_batch([data], sp_mesh=sp_mesh)[0]
                seq_ctx = full_seq_ctx.split(sp_mesh)
                seq_ctx_list.append(seq_ctx)
                data_batches.append(ModelItem(seq_ctx=seq_ctx, loss_ctx=loss_ctx))

            with mock.patch.dict(
                os.environ,
                {"XTUNER_ACTIVATION_OFFLOAD": "1", "XTUNER_DSA_TOPK_OFFLOAD": "1"},
            ):
                step_info = engine.train_step(data_batches)
                grad_norm = engine.clip_grad_norm()
                engine.step_optimizer(grad_norm)

            self.assertTrue(math.isfinite(step_info["total_loss"]))
            self.assertTrue(math.isfinite(step_info["logs_info"]["reduced_llm_loss"]))
            self.assertTrue(math.isfinite(step_info["logs_info"]["reduced_mtp_loss"]))
            self.assertTrue(math.isfinite(float(grad_norm)))
            self.assertTrue(engine.optimizer.state)
            for seq_ctx in seq_ctx_list:
                self.assertEqual(seq_ctx.dsa_topk_cache.indices, {})
                self.assertEqual(seq_ctx.dsa_topk_cache.offloaded, {})
        finally:
            del engine
            torch.cuda.empty_cache()

    @property
    def world_size(self) -> int:
        return 8


@unittest.skipUnless(
    torch.cuda.device_count() >= 8 and GLM5_2_TINY_MOE_PATH.exists(),
    f"requires 8 CUDA devices and GLM-5.2 checkpoint at {GLM5_2_TINY_MOE_PATH}",
)
class TestGlm52PretrainedEngine(DeterministicDDPTestCase):
    def test_ep8_loss_curve_matches_reference(self):
        # 验证预训练 GLM-5.2 经 EP8 连续三个优化步骤后复现已知 loss 轨迹。
        self.create_pg("cuda")
        tokenizer = AutoTokenizer.from_pretrained(GLM5_2_TINY_MOE_PATH)

        losses = _run_loss_curve(
            tokenizer,
            dispatcher="all2all",
            ep_size=8,
            float8_cfg=None,
            num_steps=3,
        )

        self._check_loss_curve(
            losses,
            torch.tensor([11.9815, 11.9940, 12.0483]),
        )

    def test_tilewise_fp8_loss_curve_matches_bf16(self):
        # 验证相同预训练权重的 tilewise FP8 两步轨迹与 BF16 基线保持在量化容差内。
        self.create_pg("cuda")
        tokenizer = AutoTokenizer.from_pretrained(GLM5_2_TINY_MOE_PATH)

        bf16_losses = _run_loss_curve(
            tokenizer,
            dispatcher=None,
            ep_size=1,
            float8_cfg=None,
            num_steps=2,
        )
        fp8_losses = _run_loss_curve(
            tokenizer,
            dispatcher=None,
            ep_size=1,
            float8_cfg=_tilewise_float8_config(),
            num_steps=2,
        )

        self._check_loss_curve(
            losses=fp8_losses,
            losses_ref=bf16_losses,
            sim_tol=2e-2,
            rtol=5e-2,
        )

    def test_tilewise_fp8_ep4_train_step(self):
        # 验证 tilewise FP8 与 EP4 联合执行一次真实训练步时 loss 保持有限。
        self.create_pg("cuda")
        tokenizer = AutoTokenizer.from_pretrained(GLM5_2_TINY_MOE_PATH)

        losses = _run_loss_curve(
            tokenizer,
            dispatcher="all2all",
            ep_size=4,
            float8_cfg=_tilewise_float8_config(),
            num_steps=1,
        )

        self.assertTrue(torch.isfinite(losses).all())

    @property
    def world_size(self) -> int:
        return 8


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires 2 CUDA devices")
class TestGlm52CheckpointEngine(DeterministicDDPTestCase):
    def test_dcp_round_trip_preserves_model_and_optimizer(self):
        # 验证 tiny EP2 engine 的 DCP 往返后模型参数、router buffer 与优化器状态逐项相同。
        self.create_pg("cuda")
        temp_dir = tempfile.mkdtemp() if dist.get_rank() == 0 else None
        syncdir = [temp_dir]
        dist.broadcast_object_list(syncdir, src=0)
        weights_dir = Path(syncdir[0]) / "weights"

        try:
            config = _tiny_checkpoint_config(dispatcher="all2all", ep_size=2)
            engine = TrainEngine(
                model_cfg=config,
                optim_cfg=AdamWConfig(),
                fsdp_cfg=FSDPConfig(cpu_offload=False, ep_size=2),
            )
            engine.init_model_weights()
            with torch.no_grad():
                for module in engine.model.modules():
                    if isinstance(module, NoAuxRouter):
                        bias = module.e_score_correction_bias
                        bias.copy_(torch.arange(bias.numel(), device=bias.device, dtype=bias.dtype))
            engine.save_dcp(weights_dir=weights_dir)
            dist.barrier()

            restored = TrainEngine(
                model_cfg=config,
                optim_cfg=AdamWConfig(),
                fsdp_cfg=FSDPConfig(cpu_offload=False, ep_size=2),
            )
            restored.init_model_weights()
            restored.load_dcp(weights_dir=weights_dir)

            expected_state = engine.model.state_dict()
            actual_state = restored.model.state_dict()
            assert actual_state.keys() == expected_state.keys()
            for key, expected in expected_state.items():
                expected = expected._local_tensor if isinstance(expected, DTensor) else expected
                actual = actual_state[key]
                actual = actual._local_tensor if isinstance(actual, DTensor) else actual
                self.assertTrue(torch.equal(actual, expected), f"model state mismatch: {key}")

            expected_optimizer = engine.optimizer.state_dict()["state"]
            actual_optimizer = restored.optimizer.state_dict()["state"]
            assert actual_optimizer.keys() == expected_optimizer.keys()
            assert expected_optimizer
            for param_id, expected_values in expected_optimizer.items():
                actual_values = actual_optimizer[param_id]
                assert actual_values.keys() == expected_values.keys()
                for state_key, expected in expected_values.items():
                    expected = expected._local_tensor if isinstance(expected, DTensor) else expected
                    actual = actual_values[state_key]
                    actual = actual._local_tensor if isinstance(actual, DTensor) else actual
                    self.assertTrue(
                        torch.equal(actual, expected),
                        f"optimizer state mismatch: {param_id}.{state_key}",
                    )
        finally:
            dist.barrier()
            if dist.get_rank() == 0:
                shutil.rmtree(syncdir[0], ignore_errors=True)
            torch.cuda.empty_cache()

    @property
    def world_size(self) -> int:
        return 2
