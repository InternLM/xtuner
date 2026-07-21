import math
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import parametrize
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
from xtuner.v1.utils.compile import is_compiled_function
from xtuner.v1.utils.device import get_device
from xtuner.v1.utils.test_utils import init_data_mesh


GLM5_2_TINY_MOE_PATH = Path(os.environ["GLM5_2_TINY_MOE_PATH"])
DEVICE = get_device()


def _glm52_engine_config(dispatcher: str | None, ep_size: int) -> Glm52MoEConfig:
    cfg: Glm52MoEConfig = get_model_config_from_hf(GLM5_2_TINY_MOE_PATH)
    # These engine regressions isolate the base LM path; MTP training is covered
    # separately by test_sequence_parallel_mtp_train_step.
    cfg.mtp_config = None
    cfg.compile_cfg = False
    cfg.dispatcher = dispatcher
    cfg.ep_size = ep_size
    return cfg


def _tiny_glm52_checkpoint_config(dispatcher: str | None, ep_size: int) -> Glm52MoEConfig:
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


def _tiny_glm52_sp_mtp_config(dispatcher: str | None, ep_size: int) -> Glm52MoEConfig:
    cfg = _tiny_glm52_checkpoint_config(dispatcher=dispatcher, ep_size=ep_size)
    cfg.hidden_size = 128
    cfg.intermediate_size = 128
    cfg.moe_intermediate_size = 128
    cfg.attention.indexer_types = ["full", "shared", "full"]
    cfg.num_nextn_predict_layers = 2
    cfg.mtp_config = MTPConfig(
        num_layers=2,
        share_weights=True,
    )
    cfg.lm_loss_cfg = CELossConfig(mode="eager")
    return cfg


def _tilewise_float8_config() -> Float8Config:
    return Float8Config(
        scaling_granularity_gemm=ScalingGranularity.TILEWISE,
        scaling_granularity_grouped_gemm=ScalingGranularity.TILEWISE,
    )


def _build_engine(
    dispatcher: str | None,
    ep_size: int,
    hsdp_sharding_size: int | None = None,
    float8_cfg: Float8Config | None = None,
) -> TrainEngine:
    cfg = _glm52_engine_config(dispatcher=dispatcher, ep_size=ep_size)
    cfg.float8_cfg = float8_cfg
    optim_cfg = AdamWConfig()
    fsdp_cfg = FSDPConfig(cpu_offload=False, ep_size=ep_size, hsdp_sharding_size=hsdp_sharding_size)
    return TrainEngine(model_cfg=cfg, optim_cfg=optim_cfg, fsdp_cfg=fsdp_cfg)


def _build_tiny_checkpoint_engine(dispatcher: str | None, ep_size: int) -> TrainEngine:
    cfg = _tiny_glm52_checkpoint_config(dispatcher=dispatcher, ep_size=ep_size)
    optim_cfg = AdamWConfig()
    fsdp_cfg = FSDPConfig(cpu_offload=False, ep_size=ep_size)
    return TrainEngine(model_cfg=cfg, optim_cfg=optim_cfg, fsdp_cfg=fsdp_cfg)


def _build_train_engine_input(tokenizer: AutoTokenizer, loss_cfg: CELossConfig) -> list[ModelItem]:
    text = "吃葡萄不吐葡萄皮。GLM-5.2 engine smoke."
    input_ids = tokenizer.encode(text, return_tensors="pt").view(1, -1)
    labels = input_ids.clone()
    input_ids = input_ids[:, :-1]
    labels = labels[:, 1:]
    pack_len = 256 - input_ids.shape[1]
    input_ids = pad_to_max_length(input_ids, 0, max_length=256)
    labels = pad_to_max_length(labels, -100, max_length=256)

    seq_ctx = SequenceContext.from_input_ids((input_ids,), device=DEVICE)
    seq_ctx.num_padding = pack_len
    loss_ctx = loss_cfg.build(data={"shifted_labels": labels.to(DEVICE)}, sp_mesh=None)
    assert loss_ctx is not None
    loss_ctx = loss_cfg.loss_ctx_cls.build_batches([loss_ctx])[0]
    return [ModelItem(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx})]


def _run_tiny_engine_loss_curve(
    tokenizer: AutoTokenizer,
    dispatcher: str | None,
    ep_size: int,
    float8_cfg: Float8Config | None = None,
    num_steps: int = 3,
) -> torch.Tensor:
    engine = _build_engine(dispatcher=dispatcher, ep_size=ep_size, float8_cfg=float8_cfg)
    engine.from_hf(GLM5_2_TINY_MOE_PATH, strict=False)
    loss_cfg = CELossConfig()
    lr_cfg = LRConfig()
    warmup_steps = 1000 * lr_cfg.warmup_ratio

    def warmup_fn(step):
        return step / warmup_steps if step < warmup_steps else 1

    lr_scheduler = LambdaLR(engine.optimizer, warmup_fn)
    losses = []
    try:
        for _ in range(num_steps):
            loss_log = engine.train_step(_build_train_engine_input(tokenizer, loss_cfg))["logs_info"]
            grad_norm = engine.clip_grad_norm()
            engine.step_optimizer(grad_norm)
            lr_scheduler.step()
            losses.append(loss_log["reduced_llm_loss"])
        return torch.tensor(losses)
    finally:
        del engine
        torch.cuda.empty_cache()


def _local_tensor(value: torch.Tensor) -> torch.Tensor:
    return value._local_tensor if isinstance(value, DTensor) else value


def _full_bf16_tensor(value: torch.Tensor) -> torch.Tensor:
    value = value.full_tensor() if isinstance(value, DTensor) else value
    return value.bfloat16()


def _assert_tiny_checkpoint_engine_uses_shared_indexer(engine: TrainEngine) -> None:
    layer0_attn = engine.model.layers["0"].self_attn
    layer1_attn = engine.model.layers["1"].self_attn
    state_keys = set(engine.model.state_dict())

    assert layer0_attn.source_layer_idx == 0
    assert layer1_attn.source_layer_idx == 0
    assert hasattr(layer0_attn, "indexer")
    assert not hasattr(layer1_attn, "indexer")
    assert "layers.0.self_attn.indexer.wq_b.weight" in state_keys
    assert "layers.1.self_attn.indexer.wq_b.weight" not in state_keys


@unittest.skipUnless(
    torch.cuda.device_count() >= 8 and GLM5_2_TINY_MOE_PATH.exists(),
    f"requires at least 8 CUDA devices and tiny GLM-5.2 checkpoint at {GLM5_2_TINY_MOE_PATH}",
)
class TestGlm52MoEEngine(DeterministicDDPTestCase):
    @parametrize.parametrize(
        "compile_and_offload,dispatcher,ep_size",
        [
            (False, None, 1),
            (True, None, 1),
            (False, "all2all", 4),
        ],
    )
    def test_sequence_parallel_mtp_train_step(
        self,
        compile_and_offload: bool,
        dispatcher: str | None,
        ep_size: int,
    ):
        self.create_pg("cuda")

        model_cfg = _tiny_glm52_sp_mtp_config(dispatcher=dispatcher, ep_size=ep_size)
        if compile_and_offload:
            model_cfg.compile_cfg = None
        engine = TrainEngine(
            model_cfg=model_cfg,
            optim_cfg=AdamWConfig(lr=1e-3, foreach=False),
            fsdp_cfg=FSDPConfig(
                ep_size=ep_size,
                cpu_offload=False,
                recompute_ratio=1.0,
                torch_compile=compile_and_offload,
            ),
        )
        engine.init_model_weights()
        # Scratch initialization covers parameters, while production loads this
        # persistent router buffer from HF. Give the synthetic fixture the same
        # well-defined starting state instead of retaining torch.empty contents.
        with torch.no_grad():
            for module in engine.model.modules():
                if isinstance(module, NoAuxRouter):
                    module.e_score_correction_bias.zero_()

        sequence_0 = torch.arange(2, 12).view(1, -1)
        sequence_1 = torch.arange(20, 28).view(1, -1)
        packed_inputs = (sequence_0[:, :-1], sequence_1[:, :-1])
        shifted_labels = torch.cat((sequence_0[:, 1:], sequence_1[:, 1:]), dim=1)
        sp_mesh = init_data_mesh(str(DEVICE), sp_size=2)["sp"]

        try:
            full_seq_ctx = SequenceContext.from_input_ids(packed_inputs, device=DEVICE)
            data = {"seq_ctx": full_seq_ctx, "shifted_labels": shifted_labels}
            loss_ctx = engine.model.build_loss_ctx_batch([data], sp_mesh=sp_mesh)[0]
            seq_ctx = full_seq_ctx.split(sp_mesh)

            optimized_env = (
                {
                    "XTUNER_ACTIVATION_OFFLOAD": "1",
                    "XTUNER_DSA_TOPK_OFFLOAD": "1",
                }
                if compile_and_offload
                else {}
            )
            with mock.patch.dict(os.environ, optimized_env):
                step_info = engine.train_step([ModelItem(seq_ctx=seq_ctx, loss_ctx=loss_ctx)])
                grad_norm = engine.clip_grad_norm()
                engine.step_optimizer(grad_norm)

            self.assertTrue(math.isfinite(step_info["total_loss"]))
            self.assertTrue(math.isfinite(step_info["logs_info"]["reduced_llm_loss"]))
            self.assertTrue(math.isfinite(step_info["logs_info"]["reduced_mtp_loss"]))
            self.assertTrue(math.isfinite(float(grad_norm)))
            self.assertTrue(engine.optimizer.state)
            self.assertEqual(seq_ctx.dsa_topk_cache.indices, {})
            self.assertEqual(seq_ctx.dsa_topk_cache.offloaded, {})
            if compile_and_offload:
                wrapped_layer = engine.model.layers["0"]._checkpoint_wrapped_module
                self.assertTrue(is_compiled_function(wrapped_layer.forward))
        finally:
            del engine
            torch.cuda.empty_cache()

    @parametrize.parametrize(
        "device,dispatcher,ep_size",
        [
            ("cuda", "all2all", 8),
        ],
    )
    def test_moe_engine_train(self, device, dispatcher, ep_size):
        self.create_pg(device)

        engine = _build_engine(dispatcher=dispatcher, ep_size=ep_size)
        engine.from_hf(GLM5_2_TINY_MOE_PATH, strict=False)
        tokenizer = AutoTokenizer.from_pretrained(GLM5_2_TINY_MOE_PATH)
        loss_cfg = CELossConfig()
        lr_cfg = LRConfig()
        total_steps = 1000
        warmup_steps = total_steps * lr_cfg.warmup_ratio

        def warmup_fn(step):
            return step / warmup_steps if step < warmup_steps else 1

        lr_scheduler = LambdaLR(engine.optimizer, warmup_fn)
        losses = []
        try:
            for _ in range(10):
                loss_log = engine.train_step(_build_train_engine_input(tokenizer, loss_cfg))["logs_info"]
                grad_norm = engine.clip_grad_norm()
                engine.step_optimizer(grad_norm)
                lr_scheduler.step()
                losses.append(loss_log["reduced_llm_loss"])

            # TODO: Replace this temporary baseline with the official GLM-5.2
            # engine-training reference once it is available.
            losses_ref = torch.tensor(
                [11.9815, 11.9940, 12.0483, 11.9849, 11.8578, 11.8326, 11.3845, 11.2681, 10.2004, 9.8985]
            )
            self._check_loss_curve(torch.tensor(losses), losses_ref)
        finally:
            torch.cuda.empty_cache()

    @parametrize.parametrize(
        "device,dispatcher,ep_size",
        [
            ("cuda", "all2all", 8),
        ],
    )
    def test_activation_offload_train_step(self, device, dispatcher, ep_size):
        self.create_pg(device)

        engine = _build_engine(dispatcher=dispatcher, ep_size=ep_size)
        engine.from_hf(GLM5_2_TINY_MOE_PATH, strict=False)
        tokenizer = AutoTokenizer.from_pretrained(GLM5_2_TINY_MOE_PATH)
        loss_cfg = CELossConfig()

        try:
            # Exercise GLM's sparse-layer activation-offload branch through the
            # public TrainEngine path, including backward and optimizer update.
            with mock.patch.dict(os.environ, {"XTUNER_ACTIVATION_OFFLOAD": "1"}):
                loss_log = engine.train_step(_build_train_engine_input(tokenizer, loss_cfg))["logs_info"]
                grad_norm = engine.clip_grad_norm()
                engine.step_optimizer(grad_norm)

            assert math.isfinite(loss_log["reduced_llm_loss"])
            assert math.isfinite(float(grad_norm))
        finally:
            torch.cuda.empty_cache()

    @parametrize.parametrize(
        "device,dispatcher,ep_size",
        [
            ("cuda", None, 1),
        ],
    )
    def test_tile_wise_fp8_train_matches_bf16_loss_curve(self, device, dispatcher, ep_size):
        self.create_pg(device)

        tokenizer = AutoTokenizer.from_pretrained(GLM5_2_TINY_MOE_PATH)
        bf16_losses = _run_tiny_engine_loss_curve(
            tokenizer=tokenizer,
            dispatcher=dispatcher,
            ep_size=ep_size,
            float8_cfg=None,
        )
        fp8_losses = _run_tiny_engine_loss_curve(
            tokenizer=tokenizer,
            dispatcher=dispatcher,
            ep_size=ep_size,
            float8_cfg=_tilewise_float8_config(),
        )

        # FP8 tilewise training quantizes GEMM and grouped-GEMM weights/activations.
        # Compare against the same bf16 training path to catch GLM-specific
        # precision regressions while allowing expected quantization drift.
        self._check_loss_curve(losses=fp8_losses, losses_ref=bf16_losses, sim_tol=2e-2, rtol=5e-2)

    @parametrize.parametrize(
        "device,dispatcher,ep_size",
        [
            ("cuda", "all2all", 2),
            ("cuda", "all2all", 4),
            ("cuda", "all2all", 8),
        ],
    )
    def test_tile_wise_fp8_train_with_ep(self, device, dispatcher, ep_size):
        self.create_pg(device)

        tokenizer = AutoTokenizer.from_pretrained(GLM5_2_TINY_MOE_PATH)
        losses = _run_tiny_engine_loss_curve(
            tokenizer=tokenizer,
            dispatcher=dispatcher,
            ep_size=ep_size,
            float8_cfg=_tilewise_float8_config(),
            num_steps=1,
        )
        self.assertTrue(torch.isfinite(losses).all(), f"FP8 EP training produced non-finite losses: {losses}")

    @parametrize.parametrize(
        "device,ep_size,hsdp_sharding_size",
        [
            ("cuda", 1, 8),
        ],
    )
    def test_save_and_load(self, device, ep_size, hsdp_sharding_size):
        self.create_pg(device)

        temp_dir = tempfile.mkdtemp() if dist.get_rank() == 0 else None
        syncdir = [temp_dir]
        dist.broadcast_object_list(syncdir, src=0)
        temp_dir = Path(syncdir[0])
        try:
            engine = _build_engine(dispatcher=None, ep_size=ep_size, hsdp_sharding_size=hsdp_sharding_size)
            engine.from_hf(GLM5_2_TINY_MOE_PATH, strict=False)
            engine.save_hf(hf_dir=str(temp_dir), save_dtype=torch.bfloat16)
            dist.barrier()
            time.sleep(1)

            engine2 = _build_engine(dispatcher=None, ep_size=ep_size, hsdp_sharding_size=hsdp_sharding_size)
            engine2.from_hf(temp_dir)

            state_dict = engine.model.state_dict()
            state_dict2 = engine2.model.state_dict()
            assert len(state_dict) == len(state_dict2)
            for key, val in state_dict.items():
                val = _full_bf16_tensor(val)
                val2 = _full_bf16_tensor(state_dict2[key])
                if val.shape != val2.shape and val2.shape[0] >= val.shape[0]:
                    val2 = val2[: val.shape[0]]
                self.assertTrue(torch.equal(val, val2), f"Mismatch in {key}, {val.shape} and {val2.shape}")
        finally:
            dist.barrier()
            if dist.get_rank() == 0:
                shutil.rmtree(temp_dir, ignore_errors=True)
            torch.cuda.empty_cache()

    @parametrize.parametrize(
        "device,dispatcher,ep_size",
        [
            ("cuda", "all2all", 8),
        ],
    )
    def test_checkpoint_save_load(self, device, dispatcher, ep_size):
        self.create_pg(device)

        temp_dir = tempfile.mkdtemp() if dist.get_rank() == 0 else None
        syncdir = [temp_dir]
        dist.broadcast_object_list(syncdir, src=0)
        weights_dir = Path(syncdir[0]) / "weights"
        try:
            # The HF tiny checkpoint keeps production GLM-5.2 expert dimensions.
            # Use a small in-memory GLM config here so the DCP regression still
            # covers one dense layer plus one MoE layer without a large DCP load.
            engine = _build_tiny_checkpoint_engine(dispatcher=dispatcher, ep_size=ep_size)
            _assert_tiny_checkpoint_engine_uses_shared_indexer(engine)
            engine.init_model_weights()
            with torch.no_grad():
                for module in engine.model.modules():
                    if isinstance(module, NoAuxRouter):
                        bias = module.e_score_correction_bias
                        bias.copy_(torch.arange(bias.numel(), device=bias.device, dtype=bias.dtype))
            engine.save_dcp(weights_dir=weights_dir)
            dist.barrier()
            time.sleep(1)

            engine2 = _build_tiny_checkpoint_engine(dispatcher=dispatcher, ep_size=ep_size)
            _assert_tiny_checkpoint_engine_uses_shared_indexer(engine2)
            engine2.init_model_weights()
            engine2.load_dcp(weights_dir=weights_dir)

            state_dict = engine.model.state_dict()
            state_dict2 = engine2.model.state_dict()
            assert len(state_dict) == len(state_dict2)
            for key, val in state_dict.items():
                val = _local_tensor(val)
                val2 = _local_tensor(state_dict2[key])
                self.assertTrue(torch.equal(val, val2), f"Mismatch in {key}, {val.shape} and {val2.shape}")

            opt_state = engine.optimizer.state_dict()["state"]
            opt_state2 = engine2.optimizer.state_dict()["state"]
            assert len(opt_state) == len(opt_state2)
            assert len(opt_state) != 0
            for param_id, cur_state_dict in opt_state.items():
                cur_state_dict2 = opt_state2[param_id]
                assert len(cur_state_dict) == len(cur_state_dict2)
                assert len(cur_state_dict) != 0
                for state_key, val in cur_state_dict.items():
                    val = _local_tensor(val)
                    val2 = _local_tensor(cur_state_dict2[state_key])
                    self.assertTrue(
                        torch.equal(val, val2),
                        f"Mismatch in optimizer {param_id}.{state_key}, {val.shape} and {val2.shape}",
                    )
        finally:
            dist.barrier()
            if dist.get_rank() == 0:
                shutil.rmtree(syncdir[0], ignore_errors=True)
            torch.cuda.empty_cache()

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "8"))
