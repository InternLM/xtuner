import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

import parametrize
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.optim.lr_scheduler import LambdaLR

from transformers import AutoTokenizer
from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.engine.train_engine import TrainEngine
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model import get_model_config_from_hf
from xtuner.v1.model.base import ModelItem
from xtuner.v1.model.moe.glm52 import Glm52MoEConfig
from xtuner.v1.model.moe.moe import SequenceContext
from xtuner.v1.module.attention import DSAMLAConfig
from xtuner.v1.module.router.noaux_router import NoAuxRouter, NoAuxRouterConfig
from xtuner.v1.utils import pad_to_max_length
from xtuner.v1.utils.device import get_device


GLM5_2_TINY_MOE_PATH = Path(
    os.environ.get("GLM5_2_TINY_MOE_PATH", "/mnt/shared-storage-user/zhaopenghao/slime0701/ckpts/GLM-5.2-tiny-4L")
)
DEVICE = get_device()


def _glm52_engine_config(dispatcher: str | None, ep_size: int) -> Glm52MoEConfig:
    cfg: Glm52MoEConfig = get_model_config_from_hf(GLM5_2_TINY_MOE_PATH)
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
        index_topk=4,
        index_head_dim=4,
        index_n_heads=2,
        indexer_types=["full", "shared"],
        compile_cfg=False,
        dispatcher=dispatcher,
        ep_size=ep_size,
    )


def _build_engine(dispatcher: str | None, ep_size: int, hsdp_sharding_size: int | None = None) -> TrainEngine:
    cfg = _glm52_engine_config(dispatcher=dispatcher, ep_size=ep_size)
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
        "device,dispatcher,ep_size",
        [
            ("cuda", "all2all", 8),
        ],
    )
    def test_moe_engine_train(self, device, dispatcher, ep_size):
        self.create_pg(device)

        engine = _build_engine(dispatcher=dispatcher, ep_size=ep_size)
        engine.from_hf(GLM5_2_TINY_MOE_PATH)
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
                [12.5259, 12.5333, 12.5516, 12.5002, 12.4089, 12.3696, 11.9864, 11.8645, 10.7606, 10.3792]
            )
            self._check_loss_curve(torch.tensor(losses), losses_ref)
        finally:
            torch.cuda.empty_cache()

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
            engine.from_hf(GLM5_2_TINY_MOE_PATH)
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
