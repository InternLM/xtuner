import os
import tempfile
import shutil
from pydantic import ConfigDict
import time

import parametrize
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_distributed import DistributedTestBase
from transformers import AutoModelForCausalLM, AutoTokenizer

from xtuner.v1.model.moe.moe import MoEConfig, SequenceContext
from xtuner.v1.config import AdamWConfig, Float8Config, FSDPConfig, LRConfig, MoEConfig, OptimConfig, BalancingLossConfig, ZLossConfig
from xtuner.v1.engine.moe_train_engine import MoETrainEngine
from xtuner.v1.float8.float8_tensor import ScalingGranularity
from xtuner.v1.model.moe.qwen3 import Qwen3MoE30BA3Config
from xtuner.v1.utils import pad_to_max_length
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR
from xtuner.v1.loss import CELossContext
from xtuner.utils.device import get_device

# Qwen3 30B A3
QWEN3_MOE_PATH = os.environ["QWEN3_MOE_PATH"]
DEVICE = get_device()


class TestMoEEngineFloat8(DistributedTestBase):

    @parametrize.parametrize(
        "device,ep_size,hsdp_sharding_size",
        [
            ("cuda", 1, int(os.getenv("XTUNER_TEST_WORLD_SIZE", "8"))),  # todo: test ep8 and hsdp, OOM in 8 gpus
        ],
    )
    def test_tile_wise_fp8(self, device, ep_size, hsdp_sharding_size):
        pg = self.create_pg(device)

        moe_cfg = Qwen3MoE30BA3Config(
            balancing_loss_cfg=BalancingLossConfig(),
            z_loss_cfg=ZLossConfig(),
            float8_cfg=Float8Config(
                scaling_granularity_gemm=ScalingGranularity.TILEWISE,
                scaling_granularity_grouped_gemm=ScalingGranularity.TILEWISE,
            ),
        )
        optim_cfg: AdamWConfig = AdamWConfig()
        lr_cfg: LRConfig = LRConfig()
        fsdp_cfg: FSDPConfig = FSDPConfig(
            torch_compile=True,
            cpu_offload=False,
            ep_size=ep_size,
            # hsdp_sharding_size=8,
        )
        engine = MoETrainEngine(
            model_cfg=moe_cfg,
            optim_cfg=optim_cfg,
            fsdp_cfg=fsdp_cfg,
        )
        engine.from_hf(hf_path=QWEN3_MOE_PATH)

        total_steps = 1000
        warmup_steps = total_steps * lr_cfg.warmup_ratio
        def warmup_fn(x):
            return x / warmup_steps if x < warmup_steps else 1
        
        lr_scheduler = LambdaLR(engine.optimizer, warmup_fn)

        tok = AutoTokenizer.from_pretrained(
            QWEN3_MOE_PATH
        )
        txt = "根据国际地球自转和参考系服务机构的数据，今年夏天是自2020年以来第六次地球自转加速。7月9日将成为有史以来最短的一天，比平时短1.3到1.6毫秒。 "
        input_ids = tok.encode(txt, return_tensors="pt").view(1, -1)
        labels = input_ids.clone()
        input_ids = input_ids[:, :-1]
        labels = labels[:, 1:]
        pack_len = 8192 - input_ids.shape[1]
        input_ids = pad_to_max_length(input_ids, 0, max_length=8192)
        labels = pad_to_max_length(labels, -100, max_length=8192)
        losses = []
        for _ in range(10):
            seq_ctx = SequenceContext.from_input_ids((input_ids,), device=DEVICE)
            labels = labels.to(DEVICE)
            seq_ctx.num_padding = pack_len
            data_batch = [{'seq_ctx': seq_ctx, 'labels': labels}]
            loss_ctx = CELossContext()
            data_batch = loss_ctx.build_list_ctx(data_batch, device=DEVICE)
            loss_log, _ = engine.train_step(data_batch)
            grad_norm = engine.clip_grad_norm()
            engine.step_optimizer(grad_norm)
            lr_scheduler.step()
            losses.append(loss_log["reduced_llm_loss"])
        losses_ref = [2.41, 2.41, 1.79, 1.39, 1.02, 0.68, 0.52, 0.31, 0.18, 0.12]

        for loss, loss_ref in zip(losses, losses_ref):
            self.assertTrue(abs(loss - loss_ref) < 0.1)
        
        torch.cuda.empty_cache()
        try:
            dist.destroy_process_group(pg)
        except:
            pass
    
    @parametrize.parametrize(
        "device,ep_size,hsdp_sharding_size",
        [
            ("cuda", 1, int(os.getenv("XTUNER_TEST_WORLD_SIZE", "8"))),  # todo: test ep8 and hsdp, OOM in 8 gpus
        ],
    )
    def test_tensor_wise_fp8(self, device, ep_size, hsdp_sharding_size):
        pg = self.create_pg(device)

        moe_cfg = Qwen3MoE30BA3Config(
            ep_size=ep_size,
            balancing_loss_cfg=BalancingLossConfig(),
            z_loss_cfg=ZLossConfig(),
            float8_cfg=Float8Config(
                scaling_granularity_gemm=ScalingGranularity.TENSORWISE,
                scaling_granularity_grouped_gemm=ScalingGranularity.TILEWISE,
            )
        )

        optim_cfg: AdamWConfig = AdamWConfig()
        lr_cfg: LRConfig = LRConfig()
        fsdp_cfg: FSDPConfig = FSDPConfig(
            torch_compile=True,
            cpu_offload=False,
            ep_size=ep_size,
            # hsdp_sharding_size=hsdp_sharding_size,
        )
        engine = MoETrainEngine(
            model_cfg=moe_cfg, optim_cfg=optim_cfg, fsdp_cfg=fsdp_cfg
        )
        engine.from_hf(hf_path=QWEN3_MOE_PATH)

        total_steps = 1000
        warmup_steps = total_steps * lr_cfg.warmup_ratio
        def warmup_fn(x):
            return x / warmup_steps if x < warmup_steps else 1
        
        lr_scheduler = LambdaLR(engine.optimizer, warmup_fn)

        tok = AutoTokenizer.from_pretrained(
            QWEN3_MOE_PATH
        )
        txt = "根据国际地球自转和参考系服务机构的数据，今年夏天是自2020年以来第六次地球自转加速。7月9日将成为有史以来最短的一天，比平时短1.3到1.6毫秒。 "
        input_ids = tok.encode(txt, return_tensors="pt").view(1, -1)
        labels = input_ids.clone()
        input_ids = input_ids[:, :-1]
        labels = labels[:, 1:]
        pack_len = 8192 - input_ids.shape[1]
        input_ids = pad_to_max_length(input_ids, 0, max_length=8192)
        labels = pad_to_max_length(labels, -100, max_length=8192)
        losses = []
        for _ in range(10):
            seq_ctx = SequenceContext.from_input_ids((input_ids,), device=DEVICE)
            labels = labels.to(DEVICE)
            seq_ctx.num_padding = pack_len
            data_batch = [{'seq_ctx': seq_ctx, 'labels': labels}]
            loss_ctx = CELossContext()
            data_batch = loss_ctx.build_list_ctx(data_batch, device=DEVICE)
            loss_log, _ = engine.train_step(data_batch)
            grad_norm = engine.clip_grad_norm()
            engine.step_optimizer(grad_norm)
            lr_scheduler.step()
            losses.append(loss_log["reduced_llm_loss"])
        losses_ref = [2.45, 2.45, 1.78, 1.31, 0.95, 0.67, 0.45, 0.31, 0.18, 0.12]

        for loss, loss_ref in zip(losses, losses_ref):
            self.assertTrue(abs(loss - loss_ref) < 0.1)
        
        torch.cuda.empty_cache()
        try:
            dist.destroy_process_group(pg)
        except:
            pass
    
    @parametrize.parametrize(
        "device,ep_size,hsdp_sharding_size",
        [
            ("cuda", 1, int(os.getenv("XTUNER_TEST_WORLD_SIZE", "8"))),  # todo: test ep8 and hsdp, OOM in 8 gpus
        ],
    )
    def test_save_and_load(self, device, ep_size, hsdp_sharding_size):
        pg = self.create_pg(device)
        temp_dir = tempfile.mkdtemp()
        if dist.get_rank() == 0:
            temp_dir = [temp_dir]
        else:
            temp_dir = [None]
        dist.broadcast_object_list(temp_dir, src=0)
        temp_dir = temp_dir[0]

        moe_cfg = Qwen3MoE30BA3Config(
            ep_size=ep_size,
            balancing_loss_cfg=BalancingLossConfig(),
            z_loss_cfg=ZLossConfig(),
            float8_cfg=Float8Config(
                scaling_granularity_gemm=ScalingGranularity.TILEWISE,
                scaling_granularity_grouped_gemm=ScalingGranularity.TILEWISE,
            ),
        )
        optim_cfg: AdamWConfig = AdamWConfig()
        lr_cfg: LRConfig = LRConfig()
        fsdp_cfg: FSDPConfig = FSDPConfig(
            torch_compile=True,
            cpu_offload=False,
            ep_size=ep_size,
            # hsdp_sharding_size=hsdp_sharding_size,
        )
        engine = MoETrainEngine(
            model_cfg=moe_cfg,
            optim_cfg=optim_cfg,
            fsdp_cfg=fsdp_cfg,
        )
        engine.from_hf(hf_path=QWEN3_MOE_PATH)

        engine.save_hf(
            hf_dir=temp_dir,
            save_dtype=torch.float8_e4m3fn,
        )

        engine.from_hf(
            hf_path=temp_dir,
        )

        total_steps = 1000
        warmup_steps = total_steps * lr_cfg.warmup_ratio
        def warmup_fn(x):
            return x / warmup_steps if x < warmup_steps else 1
        
        lr_scheduler = LambdaLR(engine.optimizer, warmup_fn)

        torch.cuda.empty_cache()

        tok = AutoTokenizer.from_pretrained(
            QWEN3_MOE_PATH
        )
        txt = "根据国际地球自转和参考系服务机构的数据，今年夏天是自2020年以来第六次地球自转加速。7月9日将成为有史以来最短的一天，比平时短1.3到1.6毫秒。 "
        input_ids = tok.encode(txt, return_tensors="pt").view(1, -1)
        labels = input_ids.clone()
        input_ids = input_ids[:, :-1]
        labels = labels[:, 1:]
        input_ids = pad_to_max_length(input_ids, 0, max_length=8192)
        labels = pad_to_max_length(labels, -100, max_length=8192)
        pad_len = 8192 - input_ids.shape[1]
        losses = []
        for _ in range(10):
            seq_ctx = SequenceContext.from_input_ids((input_ids,), device=DEVICE)
            labels = labels.to(DEVICE)
            seq_ctx.num_padding = pad_len
            seq_ctx.to('cuda')
            data_batch = [{'seq_ctx': seq_ctx, 'labels': labels}]
            loss_ctx = CELossContext()
            data_batch = loss_ctx.build_list_ctx(data_batch, device=DEVICE)
            loss_log, _ = engine.train_step(data_batch)
            grad_norm = engine.clip_grad_norm()
            engine.step_optimizer(grad_norm)
            lr_scheduler.step()
            losses.append(loss_log["reduced_llm_loss"])
        losses_ref = [2.41, 2.41, 2.47, 2.42, 2.44, 2.44, 2.42, 2.38, 2.31, 2.30]

        for loss, loss_ref in zip(losses, losses_ref):
            self.assertTrue(abs(loss - loss_ref) < 0.1)

        if dist.get_rank() == 0:
            shutil.rmtree(temp_dir)
        
        torch.cuda.empty_cache()
        try:
            dist.destroy_process_group(pg)
        except:
            pass
    
    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "8"))

    @property
    def destroy_pg_upon_exit(self) -> bool:
        return False


class TestMoEEngineFloat8Case2(DistributedTestBase):

    @parametrize.parametrize(
        "device,ep_size,hsdp_sharding_size",
        [
            ("cuda", 1, int(os.getenv("XTUNER_TEST_WORLD_SIZE", "6"))),  # todo: test ep8 and hsdp, OOM in 8 gpus
        ],
    )
    def test_save_and_load1(self, device, ep_size, hsdp_sharding_size):
        self.create_pg(device)
        temp_dir = tempfile.mkdtemp()
        if dist.get_rank() == 0:
            temp_dir = [temp_dir]
        else:
            temp_dir = [None]
        dist.broadcast_object_list(temp_dir, src=0)
        temp_dir = temp_dir[0]
        moe_cfg = Qwen3MoE30BA3Config(
            ep_size=ep_size,
            balancing_loss_cfg=BalancingLossConfig(),
            z_loss_cfg=ZLossConfig(),
        )
        optim_cfg: AdamWConfig = AdamWConfig()
        fsdp_cfg: FSDPConfig = FSDPConfig(
            torch_compile=True,
            cpu_offload=False,
            ep_size=ep_size,
            max_length=8192,
            hsdp_sharding_size=hsdp_sharding_size,
        )
        engine_bf16 = MoETrainEngine(
            model_cfg=moe_cfg,
            optim_cfg=optim_cfg,
            fsdp_cfg=fsdp_cfg,
        )
        engine_bf16.init_model()
        engine_bf16.save_hf(
            hf_dir=temp_dir,
            save_dtype=torch.bfloat16,
        )

        dist.barrier()
        time.sleep(1)

        moe_cfg_fp8 = Qwen3MoE30BA3Config(
            balancing_loss_cfg=BalancingLossConfig(),
            z_loss_cfg=ZLossConfig(),
            float8_cfg=Float8Config(
                scaling_granularity_gemm=ScalingGranularity.TILEWISE,
                scaling_granularity_grouped_gemm=ScalingGranularity.TILEWISE,
            ),
        )
        engine_fp8 = MoETrainEngine(
            model_cfg=moe_cfg_fp8,
            optim_cfg=optim_cfg,
            fsdp_cfg=fsdp_cfg,
        )
        engine_fp8.from_hf(
            hf_path=temp_dir,
        )

        state_dict_bf16 = engine_bf16.model.state_dict()
        state_dict_fp8 = engine_fp8.model.state_dict()
        for key, val in state_dict_bf16.items():
            val_fp8 = state_dict_fp8[key]
            val = val.full_tensor().bfloat16()
            val_fp8 = val_fp8.full_tensor().bfloat16()
            self.assertTrue(torch.equal(val, val_fp8[:val.shape[0]]), f"Mismatch in {key} between bf16 and fp8, {val} and {val_fp8[:val.shape[0]]}")
            self.assertTrue((val_fp8[val.shape[0]:] == 0).all())

        temp_dir2 = tempfile.mkdtemp()
        if dist.get_rank() == 0:
            temp_dir2 = [temp_dir2]
        else:
            temp_dir2 = [None]
        dist.broadcast_object_list(temp_dir2, src=0)
        temp_dir2 = temp_dir2[0]

        engine_fp8.save_hf(
            hf_dir=temp_dir2,
            save_dtype=torch.bfloat16,
        )

        engine_fp8.from_hf(
            hf_path=temp_dir2,
        )

        state_dict_bf16 = engine_bf16.model.state_dict()
        state_dict_fp8 = engine_fp8.model.state_dict()
        for key, val in state_dict_bf16.items():
            val_fp8 = state_dict_fp8[key]
            val = val.full_tensor().bfloat16()
            val_fp8 = val_fp8.full_tensor().bfloat16()
            self.assertTrue(torch.equal(val, val_fp8[:val.shape[0]]), f"Mismatch in {key} between bf16 and fp8, {val} and {val_fp8[:val.shape[0]]}")
            self.assertTrue((val_fp8[val.shape[0]:] == 0).all())

        if dist.get_rank() == 0:
            shutil.rmtree(temp_dir)
            shutil.rmtree(temp_dir2)

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "6"))
