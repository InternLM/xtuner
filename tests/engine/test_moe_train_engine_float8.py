import os
import tempfile
import shutil
import time
from pathlib import Path

import parametrize
import torch
import torch.distributed as dist
from xtuner._testing import DeterministicDDPTestCase
from transformers import AutoTokenizer

from xtuner.v1.model.moe.moe import SequenceContext
from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.engine.train_engine import TrainEngine
from xtuner.v1.float8.config import ScalingGranularity, Float8Config
from xtuner.v1.model.moe.qwen3 import Qwen3MoE30BA3Config
from xtuner.v1.utils import pad_to_max_length
from torch.optim.lr_scheduler import LambdaLR
from xtuner.v1.utils.device import get_device
from xtuner.v1.model.base import ModelItem
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model.moe.moe import BalancingLossConfig



# Qwen3 30B A3
QWEN3_MOE_PATH = os.environ["QWEN3_MOE_PATH"]
DEVICE = get_device()


class TestMoEEngineFloat8(DeterministicDDPTestCase):

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
            float8_cfg=Float8Config(
                scaling_granularity_gemm=ScalingGranularity.TILEWISE,
                scaling_granularity_grouped_gemm=ScalingGranularity.TILEWISE,
            ),
        )
        optim_cfg: AdamWConfig = AdamWConfig()
        lr_cfg: LRConfig = LRConfig()
        fsdp_cfg: FSDPConfig = FSDPConfig(
            cpu_offload=False,
            ep_size=ep_size,
            # hsdp_sharding_size=8,
        )
        engine = TrainEngine(
            model_cfg=moe_cfg,
            optim_cfg=optim_cfg,
            fsdp_cfg=fsdp_cfg,
        )
        engine.from_hf(hf_path=QWEN3_MOE_PATH)

        loss_cfg = CELossConfig()

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
            seq_ctx_list = [seq_ctx]
            LossContext = loss_cfg.loss_ctx_cls
            loss_ctx = loss_cfg.build(data={"shifted_labels": labels}, sp_mesh=None)
            loss_ctx_list = [loss_ctx]
            loss_ctx_list = LossContext.build_batches(loss_ctx_list)
            loss_ctx = loss_ctx_list[0]
            seq_ctx = seq_ctx_list[0]
            engine_input = [ModelItem(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx})]
            loss_log = engine.train_step(engine_input)["logs_info"]
            grad_norm = engine.clip_grad_norm()
            engine.step_optimizer(grad_norm)
            lr_scheduler.step()
            losses.append(loss_log["reduced_llm_loss"])
        losses = torch.tensor(losses)
        losses_ref = torch.tensor([2.4234, 2.4234, 1.5270, 1.1483, 0.8904, 0.6388, 0.3963, 0.2589, 0.1519, 0.1101])

        self._check_loss_curve(losses, losses_ref, sim_tol=0.01, rtol=0.01)
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
            float8_cfg=Float8Config(
                scaling_granularity_gemm=ScalingGranularity.TENSORWISE,
                scaling_granularity_grouped_gemm=ScalingGranularity.TILEWISE,
            )
        )

        optim_cfg: AdamWConfig = AdamWConfig()
        lr_cfg: LRConfig = LRConfig()
        fsdp_cfg: FSDPConfig = FSDPConfig(
            cpu_offload=False,
            ep_size=ep_size,
            # hsdp_sharding_size=hsdp_sharding_size,
        )
        engine = TrainEngine(
            model_cfg=moe_cfg, optim_cfg=optim_cfg, fsdp_cfg=fsdp_cfg
        )
        engine.from_hf(hf_path=QWEN3_MOE_PATH)

        loss_cfg = CELossConfig()

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
            seq_ctx_list = [seq_ctx]
            LossContext = loss_cfg.loss_ctx_cls
            loss_ctx = loss_cfg.build(data={"shifted_labels": labels}, sp_mesh=None)
            loss_ctx_list = [loss_ctx]
            loss_ctx_list = LossContext.build_batches(loss_ctx_list)
            loss_ctx = loss_ctx_list[0]
            seq_ctx = seq_ctx_list[0]
            engine_input = [ModelItem(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx})]
            loss_log = engine.train_step(engine_input)["logs_info"]
            grad_norm = engine.clip_grad_norm()
            engine.step_optimizer(grad_norm)
            lr_scheduler.step()
            losses.append(loss_log["reduced_llm_loss"])
        losses_ref = torch.tensor([2.3874, 2.3874, 1.7667, 1.3585, 1.0056, 0.6969, 0.4769, 0.2874, 0.1653, 0.1120])
        losses = torch.tensor(losses)

        self._check_loss_curve(losses, losses_ref, sim_tol=0.01, rtol=0.01)

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
            float8_cfg=Float8Config(
                scaling_granularity_gemm=ScalingGranularity.TILEWISE,
                scaling_granularity_grouped_gemm=ScalingGranularity.TILEWISE,
            ),
        )
        optim_cfg: AdamWConfig = AdamWConfig()
        lr_cfg: LRConfig = LRConfig()
        fsdp_cfg: FSDPConfig = FSDPConfig(
            cpu_offload=False,
            ep_size=ep_size,
            # hsdp_sharding_size=hsdp_sharding_size,
        )
        loss_cfg = CELossConfig()
        engine = TrainEngine(
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
            seq_ctx_list = [seq_ctx]
            LossContext = loss_cfg.loss_ctx_cls
            loss_ctx = loss_cfg.build(data={"shifted_labels": labels}, sp_mesh=None)
            loss_ctx_list = [loss_ctx]
            loss_ctx_list = LossContext.build_batches(loss_ctx_list)
            loss_ctx = loss_ctx_list[0]
            seq_ctx = seq_ctx_list[0]
            engine_input = [ModelItem(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx})]
            logs_info = engine.train_step(engine_input)["logs_info"]
            grad_norm = engine.clip_grad_norm()
            engine.step_optimizer(grad_norm)
            lr_scheduler.step()
            losses.append(logs_info["reduced_llm_loss"])
        losses_ref = torch.tensor([2.4234, 2.4234, 2.4093, 2.4306, 2.3924, 2.4475, 2.3995, 2.3729, 2.3292, 2.3122])
        losses = torch.tensor(losses)
        self._check_loss_curve(losses, losses_ref)

        if dist.get_rank() == 0:
            shutil.rmtree(temp_dir)

        torch.cuda.empty_cache()
        try:
            dist.destroy_process_group(pg)
        except:
            pass

    @parametrize.parametrize(
        "device,ep_size",
        [
            ("cuda", 1),
        ],
    )
    def test_float8_dcp_resume(self, device, ep_size):
        """Regression test for float8 DCP resume bug.

        Float8Handler lazily registers safe globals on first train step. Before
        the fix, load_dcp() would raise _pickle.UnpicklingError when called on
        a fresh engine (before any train step) because the float8 tensor types
        had not been registered as safe globals yet.
        """
        pg = self.create_pg(device)
        temp_dir = tempfile.mkdtemp()
        if dist.get_rank() == 0:
            temp_dir = [temp_dir]
        else:
            temp_dir = [None]
        dist.broadcast_object_list(temp_dir, src=0)
        temp_dir = temp_dir[0]

        float8_cfg = Float8Config(
            scaling_granularity_gemm=ScalingGranularity.TILEWISE,
            scaling_granularity_grouped_gemm=ScalingGranularity.TILEWISE,
        )
        model_cfg = Qwen3MoE30BA3Config(
            num_hidden_layers=1,
            balancing_loss_cfg=BalancingLossConfig(),
        )
        model_cfg.float8_cfg = float8_cfg

        optim_cfg: AdamWConfig = AdamWConfig()
        fsdp_cfg: FSDPConfig = FSDPConfig(cpu_offload=False, ep_size=ep_size)
        loss_cfg = CELossConfig()

        engine = TrainEngine(model_cfg=model_cfg, optim_cfg=optim_cfg, fsdp_cfg=fsdp_cfg)
        engine.from_hf(hf_path=QWEN3_MOE_PATH, strict=False)

        tok = AutoTokenizer.from_pretrained(QWEN3_MOE_PATH)
        txt = (
            "리팩토링하면서 Float8Handler 생성을 eager 에서 lazy 로 바꾸는 과정에서 add_safe_globals"
            "의 호출 타이밍이 dcp.load 이후로 밀려난 것이 이 버그의 원인입니다."
        )
        input_ids = tok.encode(txt, return_tensors="pt").view(1, -1)
        labels = input_ids.clone()
        input_ids = input_ids[:, :-1]
        labels = labels[:, 1:]
        seq_len = 128
        input_ids = pad_to_max_length(input_ids, 0, max_length=seq_len)
        labels = pad_to_max_length(labels, -100, max_length=seq_len)
        pack_len = seq_len - input_ids.shape[1]

        def make_engine_input():
            seq_ctx = SequenceContext.from_input_ids((input_ids,), device=DEVICE)
            seq_ctx.num_padding = pack_len
            lbl = labels.to(DEVICE)
            LossContext = loss_cfg.loss_ctx_cls
            loss_ctx = loss_cfg.build(data={"shifted_labels": lbl}, sp_mesh=None)
            loss_ctx_list = LossContext.build_batches([loss_ctx])
            return [ModelItem(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx_list[0]})]

        # Run one training step so float8 tensors are materialized in the checkpoint.
        engine.train_step(make_engine_input())
        grad_norm = engine.clip_grad_norm()
        engine.step_optimizer(grad_norm)

        dcp_dir = os.path.join(temp_dir, "step_1")
        engine.save_dcp(Path(dcp_dir))

        # engine1's train_step triggered Float8Handler.__init__ which called
        # add_safe_globals globally. Clear those globals now to simulate a fresh
        # process that has never run a training step, which is exactly the
        # scenario that occurs during a real DCP resume.
        saved_safe_globals = torch.serialization.get_safe_globals()
        torch.serialization.clear_safe_globals()
        try:
            # Create a fresh engine that has never run a training step, then call
            # load_dcp.
            engine2 = TrainEngine(model_cfg=model_cfg, optim_cfg=optim_cfg, fsdp_cfg=fsdp_cfg)
            engine2.model.set_hf(QWEN3_MOE_PATH)
            engine2.load_dcp(Path(dcp_dir))
        finally:
            torch.serialization.add_safe_globals(saved_safe_globals)

        if dist.get_rank() == 0:
            shutil.rmtree(temp_dir)

        torch.cuda.empty_cache()
        try:
            dist.destroy_process_group(pg)
        except Exception:
            pass

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "8"))

    @property
    def destroy_pg_upon_exit(self) -> bool:
        return False


class TestMoEEngineFloat8Case2(DeterministicDDPTestCase):

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
        )
        optim_cfg: AdamWConfig = AdamWConfig()
        fsdp_cfg: FSDPConfig = FSDPConfig(
            cpu_offload=False,
            ep_size=ep_size,
            hsdp_sharding_size=hsdp_sharding_size,
        )
        engine_bf16 = TrainEngine(
            model_cfg=moe_cfg,
            optim_cfg=optim_cfg,
            fsdp_cfg=fsdp_cfg,
        )
        engine_bf16.from_hf(hf_path=QWEN3_MOE_PATH)
        engine_bf16.save_hf(
            hf_dir=temp_dir,
            save_dtype=torch.bfloat16,
        )

        dist.barrier()
        time.sleep(1)

        moe_cfg_fp8 = Qwen3MoE30BA3Config(
            balancing_loss_cfg=BalancingLossConfig(),
            float8_cfg=Float8Config(
                scaling_granularity_gemm=ScalingGranularity.TILEWISE,
                scaling_granularity_grouped_gemm=ScalingGranularity.TILEWISE,
            ),
        )
        engine_fp8 = TrainEngine(
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
