import os
from pathlib import Path
import tempfile
import shutil
import time
from torch.distributed.device_mesh import init_device_mesh
import parametrize
import torch
import torch.distributed as dist
from xtuner._testing import DeterministicDDPTestCase
from transformers import AutoTokenizer
from collections import defaultdict

from xtuner.v1.model.moe.moe import SequenceContext
from xtuner.v1.model.base import ModelItem
from xtuner.v1.loss.ce_loss import CELossConfig, CELossContextInputItem
from xtuner.v1.model.moe.qwen3 import Qwen3MoE30BA3Config
from xtuner.v1.config import FSDPConfig, LRConfig, AdamWConfig
from xtuner.v1.model.moe.moe import BalancingLossConfig, ZLossConfig
from xtuner.v1.engine.train_engine import TrainEngine
from torch.optim.lr_scheduler import LambdaLR
from xtuner.v1.utils import pad_to_max_length
from xtuner.v1.utils.device import get_device
from xtuner.v1.utils.test_utils import init_data_mesh
from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEDecoderLayer


# Qwen3 30B A3
QWEN3_MOE_PATH = os.environ["QWEN3_MOE_PATH"]
DEVICE = get_device()


class TestMoEEngine(DeterministicDDPTestCase):
    @parametrize.parametrize(
        "device,ep_size,sp_size",
        [
            ("cuda", 1, 1),
            ("cuda", 1, 2),
        ],
    )
    def test_moe_engine_train(self, device, ep_size, sp_size):
        pg = self.create_pg(device)

        moe_cfg = Qwen3MoE30BA3Config(
            ep_size=ep_size,
            balancing_loss_cfg=BalancingLossConfig(),
            z_loss_cfg=ZLossConfig(),
        )
        optim_cfg: AdamWConfig = AdamWConfig()
        lr_cfg: LRConfig = LRConfig()
        fsdp_cfg: FSDPConfig = FSDPConfig(
            torch_compile=False,
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

        tok = AutoTokenizer.from_pretrained(QWEN3_MOE_PATH)
        txt = "根据国际地球自转和参考系服务机构的数据，今年夏天是自2020年以来第六次地球自转加速。7月9日将成为有史以来最短的一天，比平时短1.3到1.6毫秒。 "
        input_ids = tok.encode(txt, return_tensors="pt").view(1, -1)
        labels = input_ids.clone()
        input_ids = input_ids[:, :-1]
        labels = labels[:, 1:]
        pack_len = 8192 - input_ids.shape[1]
        input_ids = pad_to_max_length(input_ids, 0, max_length=8192)
        labels = pad_to_max_length(labels, -100, max_length=8192)
        losses = []

        data_mesh = None
        if sp_size > 1:
            data_mesh = init_data_mesh(str(DEVICE), sp_size)

        for _ in range(10):
            seq_ctx = SequenceContext.from_input_ids((input_ids,), device=DEVICE)
            labels = labels.to(DEVICE)
            seq_ctx.num_padding = pack_len
            seq_ctx_list = [seq_ctx]
            loss_ctx_input_list: list[CELossContextInputItem] = [CELossContextInputItem(shifted_labels=labels)]
            LossContext = loss_cfg.loss_ctx_cls
            batches_loss_kwargs = LossContext.build_batches_loss_kwargs(
                loss_ctx_input_list, 
                loss_cfg,
            )
            loss_kwargs = batches_loss_kwargs[0]
            loss_ctx = LossContext(loss_cfg, loss_kwargs)
            seq_ctx = seq_ctx_list[0]
            engine_input = [ModelItem(seq_ctx=seq_ctx, loss_ctx=loss_ctx)]
            loss_log, _ = engine.train_step(engine_input)
            grad_norm = engine.clip_grad_norm()
            engine.step_optimizer(grad_norm)
            lr_scheduler.step()
            losses.append(loss_log["reduced_llm_loss"])

        losses_ref = torch.tensor([2.44, 2.44, 2.42, 2.41, 2.34, 2.33, 2.16, 2.13, 1.71, 1.55])
        losses = torch.tensor(losses)
        self._check_loss_curve(losses, losses_ref)

        torch.cuda.empty_cache()
        try:
            dist.destroy_process_group(pg)
        except:
            pass

    @parametrize.parametrize(
        "device,ep_size,sp_size",
        [
            ("cuda", 1, 1),
        ],
    )
    def test_moe_engine_train_freeze_routers(self, device, ep_size, sp_size):
        pg = self.create_pg(device)

        moe_cfg = Qwen3MoE30BA3Config(
            ep_size=ep_size,
            balancing_loss_cfg=BalancingLossConfig(),
            z_loss_cfg=ZLossConfig(),
            freeze_routers=True,
        )
        optim_cfg: AdamWConfig = AdamWConfig()
        lr_cfg: LRConfig = LRConfig()
        fsdp_cfg: FSDPConfig = FSDPConfig(
            torch_compile=False,
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

        tok = AutoTokenizer.from_pretrained(QWEN3_MOE_PATH)
        txt = "根据国际地球自转和参考系服务机构的数据，今年夏天是自2020年以来第六次地球自转加速。7月9日将成为有史以来最短的一天，比平时短1.3到1.6毫秒。 "
        input_ids = tok.encode(txt, return_tensors="pt").view(1, -1)
        labels = input_ids.clone()
        input_ids = input_ids[:, :-1]
        labels = labels[:, 1:]
        pack_len = 8192 - input_ids.shape[1]
        input_ids = pad_to_max_length(input_ids, 0, max_length=8192)
        labels = pad_to_max_length(labels, -100, max_length=8192)
        losses = []

        data_mesh = None
        if sp_size > 1:
            data_mesh = init_data_mesh(str(DEVICE), sp_size)

        # check the gradient and parameters of the routers
        gate_means = defaultdict(list)
        gate_stds = defaultdict(list)
        for name, layer in engine.model.layers.items():
            if isinstance(layer, MoEDecoderLayer):
                self.assertFalse(layer.gate.weight.requires_grad)
                self.assertTrue(layer.gate.weight.is_leaf)
                gate_means[name].append(layer.gate.weight.full_tensor().mean())
                gate_stds[name].append(layer.gate.weight.full_tensor().std())

        for _ in range(4):
            seq_ctx = SequenceContext.from_input_ids((input_ids,), device=DEVICE)
            labels = labels.to(DEVICE)
            seq_ctx.num_padding = pack_len
            seq_ctx_list = [seq_ctx]
            loss_ctx_input_list: list[CELossContextInputItem] = [CELossContextInputItem(shifted_labels=labels)]
            LossContext = loss_cfg.loss_ctx_cls
            batches_loss_kwargs = LossContext.build_batches_loss_kwargs(
                loss_ctx_input_list, 
                loss_cfg,
            )
            loss_kwargs = batches_loss_kwargs[0]
            loss_ctx = LossContext(loss_cfg, loss_kwargs)
            seq_ctx = seq_ctx_list[0]
            engine_input = [ModelItem(seq_ctx=seq_ctx, loss_ctx=loss_ctx)]
            loss_log, _ = engine.train_step(engine_input)
            grad_norm = engine.clip_grad_norm()
            engine.step_optimizer(grad_norm)
            lr_scheduler.step()
            losses.append(loss_log["reduced_llm_loss"])
            for name, layer in engine.model.layers.items():
                if isinstance(layer, MoEDecoderLayer):
                    assert torch.equal(layer.gate.weight.full_tensor().mean(), gate_means[name][-1]), (
                        f"Mismatch in gate mean in layer {name}, {layer.gate.weight.full_tensor().mean()} and {gate_means[name][-1]}"
                    )
                    assert torch.equal(layer.gate.weight.full_tensor().std(), gate_stds[name][-1]), (
                        f"Mismatch in gate std in layer {name}, {layer.gate.weight.full_tensor().std()} and {gate_stds[name][-1]}"
                    )
                    gate_means[name].append(layer.gate.weight.full_tensor().mean())
                    gate_stds[name].append(layer.gate.weight.full_tensor().std())

        torch.cuda.empty_cache()
        try:
            dist.destroy_process_group(pg)
        except:
            pass

    @parametrize.parametrize(
        "device,ep_size,hsdp_sharding_size",
        [
            ("cuda", 1, 8),  # todo: test ep8 and hsdp, OOM in 8 gpus
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
        )
        optim_cfg: AdamWConfig = AdamWConfig()
        fsdp_cfg: FSDPConfig = FSDPConfig(
            torch_compile=False,
            cpu_offload=False,
            ep_size=ep_size,
            hsdp_sharding_size=hsdp_sharding_size,
        )
        engine = TrainEngine(
            model_cfg=moe_cfg,
            optim_cfg=optim_cfg,
            fsdp_cfg=fsdp_cfg,
        )

        engine.init_model_weights()
        engine.from_hf(hf_path=QWEN3_MOE_PATH)
        engine.save_hf(
            hf_dir=temp_dir,
            save_dtype=torch.bfloat16,
        )

        dist.barrier()
        time.sleep(1)

        engine2 = TrainEngine(
            model_cfg=moe_cfg,
            optim_cfg=optim_cfg,
            fsdp_cfg=fsdp_cfg,
        )
        engine2.from_hf(hf_path=temp_dir)

        state_dict = engine.model.state_dict()
        state_dict2 = engine2.model.state_dict()
        for key, val in state_dict.items():
            val2 = state_dict2[key]
            val = val.full_tensor().bfloat16()
            val2 = val2.full_tensor().bfloat16()
            self.assertTrue(torch.equal(val, val2[:val.shape[0]]),
                            f"Mismatch in {key} between bf16 and fp8, {val} and {val2[:val.shape[0]]}")

        if dist.get_rank() == 0:
            shutil.rmtree(temp_dir)

        torch.cuda.empty_cache()
        try:
            dist.destroy_process_group(pg)
        except:
            pass
    
    @parametrize.parametrize(
        "device",
        [
            ("cuda",),
        ],
    )
    def test_load_optimizer_with_new_lr(self, device):
        pg = self.create_pg(device)

        temp_dir = tempfile.mkdtemp()
        if dist.get_rank() == 0:
            temp_dir = [temp_dir]
        else:
            temp_dir = [None]
        dist.broadcast_object_list(temp_dir, src=0)
        temp_dir = Path(temp_dir[0])
        model_dir = temp_dir / "model"
        optimizer_dir = temp_dir / "optimizer"
        moe_cfg = Qwen3MoE30BA3Config(
            num_hidden_layers=2,
        )
        lr1 = 1e-4
        eps1 = 1e-7
        optim_cfg: AdamWConfig = AdamWConfig(lr=lr1, eps=eps1)
        fsdp_cfg: FSDPConfig = FSDPConfig()
        engine = TrainEngine(
            model_cfg=moe_cfg,
            optim_cfg=optim_cfg,
            fsdp_cfg=fsdp_cfg,
        )
        engine.init_model_weights()
        engine.save_dcp(model_dir=model_dir, optimizer_dir=optimizer_dir)
        dist.barrier()
        time.sleep(1)

        lr2 = 1e-3
        eps2 = 1e-5
        optim_cfg2: AdamWConfig = AdamWConfig(lr=lr2, eps=eps2)
        engine2 = TrainEngine(
            model_cfg=moe_cfg,
            optim_cfg=optim_cfg2,
            fsdp_cfg=fsdp_cfg,
        )
        engine2.load_dcp(model_dir=model_dir, optimizer_dir=optimizer_dir, load_args=False)
        # print(f"len(engine.optimizer.state), len(engine2.optimizer.state): {len(engine.optimizer.state)}, {len(engine2.optimizer.state)}")
        assert len(engine.optimizer.state) == len(engine2.optimizer.state)
        assert len(engine.optimizer.state) != 0
        for param_group in engine2.optimizer.param_groups:
            # print(f"param_group['lr']: {param_group['lr']}")
            assert param_group['lr'] == lr2
            assert param_group['eps'] == eps2
        
        lr3 = 1e-1
        eps3 = 1e-3
        optim_cfg3 = AdamWConfig(lr=lr3, eps=eps3)
        engine3 = TrainEngine(
            model_cfg=moe_cfg,
            optim_cfg=optim_cfg3,
            fsdp_cfg=fsdp_cfg,
        )
        engine3.load_dcp(model_dir=model_dir, optimizer_dir=optimizer_dir, load_states=False)
        assert len(engine3.optimizer.state) == 0
        for param_group in engine3.optimizer.param_groups:
            assert param_group['lr'] == lr1
            assert param_group['eps'] == eps1

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
