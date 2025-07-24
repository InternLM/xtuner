import os
import tempfile
import shutil
import time
import copy

import parametrize
import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import DistributedTestBase
from transformers import AutoTokenizer

from xtuner.v1.model.moe.moe import SequenceContext
from xtuner.v1.model.moe.qwen3 import Qwen3MoE30BA3Config
from xtuner.v1.config import FSDPConfig, LRConfig, AdamWConfig, BalancingLossConfig, ZLossConfig
from xtuner.v1.engine.moe_train_engine import MoETrainEngine
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR
from xtuner.v1.utils import pad_to_max_length


# Qwen3 30B A3
QWEN3_MOE_PATH = os.environ["QWEN3_MOE_PATH"]


class TestMoEEngine(DistributedTestBase):
    @parametrize.parametrize(
        "device,ep_size,hsdp_sharding_size",
        [
            ("cuda", 1, 8),  # todo: test ep8 and hsdp, OOM in 8 gpus
        ],
    )
    def test_moe_engine_train(self, device, ep_size, hsdp_sharding_size):
        pg = self.create_pg(device)

        moe_cfg = Qwen3MoE30BA3Config(
            ep_size=ep_size,
            balancing_loss_cfg=BalancingLossConfig(),
            z_loss_cfg=ZLossConfig(),
        )
        optim_cfg: AdamWConfig = AdamWConfig()
        lr_cfg: LRConfig = LRConfig()
        fsdp_cfg: FSDPConfig = FSDPConfig(
            torch_compile=True,
            cpu_offload=False,
            ep_size=ep_size,
            max_length=8192,
            hsdp_sharding_size=hsdp_sharding_size,
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
            "/cpfs01/shared/llm_ddd/opencompass/models/hf_hub/models--Qwen--Qwen3-30B-A3B/snapshots/4c446470ba0aec43e22ac1128f9ffd915f338ba3/"
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
            seq_ctx = SequenceContext.from_input_ids((input_ids,))
            seq_ctx.num_padding = pack_len
            loss_log, _ = engine.train_step([{"seq_ctx": seq_ctx, "labels": labels}])
            grad_norm = engine.clip_grad_norm()
            engine.step_optimizer(grad_norm)
            lr_scheduler.step()
            losses.append(loss_log["reduced_llm_loss"])
        losses_ref = [2.44, 2.44, 2.42, 2.41, 2.34, 2.33, 2.16, 2.13, 1.71, 1.55]
        for loss, loss_ref in zip(losses, losses_ref):
            self.assertTrue(abs(loss - loss_ref) < 0.05)
        
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
            torch_compile=True,
            cpu_offload=False,
            ep_size=ep_size,
            max_length=8192,
            hsdp_sharding_size=hsdp_sharding_size,
        )
        engine = MoETrainEngine(
            model_cfg=moe_cfg,
            optim_cfg=optim_cfg,
            fsdp_cfg=fsdp_cfg,
        )

        engine.init_model()
        engine.save_hf(
            hf_dir=temp_dir,
            save_dtype=torch.bfloat16,
        )

        dist.barrier()
        time.sleep(1)

        engine2 = MoETrainEngine(
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
            self.assertTrue(torch.equal(val, val2[:val.shape[0]]), f"Mismatch in {key} between bf16 and fp8, {val} and {val2[:val.shape[0]]}")

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
