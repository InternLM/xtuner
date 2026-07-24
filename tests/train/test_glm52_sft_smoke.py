"""GLM-5.2 SFT 端到端训练行为测试。

TestTinyGlm52SFT
    test_one_step_sft_produces_finite_loss: OpenAI 数据经分词、packing 与 Trainer 后完成一步训练。
"""

import json
import math
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.config import FSDPConfig
from xtuner.v1.datasets.config import DataloaderConfig
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.train.arguments.arguments import TrainingArguments
from xtuner.v1.train.trainer import HooksConfig, Trainer


GLM5_2_TINY_MOE_PATH = Path(os.environ["GLM5_2_TINY_MOE_PATH"])


def _write_sft_sample(path: Path) -> None:
    sample = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Return exactly: GLM52 smoke ok."},
        {
            "role": "assistant",
            "reasoning_content": "The user requested an exact short answer.",
            "content": "GLM52 smoke ok.",
        },
    ]
    path.write_text(json.dumps(sample) + "\n")


@unittest.skipUnless(
    torch.cuda.device_count() >= 8 and GLM5_2_TINY_MOE_PATH.exists(),
    f"requires 8 CUDA devices and GLM-5.2 checkpoint at {GLM5_2_TINY_MOE_PATH}",
)
class TestTinyGlm52SFT(DeterministicDDPTestCase):
    def test_one_step_sft_produces_finite_loss(self):
        # 验证真实 OpenAI 样本通过公共 Trainer 路径完成一步 SFT 并返回有限 loss。
        self.create_pg("cuda")
        rank = dist.get_rank()
        tmp_holder = [tempfile.mkdtemp() if rank == 0 else None]
        dist.broadcast_object_list(tmp_holder, src=0)
        tmp_dir = Path(tmp_holder[0])

        try:
            dataset_path = tmp_dir / "glm52_sft.jsonl"
            if rank == 0:
                _write_sft_sample(dataset_path)
            dist.barrier()

            args = TrainingArguments(
                load_from=str(GLM5_2_TINY_MOE_PATH),
                tokenizer_path=GLM5_2_TINY_MOE_PATH,
                chat_template="glm5.2",
                tokenize_fn="openai",
                dataset=dataset_path,
                dataloader_cfg=DataloaderConfig(
                    pack_level="soft",
                    pack_max_length=256,
                    pack_chunk_size=1,
                    pack_workers=1,
                    global_pack=False,
                    num_workers=0,
                ),
                cache_dir=tmp_dir / "cache",
                max_length=256,
                lr=1e-5,
                lr_min=1e-5,
                scheduler_type="constant",
                warmup_ratio=0.0,
                loss_config=CELossConfig(mode="eager"),
                total_step=1,
                work_dir=tmp_dir / "work_dir",
                global_batch_size=self.world_size,
                fsdp_config=FSDPConfig(ep_size=1, cpu_offload=False, torch_compile=False),
            )
            trainer_cfg = args.to_trainer_config()
            trainer_cfg.model_cfg.mtp_config = None
            trainer_cfg.strict_load = False
            trainer_cfg.dataloader_cfg.dataset_config_list = trainer_cfg.dataset_cfg
            losses = []

            def record_loss(train_step_info, step, epoch, total_step, total_epoch) -> None:
                losses.append(float(train_step_info["logs_info"]["reduced_llm_loss"]))

            trainer_cfg.hooks_config = HooksConfig(after_train_step=record_loss)
            trainer_cfg.checkpoint_interval = None
            trainer_cfg.hf_interval = None
            trainer_cfg.profile_time = False

            trainer = Trainer.from_config(trainer_cfg)
            trainer.fit()

            assert trainer.cur_step == 1
            assert len(losses) == 1
            assert math.isfinite(losses[0])
            dist.barrier()
        finally:
            if rank == 0:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    @property
    def world_size(self) -> int:
        return 8
