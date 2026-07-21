import json
import math
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist

from transformers import AutoTokenizer
from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.config import FSDPConfig
from xtuner.v1.datasets.config import DataloaderConfig
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.train.arguments.arguments import TrainingArguments
from xtuner.v1.train.trainer import HooksConfig, Trainer
from xtuner.v1.utils import IGNORE_INDEX


GLM5_2_TINY_MOE_PATH = Path(os.environ["GLM5_2_TINY_MOE_PATH"])


def _write_tiny_sft_dataset(path: Path) -> None:
    samples = [
        [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Return exactly: GLM52 smoke ok."},
            {
                "role": "assistant",
                "content": "GLM52 smoke ok.",
                "reasoning_content": "The user requested an exact short answer.",
            },
        ],
        {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "add_numbers",
                        "description": "Add two integers.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "integer"},
                                "b": {"type": "integer"},
                            },
                            "required": ["a", "b"],
                        },
                    },
                }
            ],
            "messages": [
                {"role": "system", "content": "Use tools when arithmetic is requested."},
                {"role": "user", "content": "What is 2 plus 3?"},
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "A calculator call gives the exact sum.",
                    "tool_calls": [{"name": "add_numbers", "arguments": {"a": 2, "b": 3}}],
                },
                {"role": "tool", "content": "5"},
                {"role": "assistant", "content": "2 plus 3 equals 5."},
            ],
        },
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Summarize this unavailable image in one sentence."},
                    {"type": "image_url", "image_url": {"url": "file:///tmp/missing.png"}},
                ],
            },
            {"role": "assistant", "content": "<think>Image bytes are unavailable.</think>I cannot inspect the image."},
        ],
    ]
    with path.open("w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")


def _assert_packed_batch_has_aligned_labels(batch) -> None:
    assert len(batch) > 0
    item = batch[0]
    seq_ctx = item["seq_ctx"]
    shifted_labels = item["shifted_labels"]
    assert seq_ctx.input_ids.shape == shifted_labels.shape
    assert shifted_labels.shape[-1] == seq_ctx.position_ids.shape[-1]
    assert (shifted_labels != IGNORE_INDEX).any()
    assert seq_ctx.cu_seq_lens_q[-1].item() == shifted_labels.numel()


@unittest.skipUnless(
    torch.cuda.device_count() >= 8 and GLM5_2_TINY_MOE_PATH.exists(),
    f"requires at least 8 CUDA devices and tiny GLM-5.2 checkpoint at {GLM5_2_TINY_MOE_PATH}",
)
class TestTinyGlm52SFTSmoke(DeterministicDDPTestCase):
    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "8"))

    def test_tiny_glm52_sft_smoke(self):
        self.create_pg("cuda")
        rank = dist.get_rank()

        tmp_holder = [tempfile.mkdtemp() if rank == 0 else None]
        dist.broadcast_object_list(tmp_holder, src=0)
        tmp_dir = Path(tmp_holder[0])
        completed = False

        try:
            dataset_path = tmp_dir / "glm52_sft.jsonl"
            if rank == 0:
                _write_tiny_sft_dataset(dataset_path)
            dist.barrier()

            dataloader_cfg = DataloaderConfig(
                pack_level="soft",
                pack_max_length=256,
                pack_chunk_size=4,
                pack_workers=1,
                global_pack=False,
                num_workers=0,
            )
            args = TrainingArguments(
                load_from=str(GLM5_2_TINY_MOE_PATH),
                tokenizer_path=GLM5_2_TINY_MOE_PATH,
                chat_template="glm5.2",
                tokenize_fn="openai",
                dataset=dataset_path,
                dataloader_cfg=dataloader_cfg,
                cache_dir=tmp_dir / "cache",
                max_length=256,
                lr=1e-5,
                lr_min=1e-5,
                scheduler_type="constant",
                warmup_ratio=0.0,
                loss_config=CELossConfig(mode="eager"),
                total_step=3,
                work_dir=tmp_dir / "work_dir",
                global_batch_size=self.world_size,
                fsdp_config=FSDPConfig(ep_size=1, cpu_offload=False, torch_compile=False),
            )
            trainer_cfg = args.to_trainer_config()
            # Keep this smoke focused on SFT tokenization, packing, and training.
            # Dedicated MTP tests cover MTP, while this checkpoint also carries
            # MTP-only weights that are intentionally ignored here.
            trainer_cfg.model_cfg.mtp_config = None
            trainer_cfg.strict_load = False
            assert trainer_cfg.model_cfg.mtp_config is None
            trainer_cfg.dataloader_cfg.dataset_config_list = trainer_cfg.dataset_cfg

            # Preview the same public dataloader config before training so the smoke covers
            # GLM-5.2 chat tokenization, packing, and shifted-label alignment explicitly.
            tokenizer = AutoTokenizer.from_pretrained(GLM5_2_TINY_MOE_PATH, trust_remote_code=True)
            preview_dataloader = trainer_cfg.dataloader_cfg.build(
                tokenizer=tokenizer,
                dp_mesh=None,
                global_batch_size=1,
                micro_batch_size=1,
                seed=trainer_cfg.seed,
                total_step=trainer_cfg.total_step,
            )
            _assert_packed_batch_has_aligned_labels(next(iter(preview_dataloader)))

            losses: list[float] = []

            def record_loss(train_step_info, step, epoch, total_step, total_epoch) -> None:
                loss = float(train_step_info["logs_info"]["reduced_llm_loss"])
                assert math.isfinite(loss), f"loss must be finite, got {loss}"
                losses.append(loss)

            trainer_cfg.hooks_config = HooksConfig(after_train_step=record_loss)
            trainer_cfg.checkpoint_interval = None
            trainer_cfg.hf_interval = 3
            trainer_cfg.hf_max_keep = 1
            trainer_cfg.profile_time = False

            trainer = Trainer.from_config(trainer_cfg)
            trainer.fit()

            assert trainer.cur_step == 3
            assert len(losses) == 3
            assert losses[-1] <= losses[0], losses
            assert trainer._lr_scheduler.last_epoch == 3
            assert len(trainer.meta.latest_exp.hf_checkpoint_list) == 1

            if rank == 0:
                hf_checkpoint = Path(trainer.meta.latest_exp.hf_checkpoint_list[0])
                assert (hf_checkpoint / "config.json").exists()
                assert (trainer.exp_dir / "hf-latest").exists()
                assert (trainer.exp_dir / "logs" / "exp_tracking" / "rank0" / "tracker.jsonl").exists()
            dist.barrier()
            completed = True
        finally:
            if completed and rank == 0:
                shutil.rmtree(tmp_dir, ignore_errors=True)
