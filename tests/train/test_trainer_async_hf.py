import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.testing._internal.common_distributed import DistributedTestBase

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.datasets import FTDPTokenizeFnConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.model.moe.qwen3 import Qwen3MoE30BA3Config
from xtuner.v1.model.utils.misc import ModelForwardExtraLogInfo
from xtuner.v1.train.trainer import Trainer
from xtuner.v1.utils.device import get_device


DEVICE = get_device()


class FakeHFModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.hf_index_calls = []

    def forward(self, x):
        return self.linear(x)

    def _write_hf_index_and_config(self, hf_dir: Path | str, weight_map: dict[str, str]):
        hf_dir = Path(hf_dir)
        hf_dir.mkdir(parents=True, exist_ok=True)
        self.hf_index_calls.append({"hf_dir": hf_dir, "weight_map": dict(weight_map)})
        (hf_dir / "config.json").write_text('{"model_type": "fake_model"}')
        with (hf_dir / "model.safetensors.index.json").open("w") as f:
            json.dump({"metadata": {"total_size": len(weight_map)}, "weight_map": weight_map}, f)


class FakeAsyncHFEngine:
    def __init__(self):
        self.save_hf_calls = []
        self.wait_async_hf_calls = []
        self.train_step_calls = 0
        self.grad_norm_calls = 0
        self.optimizer_step_calls = 0
        self.async_hf_status_ok = True
        self.async_hf_status_error = ""

        self.model = model = FakeHFModel()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.has_freeze_params = False
        self._pending_async_hf = None

    def grad_accumulation_steps(self, *args, **kwargs):
        return 1

    def train_step(self, *args, **kwargs):
        self.train_step_calls += 1
        return {
            "total_loss": 1.8,
            "step_consumed_tokens": 100,
            "step_consumed_img_tokens": 0.0,
            "grad_norm": torch.tensor(1.0),
            "efficient_attn_ratio": 0.5,
            "img_efficient_attn_ratio": 0.0,
            "logs_info": {"local_loss": 1.0, "reduced_llm_loss": 0.8},
            "extra_info": ModelForwardExtraLogInfo(),
        }

    def step_optimizer(self, *args, **kwargs):
        self.optimizer_step_calls += 1
        return 1.0

    def clip_grad_norm(self, do_clip: bool = True, dtype=torch.float32):
        self.grad_norm_calls += 1
        return torch.tensor(1.0)

    def save_hf(self, hf_dir: Path | str):
        finalized_hf_dir = self.wait_async_hf()
        hf_dir = Path(hf_dir)
        hf_dir.mkdir(parents=True, exist_ok=True)
        self.save_hf_calls.append(hf_dir)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        shard_name = f"model-rank{rank}-of-{world_size}.safetensors"
        (hf_dir / shard_name).write_text(f"fake async model weights for rank {rank}")
        weight_map = {f"layers.rank{rank}.weight": shard_name} if self.async_hf_status_ok else {}
        self._pending_async_hf = {
            "hf_dir": hf_dir,
            "ok": self.async_hf_status_ok,
            "error": self.async_hf_status_error,
            "weight_map": weight_map,
        }
        return finalized_hf_dir

    def wait_async_hf(self):
        self.wait_async_hf_calls.append(None)
        if self._pending_async_hf is None:
            return None

        pending = self._pending_async_hf
        local_status = {
            "rank": dist.get_rank(),
            "ok": bool(pending["ok"]),
            "error": str(pending["error"]),
            "weight_map": pending["weight_map"],
        }
        all_status = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(all_status, local_status)
        if not all(status["ok"] for status in all_status):
            self._pending_async_hf = None
            failed = ", ".join(
                f"rank={status['rank']}({status['error']})" for status in all_status if not status["ok"]
            )
            raise RuntimeError(f"Async HF save global consistency check failed: {failed}")

        merged_weight_map = {}
        for status in all_status:
            merged_weight_map.update(status["weight_map"])

        if dist.get_rank() == 0:
            self.model._write_hf_index_and_config(
                hf_dir=Path(pending["hf_dir"]),
                weight_map=merged_weight_map,
            )
        self._pending_async_hf = None
        return Path(pending["hf_dir"])


def prepare(fn):
    def wrapper(self, *args, **kwargs):
        self.alpaca_path = Path(__file__).resolve().parents[1] / "resource" / "openai_sft.jsonl"
        self.tokenizer_path = None
        self.temp_dir = tempfile.TemporaryDirectory()
        self.fake_hf_model_dir = Path(self.temp_dir.name) / "fake_hf_model"
        self.work_dir = Path(self.temp_dir.name) / "work_dir"

        self.fake_hf_model_dir.mkdir()
        (self.fake_hf_model_dir / "config.json").write_text('{"model_type": "fake_model"}')
        (self.fake_hf_model_dir / "model.safetensors").write_text("fake weights")
        ret = fn(self, *args, **kwargs)
        self.temp_dir.cleanup()
        return ret

    return wrapper


class TestTrainerAsyncSaveHF(DistributedTestBase):
    def create_pg(self, device):
        ret = super().create_pg(device)
        os.environ["LOCAL_RANK"] = str(dist.get_rank())
        return ret

    def _broadcast_work_dir(self):
        work_dir_list = [self.work_dir]
        dist.broadcast_object_list(work_dir_list, src=0)
        self.work_dir = Path(work_dir_list[0])

    def _build_trainer(self, total_step=10, **kwargs):
        trainer_kwargs = dict(
            load_from=str(self.fake_hf_model_dir),
            model_cfg=Qwen3MoE30BA3Config(),
            optim_cfg=AdamWConfig(lr=1e-4, weight_decay=0.01),
            fsdp_cfg=FSDPConfig(tp_size=1),
            dataset_cfg=[
                {
                    "dataset": DatasetConfig(name="alpaca", anno_path=self.alpaca_path, sample_ratio=1.0),
                    "tokenize_fn": FTDPTokenizeFnConfig(),
                },
            ],
            dataloader_cfg=DataloaderConfig(),
            lr_cfg=LRConfig(lr_type="constant", warmup_ratio=0.1, lr_min=1e-6),
            tokenizer_path=self.tokenizer_path,
            global_batch_size=2,
            total_step=total_step,
            work_dir=str(self.work_dir),
            hf_interval=3,
            hf_max_keep=2,
            checkpoint_interval=None,
            snapshot_interval=None,
            seed=42,
            debug=False,
        )
        trainer_kwargs.update(kwargs)
        return Trainer(**trainer_kwargs)

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "2"))

    @patch("xtuner.v1.train.trainer.time.sleep", Mock())
    @patch("xtuner.v1.train.trainer.is_hf_model_path", Mock(return_value=True))
    @patch("xtuner.v1.train.trainer.Trainer._prepare_model_input", Mock(return_value=[]))
    @patch(
        "xtuner.v1.train.trainer.Trainer.build_engine",
        Mock(side_effect=lambda *args, **kwargs: FakeAsyncHFEngine()),
    )
    @prepare
    def test_async_save_hf_interval(self):
        self.create_pg(DEVICE)
        self._broadcast_work_dir()
        trainer = self._build_trainer(async_hf_export=True)
        trainer.fit()
        dist.barrier()

        self.assertEqual(len(trainer._engine.save_hf_calls), 4)
        if dist.get_rank() == 0:
            self.assertEqual(len(trainer._engine.model.hf_index_calls), 4)

            exp_dir = self.work_dir / trainer.exp_dir.name
            hf_dirs = sorted(d.name for d in exp_dir.iterdir() if d.name.startswith("hf-") and d.is_dir())
            self.assertEqual(hf_dirs, ["hf-10", "hf-9", "hf-latest"])

            latest_hf = exp_dir / "hf-latest"
            self.assertTrue(latest_hf.is_symlink())
            self.assertEqual(latest_hf.resolve(), (exp_dir / "hf-10").resolve())

            self.assertEqual(
                [Path(path).name for path in trainer.meta.latest_exp.hf_checkpoint_list],
                ["hf-9", "hf-10"],
            )

            hf10_dir = exp_dir / "hf-10"
            index_path = hf10_dir / "model.safetensors.index.json"
            self.assertTrue(index_path.exists())
            with index_path.open("r") as f:
                index_info = json.load(f)
            weight_map = index_info["weight_map"]
            self.assertEqual(sorted(weight_map.keys()), [f"layers.rank{rank}.weight" for rank in range(self.world_size)])
            self.assertEqual(
                sorted(weight_map.values()),
                [f"model-rank{rank}-of-{self.world_size}.safetensors" for rank in range(self.world_size)],
            )
            for shard_name in weight_map.values():
                self.assertTrue((hf10_dir / shard_name).exists())
        else:
            self.assertEqual(len(trainer._engine.model.hf_index_calls), 0)
        dist.barrier()

    @patch("xtuner.v1.train.trainer.is_hf_model_path", Mock(return_value=True))
    @patch(
        "xtuner.v1.train.trainer.Trainer.build_engine",
        Mock(side_effect=lambda *args, **kwargs: FakeAsyncHFEngine()),
    )
    @prepare
    def test_async_save_hf_raises_on_writer_failure(self):
        self.create_pg(DEVICE)
        self._broadcast_work_dir()
        trainer = self._build_trainer(total_step=3, async_hf_export=True)
        trainer._engine.async_hf_status_ok = False
        trainer._engine.async_hf_status_error = "mock async hf failure"
        trainer._cur_step = 3

        trainer._maybe_save_hf()
        with self.assertRaisesRegex(RuntimeError, "Async HF save global consistency check failed"):
            trainer._engine.wait_async_hf()
        self.assertIsNone(trainer._engine._pending_async_hf)
