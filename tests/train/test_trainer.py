import os
from os.path import samefile
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.testing._internal.common_distributed import DistributedTestBase

from xtuner.v1.config import AdamWConfig, FSDPConfig, DataloaderConfig, LRConfig, DatasetConfig
from xtuner.v1.model.moe.qwen3 import Qwen3MoE30BA3Config
from xtuner.v1.train.trainer import Trainer
from xtuner.v1.datasets import FTDPTokenizeFnConfig

from xtuner.v1.utils.device import get_device


DEVICE = get_device()


class FakeEngine:
    def __init__(self):
        self.save_hf_calls = []
        self.train_step_calls = 0
        self.grad_norm_calls = 0
        self.optimizer_step_calls = 0

        model = nn.Linear(10, 10)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def grad_accumulation_steps(self, *args, **kwargs):
        return 1

    def train_step(self, *args, **kwargs):
        self.train_step_calls += 1
        return {"total_loss": 1.0, "reduced_llm_loss": 0.8}, {"consumed_tokens": 100, "grad_norm": torch.Tensor(1.0)}

    def save_hf(self, hf_path):
        self.save_hf_calls.append(hf_path)
        # Actually create directories and files to simulate real saving
        os.makedirs(hf_path, exist_ok=True)
        # Create a fake model file
        with open(os.path.join(hf_path, "model.safetensors"), "w") as f:
            f.write("fake model weights")
        # Create a fake config file
        with open(os.path.join(hf_path, "config.json"), "w") as f:
            f.write('{"model_type": "fake_model"}')

    def step_optimizer(self, *args, **kwargs):
        self.optimizer_step_calls += 1
        return 1.0

    def clip_grad_norm(self):
        self.grad_norm_calls += 1
        return 1.0


def prepare(fn):
    def wrapper(self, *args, **kwargs):
        self.alpaca_path = os.environ["ALPACA_PATH"]
        self.tokenizer_path = os.environ["QWEN3_MOE_PATH"]
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


class TestTrainerSaveHF(DistributedTestBase):
    def create_pg(self, device):
        ret = super().create_pg(device)
        os.environ["LOCAL_RANK"] = str(dist.get_rank())
        return ret

    @patch("xtuner.v1.train.trainer.is_hf_model_path", Mock(return_value=True))
    @prepare
    def test_save_hf_interval(self):
        """Test save_hf is called at correct intervals during training."""
        self.create_pg(DEVICE)
        work_dir_list = [self.work_dir]
        dist.broadcast_object_list(work_dir_list, src=0)
        self.work_dir = Path(work_dir_list[0])
        model_cfg = Qwen3MoE30BA3Config()
        optim_cfg = AdamWConfig(lr=1e-4, weight_decay=0.01)
        fsdp_cfg = FSDPConfig(tp_size=1)
        dataset_cfg = [
            {
                "dataset": DatasetConfig(name="alpaca", anno_path=self.alpaca_path, sample_ratio=1.0),
                "tokenize_fn": FTDPTokenizeFnConfig(),
            },
        ]
        dataloader_cfg = DataloaderConfig()
        lr_cfg = LRConfig(lr_type="constant", warmup_ratio=0.1, lr_min=1e-6)

        Trainer.build_engine = Mock(return_value=FakeEngine())
        trainer = Trainer(
            load_from=str(self.fake_hf_model_dir),
            model_cfg=model_cfg,
            optim_cfg=optim_cfg,
            fsdp_cfg=fsdp_cfg,
            dataset_cfg=dataset_cfg,
            dataloader_cfg=dataloader_cfg,
            lr_cfg=lr_cfg,
            tokenizer_path=self.tokenizer_path,
            global_batch_size=2,
            total_step=10,
            work_dir=str(self.work_dir),
            hf_interval=3,
            hf_max_keep=2,
            seed=42,
            debug=False
        )

        # Run training
        trainer.fit()
        dist.barrier()

        # Verify save_hf was called at expected intervals
        expected_saves = [3, 6, 9, 10]  # steps 3, 6, 9, 10
        self.assertEqual(len(trainer._engine.save_hf_calls), 4)

        for i, step in enumerate(expected_saves):
            expected_path = str(self.work_dir / trainer.exp_dir.name / f"hf-{step}")
            self.assertEqual(trainer._engine.save_hf_calls[i], expected_path)

        # Verify max_keep logic by checking filesystem - should only keep last 2
        exp_dir = self.work_dir / trainer.exp_dir.name
        hf_dirs = [d for d in exp_dir.iterdir() if d.name.startswith("hf-") and d.is_dir()]

        # Should only have 2 directories left due to max_keep=2
        self.assertEqual(len(hf_dirs), 2)

        # Should have the last 2 checkpoints: hf-9 and hf-10
        expected_dirs = {"hf-9", "hf-10"}
        actual_dirs = {d.name for d in hf_dirs}
        self.assertEqual(actual_dirs, expected_dirs)

        # Verify the files were actually created and contain expected content
        for hf_dir in hf_dirs:
            model_file = hf_dir / "model.safetensors"
            config_file = hf_dir / "config.json"
            self.assertTrue(model_file.exists())
            self.assertTrue(config_file.exists())
            self.assertEqual(model_file.read_text(), "fake model weights")
            self.assertEqual(config_file.read_text(), '{"model_type": "fake_model"}')

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "2"))
