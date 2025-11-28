import os
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock
import pickle

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.testing._internal.common_distributed import DistributedTestBase

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.datasets.config import DatasetConfig, DataloaderConfig
from xtuner.v1.model.moe.qwen3 import Qwen3MoE30BA3Config, Qwen3MoE235BA22Config
from xtuner.v1.model.dense.qwen3 import Qwen3Dense4BConfig, Qwen3Dense8BConfig
from xtuner.v1.model.compose.intern_s1 import InternS1Config, InternS1MiniConfig
from xtuner.v1.model.compose.internvl import (
    InternVL3P5Dense8BConfig,
    InternVL3P5MoE30BA3Config,
)
from xtuner.v1.train.trainer import Trainer, LoadCheckpointConfig
from xtuner.v1.datasets import FTDPTokenizeFnConfig
from xtuner.v1.datasets.sft_tokenize_fn import OpenaiTokenizeFunctionConfig
from xtuner.v1.train.trainer import TrainerConfig
from xtuner.v1.loss import CELossConfig
from unittest import TestCase
from xtuner.v1.train.trainer import XTunerMeta, ExpInfo, ExpHistory, GitInfo
from xtuner.v1.utils.device import get_device


DEVICE = get_device()


class FakeEngine:
    def __init__(self):
        self.save_hf_calls = []
        self.train_step_calls = 0
        self.grad_norm_calls = 0
        self.optimizer_step_calls = 0

        self.model = model = nn.Linear(10, 10)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.has_freeze_params = False

    def grad_accumulation_steps(self, *args, **kwargs):
        return 1

    def train_step(self, *args, **kwargs):
        self.train_step_calls += 1
        return {"total_loss": 1.0, "reduced_llm_loss": 0.8}, {"consumed_tokens": 100, "grad_norm": torch.tensor(1.0), "efficient_attn_ratio": 0.5}

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

    def clip_grad_norm(self, do_clip: bool=True, dtype=torch.float32):
        self.grad_norm_calls += 1
        return torch.tensor(1.0)

    load_dcp = Mock()

    def save_dcp(self, model_dir: Path, optimizer_dir: Path | None):
        model_dir.mkdir(parents=True, exist_ok=True)
        if optimizer_dir is not None:
            optimizer_dir.mkdir(parents=True, exist_ok=True)


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

        # Should only have 3 directories: hf-9, hf-10, hf-latest left due to max_keep=2
        self.assertEqual(len(hf_dirs), 3)

        # Should have the last 2 checkpoints: hf-9 and hf-10, and hf-latest
        expected_dirs = {"hf-9", "hf-10", "hf-latest"}
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

    @patch("xtuner.v1.train.trainer.is_hf_model_path", Mock(return_value=True))
    @prepare
    def test_save_checkpoint_interval(self):
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

        # 1. Only save checkpoint at last step
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
            debug=False,
            checkpoint_interval=5,
        )

        trainer.fit()
        dist.barrier()
        assert len(trainer.meta.latest_exp.checkpoint_list) == 2
        for checkpoint, step in zip(trainer.meta.latest_exp.checkpoint_list, [5, 10]):
            assert f"step-{step}" in str(checkpoint)
            assert os.path.exists(checkpoint)

        # save checkpoint at step 3 6 9 10 
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
            debug=False,
            checkpoint_interval=3,
        )

        trainer.fit()
        dist.barrier()
        assert len(trainer.meta.latest_exp.checkpoint_list) == 4
        for checkpoint, step in zip(trainer.meta.latest_exp.checkpoint_list, [3, 6, 9, 10]):
            assert f"step-{step}" in str(checkpoint)
            assert os.path.exists(checkpoint)

    @patch("xtuner.v1.train.trainer.is_hf_model_path", Mock(return_value=True))
    @prepare
    def test_resume(self):
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

        # 1. Only save checkpoint at last step
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
            total_step=6,
            work_dir=str(self.work_dir),
            hf_interval=3,
            hf_max_keep=2,
            seed=42,
            debug=False,
            checkpoint_interval=2,
            checkpoint_maxkeep=2,
        )

        trainer.fit()
        dist.barrier()
        # 0. Test checkpoint_maxkeep is consistent with meta file
        assert len(trainer.meta.latest_exp.checkpoint_list) == 2

        # Test resume
        # TODO: It's hard to test the accuracy of resuming in unit test now, need to improve
        # 1. Test auto_resume
        resume_trainer1 = Trainer(
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
            debug=False,
            checkpoint_interval=2,
            checkpoint_maxkeep=2,
            auto_resume=True,
        )
        assert resume_trainer1.cur_step == 6
        assert resume_trainer1.exp_dir == trainer.exp_dir
        resume_trainer1.fit()
        dist.barrier()

        # 1.1 auto_resume twice
        resume_trainer1_2 = Trainer(
            load_from=str(self.fake_hf_model_dir),
            model_cfg=model_cfg,
            optim_cfg=optim_cfg,
            fsdp_cfg=fsdp_cfg,
            dataset_cfg=dataset_cfg,
            dataloader_cfg=dataloader_cfg,
            lr_cfg=lr_cfg,
            tokenizer_path=self.tokenizer_path,
            global_batch_size=2,
            total_step=16,
            work_dir=str(self.work_dir),
            hf_interval=3,
            hf_max_keep=2,
            seed=42,
            debug=False,
            checkpoint_interval=2,
            checkpoint_maxkeep=2,
            auto_resume=True,
        )
        assert resume_trainer1_2.cur_step == 10
        assert resume_trainer1_2.exp_dir == trainer.exp_dir
        resume_trainer1_2.fit()
        assert resume_trainer1_2.cur_step == 16
        dist.barrier()

        # 2. Test resume_from 
        resume_trainer2 = Trainer(
            load_from=str(self.fake_hf_model_dir),
            model_cfg=model_cfg,
            optim_cfg=optim_cfg,
            fsdp_cfg=fsdp_cfg,
            dataset_cfg=dataset_cfg,
            dataloader_cfg=dataloader_cfg,
            lr_cfg=lr_cfg,
            tokenizer_path=self.tokenizer_path,
            global_batch_size=2,
            total_step=20,
            work_dir=str(self.work_dir),
            hf_interval=3,
            hf_max_keep=2,
            seed=42,
            debug=False,
            checkpoint_interval=5,
            checkpoint_maxkeep=2,
            load_checkpoint_cfg=LoadCheckpointConfig(
                checkpoint_path=resume_trainer1_2.meta.latest_exp.checkpoint_list[-2],
            ),
        )
        assert resume_trainer2.cur_step == 14
        resume_trainer2.fit()
        assert resume_trainer2.cur_step == 20
        assert resume_trainer2.exp_dir != resume_trainer1_2.exp_dir

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "2"))


class TestTrainerConfig(TestCase):
    def setUp(self):
        self.dataset_config = [
            {
                "dataset": DatasetConfig(name="alpaca", anno_path="", sample_ratio=1.0),
                "tokenize_fn": OpenaiTokenizeFunctionConfig(
                    max_length=16386, chat_template="qwen3"
                ),
                # "tokenize_fn": FTDPTokenizeFnConfig(max_length=16386),
            },
        ]
        self.dataloader_config = DataloaderConfig(pack_max_length=100)

        self.optim_cfg = AdamWConfig(lr=0.1, weight_decay=0.1)
        self.lr_cfg = LRConfig(lr_type="cosine", lr_min=0.001, warmup_ratio=0.03)

        self.fsdp_cfg = FSDPConfig(torch_compile=True)

    def build_trainer(self, model_cfg):
        return TrainerConfig(
            model_cfg=model_cfg,
            optim_cfg=self.optim_cfg,
            dataset_cfg=self.dataset_config,
            dataloader_cfg=self.dataloader_config,
            lr_cfg=self.lr_cfg,
            loss_cfg=CELossConfig(mode="chunk", chunk_size=1024),
            global_batch_size=1,
            sp_size=1,
            total_epoch=10,
            load_from="",
            seed=42,
            checkpoint_interval=1,
            tokenizer_path="",
            work_dir="",
        )

    def test_dump_trainer_config(self):
        model_cfg_list = [
            Qwen3Dense4BConfig(),
            Qwen3Dense8BConfig(),
            Qwen3MoE30BA3Config(),
            Qwen3MoE235BA22Config(),
            InternS1MiniConfig(),
            InternS1Config(),
            InternVL3P5Dense8BConfig(),
            InternVL3P5MoE30BA3Config(),
        ]

        for model_cfg in model_cfg_list:
            trainer_cfg = self.build_trainer(model_cfg)
            self._dump_trainer_config(trainer_cfg)

    def _dump_trainer_config(self, trainer_cfg: TrainerConfig):
        trainer_cfg.model_dump_json()
        trainer_cfg.model_dump()
        pickle.dumps(trainer_cfg)


def test_resume_and_load_checkpoint_cfg(tmp_path: Path):
    # 0. prepare environment
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    alpaca_path = os.environ["ALPACA_PATH"]
    tokenizer_path = os.environ["QWEN3_MOE_PATH"]

    work_dir = tmp_path / "work_dir"
    fake_hf_model_dir = tmp_path / "fake_hf_model"
    fake_hf_model_dir.mkdir()
    (fake_hf_model_dir / "config.json").write_text('{"model_type": "fake_model"}')
    (fake_hf_model_dir / "model.safetensors").write_text("fake weights")

    model_cfg = Qwen3MoE30BA3Config()
    optim_cfg = AdamWConfig(lr=1e-4, weight_decay=0.01)

    dataset_cfg = [
        {
            "dataset": DatasetConfig(name="alpaca", anno_path=alpaca_path, sample_ratio=1.0),
            "tokenize_fn": FTDPTokenizeFnConfig(),
        },
    ]
    dataloader_cfg = DataloaderConfig(dataset_config_list=dataset_cfg)
    lr_cfg = LRConfig(lr_type="constant", warmup_ratio=0.1, lr_min=1e-6)

    Trainer.build_engine = Mock(return_value=FakeEngine())
    Trainer._resume = Mock()

    # 1. create: first train with auto_resume and load_checkpoint_cfg
    auto_resume = True
    load_checkpoint_cfg = LoadCheckpointConfig(
        checkpoint_path=work_dir / "other_checkpoint",
        load_optimizer_states=True,
        load_optimizer_args=True,
        load_dataset=False,
        load_scheduler=False,
    )

    # 2. operate
    trainer = Trainer(
        load_from=fake_hf_model_dir,
        model_cfg=model_cfg,
        optim_cfg=optim_cfg,
        dataloader_cfg=dataloader_cfg,
        lr_cfg=lr_cfg,
        tokenizer_path=tokenizer_path,
        global_batch_size=2,
        total_step=10,
        checkpoint_interval=5,
        work_dir=work_dir,
        auto_resume=auto_resume,
        load_checkpoint_cfg=load_checkpoint_cfg,
    )

    # 3. check: auto_resume does not overwrite load_checkpoint_cfg at first time
    assert trainer._load_checkpoint_cfg.load_dataset is False
    assert trainer._load_checkpoint_cfg.load_scheduler is False

    del trainer

    # 4. 2nd create: resume train with auto_resume and load_checkpoint_cfg
    exp_dir = work_dir / "fake_exp_dir"
    latest_checkpoint = exp_dir / "step-5"
    resume_meta = XTunerMeta(
        exps=[
            ExpInfo(
                checkpoint_list=[str(latest_checkpoint)],
                exp_dir=str(exp_dir),
                history=[
                    dict(
                        begin=0,
                        timestamp="20251126202933",
                        git_info=dict(commit="dae707", staged="", unstaged=""),
                        end=5,
                    ),
                ],
            ),
        ],
    )
    Trainer._init_xtuner_meta = Mock(return_value=resume_meta)
    trainer2 = Trainer(
        load_from=str(fake_hf_model_dir),
        model_cfg=model_cfg,
        optim_cfg=optim_cfg,
        dataloader_cfg=dataloader_cfg,
        lr_cfg=lr_cfg,
        tokenizer_path=tokenizer_path,
        global_batch_size=2,
        total_step=10,
        checkpoint_interval=5,
        work_dir=work_dir,
        auto_resume=auto_resume,
        load_checkpoint_cfg=load_checkpoint_cfg,
    )

    # 5. check: auto_resume overrides load_checkpoint_cfg when resume train
    assert trainer2._load_checkpoint_cfg.checkpoint_path == latest_checkpoint
    assert trainer2._load_checkpoint_cfg.load_dataset is True
    assert trainer2._load_checkpoint_cfg.load_scheduler is True
    assert trainer2._load_checkpoint_cfg.load_optimizer_states is True
    assert trainer2._load_checkpoint_cfg.load_optimizer_args is True

    dist.destroy_process_group()