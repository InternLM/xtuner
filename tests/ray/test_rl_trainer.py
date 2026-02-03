import os
import tempfile
import unittest
from pathlib import Path

import ray
import torch

from transformers import AutoTokenizer
from xtuner.v1.config import (
    AdamWConfig,
    FSDPConfig,
    LRConfig,
)
from xtuner.v1.data_proto.rl_data import SampleParams
from xtuner.v1.datasets import RLTokenizeFnConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.model import get_model_config_from_hf
from xtuner.v1.ray.base import AcceleratorResourcesConfig, CPUResourcesConfig
from xtuner.v1.ray.config.worker import RolloutConfig
from xtuner.v1.ray.dataflow import DataFlowConfig, ReplayBufferConfig
from xtuner.v1.ray.judger.controller import JudgerConfig
from xtuner.v1.rl.base import WorkerConfig
from xtuner.v1.rl.grpo import GRPOLossConfig
from xtuner.v1.train.rl_trainer import RLTrainer, RLTrainerConfig


MODEL_PATH = os.environ["ROLLOUT_MODEL_PATH"]
TRAIN_DATA_PATH = os.environ["ROLLOUT_DATA_PATH"]
TEST_DATA_PATH = os.environ["ROLLOUT_TEST_DATA_PATH"]
resource_map = {
    "npu": "NPU",
    "cuda": "GPU",
}


class TestRLTrainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["XTUNER_USE_FA3"] = "1"
        os.environ["LMD_SKIP_WARMUP"] = "1"

    @classmethod
    def tearDownClass(cls):
        del os.environ["XTUNER_USE_FA3"]
        del os.environ["LMD_SKIP_WARMUP"]
        
    def init_traine_worker_config(self, train_optimizer_steps, pack_max_length):
        model_cfg = get_model_config_from_hf(Path(MODEL_PATH))
        optim_cfg = AdamWConfig(lr=1e-6, betas=(0.9, 0.999), max_grad_norm=1.0, weight_decay=0.1, foreach=False)
        loss_cfg = GRPOLossConfig(
            policy_loss_cfg=dict(
                cliprange_high=0.28,
                cliprange_low=0.2,
                loss_type="vanilla",
                clip_ratio_c=10.0,
                log_prob_diff_min=-20.0,
                log_prob_diff_max=20.0,
            ),
            ignore_idx=-100,
            use_kl_loss=False,
            kl_loss_coef=0.0,
            kl_loss_type="low_var_kl",
            mode="chunk",
            chunk_size=512,
        )
        lr_cfg = LRConfig(lr_type="constant", warmup_ratio=0, lr_min=1e-6)
        fsdp_cfg = FSDPConfig(torch_compile=False, cpu_offload=False, ep_size=1)
        train_worker_cfg: WorkerConfig = WorkerConfig(
            model_cfg=model_cfg,
            load_from=MODEL_PATH,
            optim_cfg=optim_cfg,
            loss_cfg=loss_cfg,
            lr_cfg=lr_cfg,
            fsdp_cfg=fsdp_cfg,
            sp_size=1,
            optimizer_steps=train_optimizer_steps,
            pack_max_length=pack_max_length,
        )
        return train_worker_cfg

    def init_replay_buffer_config(self, max_prompt_length):
        train_dataset_cfg = [
            {
                "dataset": DatasetConfig(name="gsm8k", anno_path=TRAIN_DATA_PATH, sample_ratio=1.0),
                "tokenize_fn": RLTokenizeFnConfig(max_length=max_prompt_length),
            },
        ]
        dataloader_cfg = DataloaderConfig(
            collator="fake_collator",
            pack_level="none",
            group_by_length=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        replay_buffer_cfg = ReplayBufferConfig(
            dataset_cfg=train_dataset_cfg,
            dataloader_cfg=dataloader_cfg,
            tokenizer=tokenizer,
            worker_log_dir=self.worker_log_dir,
        )
        return replay_buffer_cfg

    def init_resources_config(self, num_workers, num_cpus_per_worker, cpu_memory_per_worker):
        resources = AcceleratorResourcesConfig(
            accelerator=resource_map[torch.accelerator.current_accelerator().type],
            num_workers=num_workers,
            num_cpus_per_worker=num_cpus_per_worker,
            cpu_memory_per_worker=cpu_memory_per_worker,
        )
        return resources

    def init_cpu_resources_config(self, num_cpus_per_worker, cpu_memory_per_worker):
        cpu_resources = CPUResourcesConfig(
            num_cpus_per_worker=num_cpus_per_worker,
            cpu_memory_per_worker=cpu_memory_per_worker,
        )
        return cpu_resources

    def init_rollout_config(self, max_prompt_length, max_response_length):
        rollout_config = RolloutConfig(
            env="test_rl_trainer",
            model_path=MODEL_PATH,
            worker_log_dir=self.worker_log_dir,
            rollout_max_batch_size_per_instance=1024,
            context_length=max_response_length + max_prompt_length,
        )
        return rollout_config

    def init_dataflow_config(self, max_response_length, global_batch_size, prompt_repeat_k, enable_partial_rollout):
        sample_params = SampleParams(
            max_tokens=max_response_length,
        )
        dataflow_config = DataFlowConfig(
            env="test_rl_trainer",
            global_batch_size=global_batch_size,
            prompt_repeat_k=prompt_repeat_k,
            worker_log_dir=self.worker_log_dir,
            sample_params=sample_params,
            enable_partial_rollout=enable_partial_rollout,
            max_concurrent=1024,
        )
        return dataflow_config

    def init_judger_config(self):
        from xtuner.v1.ray.judger.gsm8k import GSM8KJudgerConfig

        gsm8k_judger_config = GSM8KJudgerConfig(judger_name="openai/gsm8k")
        judger_cfg = JudgerConfig(reward_judger_configs=[gsm8k_judger_config], worker_log_dir=self.worker_log_dir)
        return judger_cfg

    def init_multi_judger_config(self):
        from xtuner.v1.ray.judger.gsm8k import GSM8KJudgerConfig

        # 支持一个GSM8KJudgerConfig创建多个实例
        gsm8k_judger_config_1 = GSM8KJudgerConfig(judger_name="openai/gsm8k-1")
        gsm8k_judger_config_2 = GSM8KJudgerConfig(judger_name="openai/gsm8k-2")
        judger_cfg = JudgerConfig(
            reward_judger_configs=[gsm8k_judger_config_1, gsm8k_judger_config_2],
            worker_log_dir=self.worker_log_dir,
        )
        return judger_cfg

    def setUp(self):
        ray.init(num_cpus=80, ignore_reinit_error=True)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.worker_log_dir = os.path.join(self.temp_dir.name, "work_dirs")

        train_optimizer_steps = 2
        pack_max_length = 32768
        max_prompt_length = 2048
        max_response_length = 1024
        global_batch_size = 4
        prompt_repeat_k = 4
        enable_partial_rollout = False

        self.train_worker_cfg = self.init_traine_worker_config(train_optimizer_steps, pack_max_length)
        self.replay_buffer_cfg = self.init_replay_buffer_config(max_prompt_length)
        self.resources_cfg = self.init_resources_config(
            num_workers=8, num_cpus_per_worker=8, cpu_memory_per_worker=8 * 1024**3
        )
        self.cpu_resources = self.init_cpu_resources_config(num_cpus_per_worker=1, cpu_memory_per_worker=1 * 1024**3)
        self.rollout_config = self.init_rollout_config(
            max_response_length=max_response_length, max_prompt_length=max_prompt_length
        )
        self.dataflow_config = self.init_dataflow_config(
            max_response_length=max_response_length,
            global_batch_size=global_batch_size,
            prompt_repeat_k=prompt_repeat_k,
            enable_partial_rollout=enable_partial_rollout,
        )
        self.judger_config = self.init_judger_config()

    def tearDown(self):
        self.temp_dir.cleanup()
        ray.shutdown()

    def test_rl_trainer(self):
        multi_judger_config = self.init_multi_judger_config()
        cpu_resources = self.init_cpu_resources_config(num_cpus_per_worker=2, cpu_memory_per_worker=2 * 1024**3)
        trainer_config = RLTrainerConfig(
            load_from=MODEL_PATH,
            resources=self.resources_cfg,
            cpu_resources=cpu_resources,
            rollout_config=self.rollout_config,
            dataflow_config=self.dataflow_config,
            judger_config=multi_judger_config,
            replay_buffer_config=self.replay_buffer_cfg,
            train_worker_config=self.train_worker_cfg,
            work_dir=self.worker_log_dir,
            tokenizer_path=MODEL_PATH,
            total_epochs=1,
            debug_train=True,
            rollout_steps=1,
        )
        trainer = RLTrainer.from_config(trainer_config)
        self.assertIsNotNone(trainer, "Trainer should be created successfully")
        try:
            trainer.fit()
        except Exception as e:
            self.fail(f"trainer.fit() raised unexpected exception: {e}")
        # assure all writers are closed before checking log files
        del trainer
        log_files = list(Path(self.worker_log_dir).rglob("*.log"))
        self.assertGreater(len(log_files), 0, "Should generate log files")
        trajectory_files = list(Path(self.worker_log_dir).rglob("*_trajectory.jsonl"))
        self.assertGreater(len(trajectory_files), 0, "Should generate trajectory files")

    def test_judger_cpu_pg_creation_with_error(self):
        """Test RLTrainer judger_cpu_pg creation."""
        multi_judger_config = self.init_multi_judger_config()
        # error resource with multi-judger
        cpu_resources = self.init_cpu_resources_config(num_cpus_per_worker=1, cpu_memory_per_worker=1 * 1024**3)
        trainer_config = RLTrainerConfig(
            load_from=MODEL_PATH,
            resources=self.resources_cfg,
            cpu_resources=cpu_resources,
            rollout_config=self.rollout_config,
            dataflow_config=self.dataflow_config,
            judger_config=multi_judger_config,
            replay_buffer_config=self.replay_buffer_cfg,
            train_worker_config=self.train_worker_cfg,
            work_dir=self.worker_log_dir,
            tokenizer_path=MODEL_PATH,
            total_epochs=1,
            rollout_steps=1,
        )
        with self.assertRaises(AssertionError) as cm:
            trainer = RLTrainer.from_config(trainer_config)

        print(f"Expected AssertionError caught: {cm.exception}")

if __name__ == "__main__":
    unittest.main()
