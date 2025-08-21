import os
import torch
import json
import time
import unittest
from transformers import AutoTokenizer

import ray
from xtuner.v1.ray.config.worker import RolloutConfig
from xtuner.v1.ray.judger.controller import JudgerConfig
from xtuner.v1.ray.accelerator import AcceleratorResourcesConfig, AutoAcceleratorWorkers
from xtuner.v1.ray.dataflow import DataFlow, DataFlowConfig
from xtuner.v1.ray.environment import EnvController
from xtuner.v1.datasets import RLTextTokenizeFnConfig, build_datasets, build_dataloader
from xtuner.v1.config import (
    DataloaderConfig,
    DatasetConfig,
)


MODEL_PATH = os.environ["ROLLOUT_MODEL_PATH"]
DATA_PATH = os.environ["ROLLOUT_DATA_PATH"]


class TestRollout(unittest.TestCase):
    def init_config(self):
        self.resources_cfg = AcceleratorResourcesConfig(
            accelerator="GPU",
            num_workers=8,
            cpu_memory_per_worker=16 * 1024**3,  # 16 GB
        )
        self.rollout_cfg = RolloutConfig(
            env="test_rollout",
            model_path=MODEL_PATH,
            model_name=os.path.basename(MODEL_PATH).lower(),
            tokenizer_path=MODEL_PATH,
            rollout_cross_node_comm=False,
            max_running_requests=16,
            tensor_parallel_size=8,
            expert_parallel_size=1,
            gpus_per_node=8, # gpu: 8, npu: 16
            dtype="bfloat16",
        )
        from xtuner.v1.ray.judger.gsm8k import compute_reward
        self.judger_cfg = JudgerConfig(
            reward_functions={"math": compute_reward},
            extra_info={"math": {"score": 1, "format_score": 0.5}},
            reward_ratio={"math": 1.0}
        )
        self.dataflow_cfg = DataFlowConfig(
            env="test",
            max_concurrent=32,
            prompt_repeat_k=8,
            global_batch_size=4,
            enable_partial_rollout=0
        )
        self.dataset_cfg = [
            {
            "dataset": DatasetConfig(name="gsm8k",
                                    anno_path=DATA_PATH,
                                    sample_ratio=1.0),
            "tokenize_fn": RLTextTokenizeFnConfig(max_length=16386),
            },
        ]
        self.dataloader_cfg = DataloaderConfig(
            pack_max_length=16384,
            collator='fake_collator',
            pack_level='none',
        )
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    def setUp(self):
        ray.init(num_cpus=80)
        self.data_path = DATA_PATH
        self.model_path = MODEL_PATH
        self.init_config()
        self.pg = AutoAcceleratorWorkers.build_placement_group(self.resources_cfg)
        self.datasets = build_datasets(self.dataset_cfg, self.tokenizer)
        self.dataloader = build_dataloader(
            dataloader_config=self.dataloader_cfg,
            datasets=self.datasets,
            global_batch_size=1,
            micro_batch_size=1,
            seed=1,
        )
        self.test_env = None
        
    def tearDown(self):
        ray.shutdown()

    @unittest.skipIf(os.environ.get("XTUNER_USE_LMDEPLOY", "0") == "0", "lmdeploy backend is not enabled")
    def test_lmdeploy_backend(self):
        self.dataflow_cfg.enable_partial_rollout = 0
        self.test_env = EnvController.remote(
            "test_env",
            self.pg,
            self.rollout_cfg,
            self.judger_cfg
        )
        self.test_flow = DataFlow.remote("test_env", 
                                    self.dataflow_cfg,
                                    self.datasets,
                                    self.dataloader,
                                    self.tokenizer,
                                    self.test_env
                                    )
        responses = ray.get(self.test_flow.run.remote())
        dataflow_state = ray.get(self.test_flow.state.remote())
        self.assertEqual(len(responses), self.dataflow_cfg.global_batch_size)
        ray.get(self.test_flow.shutdown.remote())
        
    @unittest.skipIf(os.environ.get("XTUNER_USE_LMDEPLOY", "0") == "0", "lmdeploy backend is not enabled")
    def test_lmdeploy_async_backend(self):
        self.dataflow_cfg.enable_partial_rollout = 1
        self.test_env = EnvController.remote(
            "test_env",
            self.pg,
            self.rollout_cfg,
            self.judger_cfg
        )
        self.test_flow = DataFlow.remote("test_env", 
                                    self.dataflow_cfg,
                                    self.datasets,
                                    self.dataloader,
                                    self.tokenizer,
                                    self.test_env
                                    )
        responses = ray.get(self.test_flow.run.remote())
        dataflow_state = ray.get(self.test_flow.state.remote())
        self.assertEqual(len(responses), self.dataflow_cfg.global_batch_size)
        ray.get(self.test_flow.shutdown.remote())
        
    
if __name__ == "__main__":
    unittest.main()