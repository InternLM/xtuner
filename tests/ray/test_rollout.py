import os
import unittest

import ray
from transformers import AutoTokenizer

from xtuner.v1.ray.config.worker import RolloutConfig
from xtuner.v1.ray.judger.controller import JudgerConfig
from xtuner.v1.ray.accelerator import AcceleratorResourcesConfig, AutoAcceleratorWorkers
from xtuner.v1.ray.dataflow import DataFlow, DataFlowConfig, ReplayBufferConfig
from xtuner.v1.ray.environment import SingleTurnEnvironment
from xtuner.v1.ray.rollout import RolloutController
from xtuner.v1.ray.judger import JudgerController
from xtuner.v1.datasets import RLTextTokenizeFnConfig, build_datasets, build_dataloader
from xtuner.v1.config import (
    DataloaderConfig,
    DatasetConfig,
)

TEST_TEXT_MESSAGES=[{"role": "user", "content": "Hello!"}]
MODEL_PATH = os.environ["ROLLOUT_MODEL_PATH"]
TRAIN_DATA_PATH = os.environ["ROLLOUT_DATA_PATH"]
TEST_DATA_PATH = os.environ["ROLLOUT_TEST_DATA_PATH"]


class TestRollout(unittest.TestCase):
    def init_config(self):
        self.resources_cfg = AcceleratorResourcesConfig(
            accelerator="GPU",
            num_workers=8,
            cpu_memory_per_worker=16 * 1024**3,  # 16 GB
        )
        self.max_prompt_length = 512
        self.rollout_cfg = RolloutConfig(
            env="test_rollout",
            model_path=MODEL_PATH,
            model_name=os.path.basename(MODEL_PATH).lower(),
            tokenizer_path=MODEL_PATH,
            rollout_cross_node_comm=False,
            tensor_parallel_size=8,
            expert_parallel_size=1,
            gpus_per_node=8, # gpu: 8, npu: 16
            dtype="bfloat16",
        )
        from xtuner.v1.ray.judger.gsm8k import GSM8KJudgerConfig
        gsm8k_judger_config = GSM8KJudgerConfig()
        self.judger_cfg = JudgerConfig(
            reward_judger_configs={"openai/gsm8k": gsm8k_judger_config}
        )
        self.dataflow_cfg = DataFlowConfig(
            env="test",
            max_concurrent=32,
            prompt_repeat_k=2,
            global_batch_size=2,
            enable_partial_rollout=0
        )
        self.train_dataset_cfg = [
            {
            "dataset": DatasetConfig(name="gsm8k",
                                    anno_path=TRAIN_DATA_PATH,
                                    sample_ratio=1.0),
            "tokenize_fn": RLTextTokenizeFnConfig(max_length=self.max_prompt_length),
            },
        ]
        self.dataloader_cfg = DataloaderConfig(
            pack_max_length=self.max_prompt_length,
            collator='fake_collator',
            pack_level='none',
        )
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        self.replay_buffer_cfg = ReplayBufferConfig(
            dataset_cfg=self.train_dataset_cfg,
            dataloader_cfg=self.dataloader_cfg,
            tokenizer=self.tokenizer,
        )

    def setUp(self):
        ray.init(num_cpus=80, ignore_reinit_error=True)
        self.data_path = TRAIN_DATA_PATH
        self.model_path = MODEL_PATH
        self.init_config()
        self.pg = AutoAcceleratorWorkers.build_placement_group(self.resources_cfg)
        
    def tearDown(self):
        ray.shutdown()

    @unittest.skipIf(os.environ.get("XTUNER_USE_LMDEPLOY", "0") == "0", "lmdeploy backend is not enabled")
    def test_lmdeploy_generate(self):
        from xtuner.v1.ray.rollout import SampleParams, LMDeployWorker
        rollout_workers_map = AutoAcceleratorWorkers.from_placement_group(
            LMDeployWorker, self.rollout_cfg, self.pg
        )
        sample_params = SampleParams(temperature=0.0)
        rollout_controller = RolloutController.remote(self.rollout_cfg, rollout_workers_map)  # type: ignore[attr-defined]
        new_prompt = self.tokenizer.apply_chat_template(TEST_TEXT_MESSAGES, add_generation_prompt=True, tokenize=False)
        ref1 = rollout_controller.rollout.remote(prompt=[new_prompt], sample_params=sample_params)
        ref2 = rollout_controller.rollout.remote(prompt=[new_prompt], sample_params=sample_params)
        results = ray.get([ref1, ref2])
        self.assertEqual(results[0], results[1], f"results[0] != results[1], results[0]={results[0]}, results[1]={results[1]}")
        ray.get(rollout_controller.shutdown.remote(), timeout=300)

    @unittest.skipIf(os.environ.get("XTUNER_USE_LMDEPLOY", "0") == "0", "lmdeploy backend is not enabled")
    def test_lmdeploy_dataflow(self):
        self.dataflow_cfg.enable_partial_rollout = 0
        self.test_env = SingleTurnEnvironment.remote(
            "test_env",
            self.pg,
            rollout_cfg=self.rollout_cfg,
        )
        self.test_flow = DataFlow.remote("test_env", 
                                         self.dataflow_cfg,
                                         self.replay_buffer_cfg,
                                         self.test_env
                                        )
        responses = ray.get(self.test_flow.run.remote(), timeout=300)
        finished_samples_count = sum(1 for data in responses for item in data if item.get("state") == "finished")
        self.assertEqual(finished_samples_count // self.dataflow_cfg.prompt_repeat_k, self.dataflow_cfg.global_batch_size)
        ray.get(self.test_env.shutdown.remote(), timeout=300)
        
    @unittest.skipIf(os.environ.get("XTUNER_USE_LMDEPLOY", "0") == "0", "lmdeploy backend is not enabled")
    def test_lmdeploy_async_dataflow(self):
        self.dataflow_cfg.enable_partial_rollout = 1
        self.test_env = SingleTurnEnvironment.remote(
            "test_env",
            self.pg,
            rollout_cfg=self.rollout_cfg,
        )
        self.test_flow = DataFlow.remote("test_env", 
                                         self.dataflow_cfg,
                                         self.replay_buffer_cfg,
                                         self.test_env
                                        )
        responses = ray.get(self.test_flow.run.remote(), timeout=300)
        finished_samples_count = sum(1 for data in responses for item in data if item.get("state") == "finished")
        self.assertEqual(finished_samples_count // self.dataflow_cfg.prompt_repeat_k, self.dataflow_cfg.global_batch_size)
        ray.get(self.test_env.shutdown.remote())

if __name__ == "__main__":
    unittest.main()