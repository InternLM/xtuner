import os
import unittest
import ray
from transformers import AutoTokenizer
import torch
from xtuner.v1.ray.config.worker import RolloutConfig
from xtuner.v1.ray.judger.controller import JudgerConfig
from xtuner.v1.ray.base import AcceleratorResourcesConfig, AutoAcceleratorWorkers
from xtuner.v1.ray.dataflow import DataFlow, DataFlowConfig, ReplayBufferConfig
from xtuner.v1.data_proto.rl_data import SampleParams
from xtuner.v1.ray.environment import SingleTurnEnvironment
from xtuner.v1.ray.judger.dapo_math import DapoMathJudgerConfig
from xtuner.v1.datasets import RLTokenizeFnConfig, build_datasets
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.ray.rollout.controller import RolloutController
# 导入 Mock Worker
from xtuner.v1.utils.rl_test_utils import MockTimeoutRolloutWorker, MockRequestErrorRolloutWorker, MockClientErrorRolloutWorker, MockServerErrorRolloutWorker


MODEL_PATH = os.environ["ROLLOUT_MODEL_PATH"] 
TRAIN_DATA_PATH = os.environ["ROLLOUT_DATA_PATH"]
TEST_DATA_PATH = os.environ["ROLLOUT_DATA_PATH"]
resource_map = {"npu": "NPU", "cuda": "GPU"}

@ray.remote
class MockTimeoutRolloutController(RolloutController):
    def _get_worker_cls(self):
        return MockTimeoutRolloutWorker

@ray.remote
class MockRequestErrorRolloutController(RolloutController):
    def _get_worker_cls(self):
        return MockRequestErrorRolloutWorker

@ray.remote
class MockClientErrorRolloutController(RolloutController):
    def _get_worker_cls(self):
        return MockClientErrorRolloutWorker

@ray.remote
class MockServerErrorRolloutController(RolloutController):
    def _get_worker_cls(self):
        return MockServerErrorRolloutWorker

class TestMockRollout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["XTUNER_USE_FA3"] = "1"

    @classmethod
    def tearDownClass(cls):
        del os.environ["XTUNER_USE_FA3"]
        ray.shutdown()

    def setUp(self):
        ray.init(num_cpus=80, ignore_reinit_error=True)
        self.global_batch_size = 3
        self.max_prompt_length = 4096
        self.max_response_length = 128
        self.max_concurrent = 3
        self.max_retry_times = 3

        self.resources_cfg = AcceleratorResourcesConfig(
            accelerator=resource_map[torch.accelerator.current_accelerator().type],
            num_workers=8,
            num_cpus_per_worker=8,
            cpu_memory_per_worker=16 * 1024**3,  # 16 GB
        )
        self.pg = AutoAcceleratorWorkers.build_placement_group(self.resources_cfg)

        self.rollout_cfg = RolloutConfig(
            env="test_mock_rollout",
            model_path=MODEL_PATH,
            model_name=os.path.basename(MODEL_PATH).lower(),
            tokenizer_path=MODEL_PATH,
            tensor_parallel_size=1,
            context_length=self.max_prompt_length + self.max_response_length,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

        self.dataflow_cfg = DataFlowConfig(
            max_concurrent=self.max_concurrent,
            global_batch_size=self.global_batch_size,
            max_retry_times=self.max_retry_times  
        )
        train_dataset_cfg = [{
            "dataset": DatasetConfig(name="mock_data", anno_path=TRAIN_DATA_PATH),
            "tokenize_fn": RLTokenizeFnConfig(max_length=self.max_prompt_length),
        }]
        dataloader_cfg = DataloaderConfig(
            collator='fake_collator',
            pack_level='none',
            group_by_length=False,
        )
        self.replay_buffer_cfg = ReplayBufferConfig(
            dataset_cfg=train_dataset_cfg,
            dataloader_cfg=dataloader_cfg,
            tokenizer=tokenizer,
        )

    def tearDown(self):
        ray.shutdown()

    def _run_mock_test(self, mock_controller_cls, error_name: str):
        rollout_controller = mock_controller_cls.remote(self.rollout_cfg, self.pg)
        self.test_env = SingleTurnEnvironment.remote("env", self.pg, self.rollout_cfg, rollout_controller=rollout_controller)
        self.test_dataflow = DataFlow.remote("dataflow", self.dataflow_cfg, self.replay_buffer_cfg, self.test_env)
        
        # 运行一个批次
        completed_rollouts = ray.get(self.test_dataflow.run.remote(num=3))

        status = ray.get(self.test_dataflow.get_replaybuffer_status.remote())
        print(f"[{error_name}] Completed rollouts: {completed_rollouts}, Status: {status}")
        self.assertEqual(len(completed_rollouts[0]), 0, f"[{error_name}] Expected no rollouts to complete successfully.")
        self.assertEqual(status["rollout_finished_count"], 0, f"[{error_name}] Completed count in buffer should be 0.")
        self.assertEqual(status["rollout_paused_count"], 0, f"[{error_name}] Expected {self.global_batch_size} rollouts to be interrupted.")
        
    def test_rollout_with_timeout_mock(self):
        self._run_mock_test(MockTimeoutRolloutController, "timeout")
        

    def test_rollout_with_request_error_mock(self):
        self._run_mock_test(MockRequestErrorRolloutController, "request error")
    
    def test_rollout_with_client_error_mock(self):
        self._run_mock_test(MockClientErrorRolloutController, "client error")
    
    def test_rollout_with_server_error_mock(self):
        self._run_mock_test(MockServerErrorRolloutController, "server error")

if __name__ == "__main__":
    unittest.main()