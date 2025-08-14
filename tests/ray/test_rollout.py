import os
import torch
import json
import time
import unittest
from transformers import AutoTokenizer

import ray
from xtuner.v1.ray.rollout.controller import RolloutController
from xtuner.v1.ray.config.worker import RolloutConfig
from xtuner.v1.ray.accelerator import AcceleratorResourcesConfig, AutoAcceleratorWorkers
from xtuner.v1.ray.dataflow.flow import DataFlowConfig
from xtuner.v1.utils.math500_utils import build_math500_judger_controller, build_math500_flow

MODEL_PATH = os.environ["ROLLOUT_MODEL_PATH"]
DATA_PATH = os.environ["ROLLOUT_DATA_PATH"]


class TestRollout(unittest.TestCase):
    def init_config(self):
        self.rollout_config = RolloutConfig(
            env="test_rollout",
            model_path=MODEL_PATH,
            model_name=os.path.basename(MODEL_PATH).lower(),
            tokenizer_path=MODEL_PATH,
            rollout_cross_node_comm=False,
            max_running_requests=16,
            tensor_parallel_size=1,
            expert_parallel_size=1,
            gpus_per_node=8, # gpu: 8, npu: 16
            dtype="bfloat16",
        )
        self.dataflow_config = DataFlowConfig(
            env="test",
            max_concurrent=1,
            prompt_repeat_k=1,
            global_batch_size=1.
        )

    def build_env_and_flow(self, rollout_worker):
        rollout_workers_map = AutoAcceleratorWorkers.from_placement_group(
            rollout_worker, self.rollout_config, self.pg
        )
        rollout_controller = RolloutController.remote(self.rollout_config, rollout_workers_map)
        judger_controller = build_math500_judger_controller(self.pg)
        test_env, test_flow = build_math500_flow(self.model_path, self.data_path, self.dataflow_config, rollout_controller, judger_controller)

        return test_env, test_flow
    

    def setUp(self):
        ray.init(num_cpus=80)
        self.data_path = DATA_PATH
        self.model_path = MODEL_PATH
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        resources = AcceleratorResourcesConfig(
            accelerator="GPU",
            num_accelerators_per_worker=1,
            num_cpus_per_worker=8,
            num_workers=8,
            cpu_memory_per_worker=16 * 1024**3,  # 16 GB
        )
        self.pg = AutoAcceleratorWorkers.build_placement_group(resources)
        self.init_config()

    def tearDown(self):
        ray.shutdown()

    @unittest.skipIf(os.environ.get("XTUNER_USE_VLLM", "0") == "0", "vLLM backend is not enabled")
    def test_vllm_backend_tp1(self):
        from xtuner.v1.ray.rollout import vLLMWorker
        _, test_flow = self.build_env_and_flow(vLLMWorker)
        responses = ray.get(test_flow.run.remote())
        self.assertEqual(len(responses), self.dataflow_config.global_batch_size)

    @unittest.skipIf(os.environ.get("XTUNER_USE_VLLM", "0") == "0", "vLLM backend is not enabled")
    def test_vllm_backend_tp8(self):
        from xtuner.v1.ray.rollout import vLLMWorker
        self.rollout_config.tensor_parallel_size = 8
        self.rollout_config.rollout_cross_node_comm = True
        _, test_flow = self.build_env_and_flow(vLLMWorker)
        responses = ray.get(test_flow.run.remote())
        self.assertEqual(len(responses), self.dataflow_config.global_batch_size)
    
    @unittest.skipIf(os.environ.get("XTUNER_USE_LMDEPLOY", "0") == "0", "SGLang backend is not enabled")
    def test_lmdeploy_backend(self):
        from xtuner.v1.ray.rollout import LMDeployWorker
        _, test_flow = self.build_env_and_flow(LMDeployWorker)
        responses = ray.get(test_flow.run.remote())
        print(f"len of response: {len(responses)}")
        self.assertEqual(len(responses), self.dataflow_config.global_batch_size)
    
if __name__ == "__main__":
    unittest.main()