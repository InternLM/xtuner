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
from xtuner.v1.ray import find_master_addr_and_port
from xtuner.v1.ray.judger import JudgerController, Math500JudgerWorker
from xtuner.v1.ray.environment import SampleParams, EnvController
from xtuner.v1.ray.dataflow.flow import Flow, DataFlowConfig
from xtuner.v1.ray.dataflow.replay_buffer import ReplayBuffer
from xtuner.v1.ray.utils import bind_train_rollout

MODEL_PATH = os.environ["ROLLOUT_MODEL_PATH"]
DATA_PATH = os.environ["ROLLOUT_DATA_PATH"]

# example running command:
# ROLLOUT_DATA_PATH="/${WORKSPACE_PATH}/data/math-500.jsonl" \
# ROLLOUT_MODEL_PATH="/${WORKSPACE_PATH}/models--Qwen--Qwen3-30B-A3B/" \
# XTUNER_USE_VLLM=1 \
# PYTHONPATH="/${WORKSPACE_PATH}/xtuner/xtuner":${PYTHONPATH} \
# python test_rollout.py

# note: if you changes the dataset, you should alse provide the load function
# for the dataset, which should return a generator of (prompt, label) pairs.
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
        self.resources_config = AcceleratorResourcesConfig(
            num_accelerators_per_worker=1,
            num_cpus_per_worker=1,
            cpu_memory_per_worker=8 * 1024 * 1024 * 1024,
            num_workers=8,
            accelerator="GPU"
        )
        self.sample_params = SampleParams(
            top_p=0.95,
            temperature=0.6,
            max_tokens=2048,
        )
    
    def init_cpu_workers(self):
        judger_workers = []
        for i in range(self.resources_config.num_workers):
            master_addr, master_port = ray.get(find_master_addr_and_port.remote())
            worker = Math500JudgerWorker.remote(
                config=dict(),
                rank=i,
                master_addr=master_addr,
                master_port=master_port,
                world_size=self.resources_config.num_workers
            )
            judger_workers.append(worker)

        return judger_workers

    def init_env(self, gpu_workers, judger_workers):
        rollout_controller = RolloutController.remote(self.rollout_config, gpu_workers)
        judger_controller = JudgerController.remote(judger_workers)
        test_env = EnvController.remote(
            environment="test",
            rollout_controller=rollout_controller,
            judger_controller=judger_controller
        )
        return test_env

    def init_dataset(self):
        class Math500Dataset(torch.utils.data.IterableDataset):
            def __init__(self, path: str, tokenizer=None):
                super().__init__()
                offsets = [0]
                with open(path) as f:
                    lines = f.readlines()
                    for line in lines[:-1]:
                        offsets.append(offsets[-1] + len(line.encode()))
                self.offsets = offsets
                self.tokenizer = tokenizer
                self.path = path

            def __iter__(self):
                with open(self.path) as f:
                    for item in self.offsets:
                        f.seek(item)
                        line = f.readline()
                        yield line 
        dataset = Math500Dataset(self.data_path, tokenizer=self.tokenizer)  
        return dataset
    
    def init_flow(self, test_env):
        self.dataflow_config = DataFlowConfig(
            env="test",
            max_concurrent=10,
            prompt_repeat_k=1,
            target_sample_counts=10.
        )
        dataset = self.init_dataset()
        self.replay_buffer = ReplayBuffer.remote(dataset)

        from xtuner.v1.ray.dataflow.flow import DataProcessor
        data_processor = DataProcessor()
        def mapping_dataset_func(meta):
            data_str = ray.get(meta.action_ref)
            rollout_input = json.loads(data_str)["problem"] + \
                            " Let's think step by step and output the final answer within \\boxed{}."
            reward_input = json.loads(data_str)["answer"]
            return rollout_input, reward_input
        test_flow = Flow.remote(self.dataflow_config, self.test_env, self.replay_buffer, data_processor, mapping_dataset_func)
        return test_flow
    
    def setUp(self):
        ray.init(num_cpus=40)
        self.data_path = DATA_PATH
        self.model_path = MODEL_PATH
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.init_config()
        self.judger_workers = self.init_cpu_workers()

    def tearDown(self):
        ray.shutdown()

    @unittest.skipIf(os.environ.get("XTUNER_USE_VLLM", "0") == "0", "vLLM backend is not enabled")
    def test_vllm_backend_tp1(self):
        from xtuner.v1.ray.rollout import vLLMWorker
        gpu_workers, _ = AutoAcceleratorWorkers.from_config(vLLMWorker, self.rollout_config, self.resources_config)
        self.test_env = self.init_env(gpu_workers, self.judger_workers)
        ray.get(self.test_env.__ray_ready__.remote())
        self.test_flow = self.init_flow(self.test_env)
        responses = ray.get(self.test_flow.run.remote())
        self.assertEqual(len(responses), self.dataflow_config.global_batch_size)

    @unittest.skipIf(os.environ.get("XTUNER_USE_VLLM", "0") == "0", "vLLM backend is not enabled")
    def test_vllm_backend_tp8(self):
        from xtuner.v1.ray.rollout import vLLMWorker
        self.rollout_config.tensor_parallel_size = 8
        self.rollout_config.rollout_cross_node_comm = True
        gpu_workers, _ = AutoAcceleratorWorkers.from_config(vLLMWorker, self.rollout_config, self.resources_config)
        self.test_env = self.init_env(gpu_workers, self.judger_workers)
        ray.get(self.test_env.__ray_ready__.remote())
        self.test_flow = self.init_flow(self.test_env)
        responses = ray.get(self.test_flow.run.remote())
        print(f"len of response: {len(responses)}, responses: {responses}")
        self.assertEqual(len(responses), self.dataflow_config.global_batch_size)
    
    @unittest.skipIf(os.environ.get("XTUNER_USE_LMDEPLOY", "0") == "0", "SGLang backend is not enabled")
    def test_lmdeploy_backend(self):
        from xtuner.v1.ray.rollout import LMDeployWorker
        workers_map, pg = AutoAcceleratorWorkers.from_config(LMDeployWorker, self.rollout_config, self.resources_config)
        rollout_controller = self._init_rollout(self.rollout_config, workers_map)
        self._run_rollout_judger_and_pause(rollout_controller, "lmdeploy")
    
    @unittest.skipIf(os.environ.get("XTUNER_USE_LMDEPLOY", "0") == "0", "SGLang backend is not enabled")
    def test_lmdeploy_update_weights(self):
        self.rollout_config.skip_load_weights = True
        from xtuner.v1.ray.rollout import LMDeployWorker
        train_workers_map, pg = self.setup_train()
        print(f"train workers ready: {ray.get([worker.__ray_ready__.remote() for worker in train_workers_map])}")
        rollout_workers_map = AutoAcceleratorWorkers.from_placement_group(
            LMDeployWorker, self.rollout_config, pg
        )
        print(f"rollout workers ready: {ray.get([worker.__ray_ready__.remote() for worker in rollout_workers_map])}")
        rollout_controller = self._init_rollout(self.rollout_config, rollout_workers_map)
        bind_train_rollout(train_workers=train_workers_map.keys(), rollout_controller=rollout_controller)
        
        # update weights
        update_weights_futures = [worker.update_weights.remote() for worker in train_workers_map]
        ray.get(update_weights_futures)
        print("update weights done!!!")
        ray.get(rollout_controller.onload.remote(tags=["kv_cache"]))
        print("rollout load kvcache")
        self._run_rollout_judger_and_pause(rollout_controller, "lmdeploy")    

if __name__ == "__main__":
    unittest.main()