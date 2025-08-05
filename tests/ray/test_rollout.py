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
from xtuner.v1.ray.dataflow import SampleParams

MODEL_PATH = os.environ["ROLLOUT_MODEL_PATH"]
DATA_PATH = os.environ["ROLLOUT_DATA_PATH"]

# example running command:
# ROLLOUT_DATA_PATH="/${WORKSPACE_PATH}/data/math-500.jsonl" \
# ROLLOUT_MODEL_PATH="/${WORKSPACE_PATH}/models--Qwen--Qwen3-30B-A3B/" \
# XTUNER_USE_VLLM=1 \
# PYTHONPATH="/${WORKSPACE_PATH}/xtuner/xtuner":${PYTHONPATH} \
# python test_rollout.py

def get_eos_token_ids(model_path: str):
    config_path = os.path.join(model_path, "generation_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    eos_token_ids = config.get("eos_token_id")
    return eos_token_ids


# note: if you changes the dataset, you should alse provide the load function
# for the dataset, which should return a generator of (prompt, label) pairs.
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
                yield (
                    json.loads(line.strip())["problem"] + 
                    " Let's think step by step and output the final answer within \\boxed{}.",
                    json.loads(line.strip())["answer"]
                )

class TestRolloutController(unittest.TestCase):
    def setUp(self):
        ray.init(num_cpus=40, ignore_reinit_error=True)
        self.data_path = DATA_PATH
        self.model_path = MODEL_PATH
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.global_batch_size = 10
        self.send_samples = 0 
        self.max_concurrent = 10
    
        self.rollout_config = RolloutConfig(
            env="test_rollout",
            model_path=MODEL_PATH,
            model_name=os.path.basename(MODEL_PATH).lower(),
            tokenizer_path=MODEL_PATH,
            rollout_cross_node_comm=False,
            max_running_requests=20,
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
            stop_token_ids=get_eos_token_ids(MODEL_PATH),
        )
        self.judger_workers = []
        self.judger_config = dict()
        for i in range(self.resources_config.num_workers):
            master_addr, master_port = ray.get(find_master_addr_and_port.remote())
            worker = Math500JudgerWorker.remote(
                config=self.judger_config,
                rank=i,
                master_addr=master_addr,
                master_port=master_port,
                world_size=self.resources_config.num_workers
            )
            self.judger_workers.append(worker)

        self.outqueue = ray.util.queue.Queue(maxsize=1000)
        self.envqueue = ray.util.queue.Queue(maxsize=1000)

    def tearDown(self):
        ray.shutdown()

    def _run_rollout_judger_and_pause(self, config, workers, backend_name):
        rollout_controller = RolloutController.remote(config, workers, outqueue=self.envqueue)
        ray.get(rollout_controller.__ray_ready__.remote())

        # rollout
        dataset = Math500Dataset(self.data_path, tokenizer=self.tokenizer)
        data_iter = iter(dataset)
        while self.envqueue.qsize() < self.global_batch_size:
            if (self.send_samples - self.envqueue.qsize()) < self.max_concurrent:
                prompt, label = next(data_iter)
                rollout_controller.rollout.remote(prompt, label, self.sample_params)
                self.send_samples += 1
            time.sleep(1) 
        ray.get(rollout_controller.pause.remote())      

        # judger
        judger_controller = JudgerController.remote(self.judger_config, self.judger_workers)
        ray.get(judger_controller.__ray_ready__.remote())
        ray.get(judger_controller.judge.remote(self.envqueue, self.outqueue))
        while self.outqueue.qsize() < self.global_batch_size:
            time.sleep(1)
        response_length = self.outqueue.qsize()
        avg_reward = 0.0
        with open(f"rollout_{backend_name}_unittest_results.jsonl", "w") as f:
            for _ in range(response_length):
                response_data = self.outqueue.get() # tuple(tuple(objectref,), reward)
                response = ray.get(response_data[0][0])
                reward = response_data[1]
                avg_reward += reward

                json.dump({"prompt": response.prompt,
                           "response": response.response,
                           "label": response.label,
                           "reward": reward}, f)
                f.write('\n')
        avg_reward /= response_length
        self.assertEqual(avg_reward, 1.0)

    def test_vllm_backend(self):
        from xtuner.v1.ray.rollout import vLLMWorker
        self.rollout_config.rollout_cross_node_comm = True
        workers, pg = AutoAcceleratorWorkers.from_config(vLLMWorker, self.rollout_config, self.resources_config)
        self._run_rollout_judger_and_pause(self.rollout_config, workers, "vllm")

    def test_sglang_backend(self):
        from xtuner.v1.ray.rollout import SGLangWorker
        workers, pg = AutoAcceleratorWorkers.from_config(SGLangWorker, self.rollout_config, self.resources_config)
        self._run_rollout_judger_and_pause(self.rollout_config, workers, "sglang")

    def test_lmdeploy_backend(self):
        pass

if __name__ == "__main__":
    unittest.main()