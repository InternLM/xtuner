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

MODEL_PATH = os.environ["ROLLOUT_MODEL_PATH"]
DATA_PATH = os.environ["ROLLOUT_DATA_PATH"]

class ActionDataset(torch.utils.data.IterableDataset):
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
                yield json.loads(line.strip())["prompt"][0]["content"]

class TestRolloutController(unittest.TestCase):
    def setUp(self):
        ray.init(num_cpus=40, ignore_reinit_error=True)
        self.data_path = DATA_PATH
        self.model_path = MODEL_PATH
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.global_batch_size = 5
        self.send_samples = 0 
        self.max_concurrent = 10
        
        self.rollout_config = RolloutConfig(
            env="test_rollout",
            model_path=MODEL_PATH,
            model_name=os.path.basename(MODEL_PATH).lower(),
            tokenizer_path=MODEL_PATH,
            rollout_cross_node_comm=False,
            max_running_requests=5,
            tensor_parallel_size=1,
            expert_parallel_size=1,
            gpus_per_node=8, # gpu: 8, npu: 16
        )
        self.resources_config = AcceleratorResourcesConfig(
            num_accelerators_per_worker=1,
            num_cpus_per_worker=1,
            cpu_memory_per_worker=8 * 1024 * 1024 * 1024,
            num_workers=1,
            accelerator="GPU"
        )

    def tearDown(self):
        ray.shutdown()

    def _run_rollout_and_pause(self, config, workers, backend_name):
        outqueue = ray.util.queue.Queue(maxsize=1000)
        rollout_controller = RolloutController.remote(config, workers, outqueue=outqueue)
        ray.get(rollout_controller.__ray_ready__.remote())

        dataset = ActionDataset(self.data_path, tokenizer=self.tokenizer)
        while outqueue.qsize() < self.global_batch_size:
            if (self.send_samples - outqueue.qsize()) < self.max_concurrent:
                data = next(iter(dataset))
                rollout_controller.rollout.remote(data)
                self.send_samples += 1
            time.sleep(0.1)
        ray.get(rollout_controller.pause.remote())      
        self.assertGreaterEqual(outqueue.qsize(), self.global_batch_size)

    def test_vllm_backend(self):
        from xtuner.v1.ray.rollout import vLLMWorker
        self.rollout_config.rollout_cross_node_comm = True
        workers, pg = AutoAcceleratorWorkers.from_config(vLLMWorker, self.rollout_config, self.resources_config)
        self._run_rollout_and_pause(self.rollout_config, workers, "vllm")

    def test_sglang_backend(self):
        from xtuner.v1.ray.rollout import SGLangWorker
        workers, pg = AutoAcceleratorWorkers.from_config(SGLangWorker, self.rollout_config, self.resources_config)
        self._run_rollout_and_pause(self.rollout_config, workers, "sglang")

    def test_lmdeploy_backend(self):
        pass

if __name__ == "__main__":
    unittest.main()