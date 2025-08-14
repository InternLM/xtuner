import os
import torch
import json
import time
import argparse
import unittest
from pathlib import Path
from transformers import AutoTokenizer

import asyncio
import queue

import ray
from xtuner.v1.ray.rollout import vLLMWorker, RolloutController
from xtuner.v1.ray.config.worker import RolloutConfig
from xtuner.v1.ray.accelerator import AcceleratorResourcesConfig, AutoAcceleratorWorkers
from xtuner.v1.ray import find_master_addr_and_port
from xtuner.v1.ray.judger import JudgerController, Math500JudgerWorker
from xtuner.v1.ray.environment import SampleParams, EnvController
from xtuner.v1.ray.dataflow import DataFlow, DataFlowConfig, DataProcessor, ReplayBuffer

MODEL_PATH = os.environ["ROLLOUT_MODEL_PATH"]
DATA_PATH = os.environ["ROLLOUT_DATA_PATH"]

def parse_args():
    parser = argparse.ArgumentParser(description="VLLM Rollout Test Script")
    parser.add_argument("--work-dir", type=str, default="work_dir")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--data-path", type=str, default=DATA_PATH)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--prompt-repeat-k", type=int, default=1)
    return parser.parse_args()

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
                yield line
                
def mapping_dataset_func(meta):
    data_str = ray.get(meta.action_ref)
    rollout_input = json.loads(data_str)["problem"] + \
                    " Let's think step by step and output the final answer within \\boxed{}."
    reward_input = json.loads(data_str)["answer"]
    return rollout_input, reward_input
    
def init_config(args):
    rollout_config = RolloutConfig(
        env="test_env",
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
    resources_config = AcceleratorResourcesConfig(
        num_accelerators_per_worker=1,
        num_cpus_per_worker=4,
        cpu_memory_per_worker=8 * 1024 * 1024 * 1024,
        num_workers=args.num_workers,
        accelerator="GPU"
    )
    sample_params = SampleParams(
        top_p=0.95,
        temperature=0.6,
        max_tokens=2048,
    )
    dataflow_config = DataFlowConfig(
        env="test",
        max_concurrent=args.max_concurrent,
        prompt_repeat_k=args.prompt_repeat_k,
        target_sample_counts=args.global_batch_size
    )
    return rollout_config, resources_config, sample_params, dataflow_config

def init_workers(rollout_config, resources_config):
    judger_workers = []
    for i in range(resources_config.num_workers):
        master_addr, master_port = ray.get(find_master_addr_and_port.remote())
        worker = Math500JudgerWorker.remote(
            config=dict(),
            rank=i,
            master_addr=master_addr,
            master_port=master_port,
            world_size=resources_config.num_workers
        )
        judger_workers.append(worker)
    gpu_workers, _ = AutoAcceleratorWorkers.from_config(vLLMWorker, rollout_config, resources_config)
    return gpu_workers, judger_workers

def main():
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    ray.init(num_cpus=40)

    data_path = DATA_PATH
    model_path = MODEL_PATH
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    rollout_config, resources_config, sample_params, dataflow_config = init_config(args)
    gpu_workers, judger_workers = init_workers(rollout_config, resources_config)
    # init env
    rollout_controller = RolloutController.remote(rollout_config, gpu_workers)
    judger_controller = JudgerController.remote(judger_workers)
    test_env = EnvController.remote(
        environment="test",
        rollout_controller=rollout_controller,
        judger_controller=judger_controller
    )
    # init dataflow
    dataset = Math500Dataset(data_path, tokenizer=tokenizer)
    replay_buffer = ReplayBuffer.remote(dataset)
    data_processor = DataProcessor()
    test_flow = DataFlow.remote(dataflow_config, test_env, replay_buffer, data_processor, mapping_dataset_func)
    
    # start run 
    responses = ray.get(test_flow.run.remote())
    avg_reward = sum(data[0][1] for data in responses) / len(responses) if responses else 0
    print(f"len of response: {len(responses)}, avg reward: {avg_reward}")
    ray.get(test_flow.shutdown.remote())


if __name__ == "__main__":
    main()