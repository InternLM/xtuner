import os
import torch
import json
import time
import argparse
import unittest
from pathlib import Path
from transformers import AutoTokenizer

import ray
from xtuner.v1.ray.rollout.controller import RolloutController
from xtuner.v1.ray.config.worker import RolloutConfig
from xtuner.v1.ray.accelerator import AcceleratorResourcesConfig, AutoAcceleratorWorkers
from xtuner.v1.ray import find_master_addr_and_port
from xtuner.v1.ray.judger import Math500JudgerWorker, JudgerController
from xtuner.v1.ray.dataflow import SampleParams
from xtuner.v1.ray.rollout import vLLMWorker

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
    parser.add_argument("--max-concurrent", type=int, default=128)
    parser.add_argument("--repeat-times", type=int, default=5)
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
                yield (
                    json.loads(line.strip())["problem"] + 
                    " Let's think step by step and output the final answer within \\boxed{}.",
                    json.loads(line.strip())["answer"]
                )

def get_eos_token_ids(model_path: str):
    config_path = os.path.join(model_path, "generation_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    eos_token_ids = config.get("eos_token_id")
    return eos_token_ids

def setup(args):
    rollout_config = RolloutConfig(
        env="test_rollout",
        model_path=args.model_path,
        model_name=os.path.basename(args.model_path).lower(),
        tokenizer_path=args.model_path,
        rollout_cross_node_comm=False,
        max_running_requests=16,
        tensor_parallel_size=1,
        expert_parallel_size=1,
        gpus_per_node=args.gpus_per_node,
        dtype="bfloat16",
    )
    resources_config = AcceleratorResourcesConfig(
        num_accelerators_per_worker=1,
        num_cpus_per_worker=1,
        cpu_memory_per_worker=8 * 1024 * 1024 * 1024,
        num_workers=args.num_workers,
        accelerator="GPU"
    )
    sample_params = SampleParams(
        top_p=0.95,
        temperature=0.6,
        max_tokens=2048,
        stop_token_ids=get_eos_token_ids(args.model_path),
    )
    math500_workers = []
    judger_config = dict()
    for i in range(resources_config.num_workers):
        master_addr, master_port = ray.get(find_master_addr_and_port.remote())
        worker = Math500JudgerWorker.remote(
            config=judger_config,
            rank=i,
            master_addr=master_addr,
            master_port=master_port,
            world_size=resources_config.num_workers
        )
        math500_workers.append(worker)
    
    gpu_workers, _ = AutoAcceleratorWorkers.from_config(vLLMWorker, rollout_config, resources_config)
    return rollout_config, judger_config, sample_params, math500_workers, gpu_workers

def main():
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    ray.init(num_cpus=40, ignore_reinit_error=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    global_batch_size = args.global_batch_size
    max_concurrent = args.max_concurrent
    rollout_config, judger_config, sample_params, math500_workers, gpu_workers = setup(args)

    outqueue = ray.util.queue.Queue(maxsize=1000)
    envqueue = ray.util.queue.Queue(maxsize=1000)

    # rollout
    rollout_controller = RolloutController.remote(rollout_config, gpu_workers, outqueue=envqueue)
    ray.get(rollout_controller.__ray_ready__.remote())
    judger_controller = JudgerController.remote(judger_config, math500_workers)
    ray.get(judger_controller.__ray_ready__.remote())

    rewards = []
    # note: here set repeat_times greater than 1 to get avg reward
    for i in range(args.repeat_times):
        dataset = Math500Dataset(args.data_path, tokenizer=tokenizer)
        send_samples = 0
        data_iter = iter(dataset)
        while envqueue.qsize() < global_batch_size:
            if (send_samples - envqueue.qsize()) < max_concurrent:
                try:
                    prompt, label = next(data_iter)
                except StopIteration:
                    continue
                rollout_controller.rollout.remote(prompt, label, sample_params)
                send_samples += 1
            time.sleep(1)
        ray.get(rollout_controller.pause.remote())

        print("envqueue size:", envqueue.qsize())
        # judger
        judger_controller.judge.remote(envqueue, outqueue)
        while outqueue.qsize() < global_batch_size:
            print(f"outqueue size: {outqueue.qsize()}, waiting for judger to finish...")
            time.sleep(2)
        ray.get(ray.get(judger_controller.pause.remote()))
        response_length = outqueue.qsize()
        avg_reward = 0.0
        result_path = Path(args.work_dir) / f"rollout_results_{i}.jsonl"
        with open(result_path, "w") as f:
            for _ in range(response_length):
                response_data = outqueue.get()
                response = ray.get(response_data[0][0])
                reward = response_data[1]
                avg_reward += reward
                json.dump({
                    "prompt": response.prompt,
                    "response": response.response,
                    "label": response.label,
                    "reward": reward
                }, f)
                f.write('\n')
        avg_reward /= response_length
        rewards.append(avg_reward)
        print(f"Avg reward: {avg_reward:.4f}")
        print(f"Results saved to {result_path}")
    print(f"Final average reward over 5 runs: {sum(rewards) / len(rewards):.4f}")

if __name__ == "__main__":
    main()