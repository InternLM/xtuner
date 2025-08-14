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
from xtuner.v1.ray.rollout import LMDeployWorker, RolloutController
from xtuner.v1.ray.config.worker import RolloutConfig
from xtuner.v1.ray.accelerator import AcceleratorResourcesConfig, AutoAcceleratorWorkers
from xtuner.v1.ray.dataflow import DataFlowConfig
from xtuner.v1.utils.math500_utils import build_math500_judger_controller, build_math500_flow

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
    parser.add_argument("--repeat-times", type=int, default=1)
    return parser.parse_args()

def init_config(args):
    rollout_config = RolloutConfig(
        env="test_env",
        model_path=args.model_path,
        model_name=os.path.basename(args.model_path).lower(),
        tokenizer_path=args.model_path,
        rollout_cross_node_comm=False,
        max_running_requests=16,
        tensor_parallel_size=8,
        expert_parallel_size=1,
        gpus_per_node=args.gpus_per_node, # gpu: 8, npu: 16
        dtype="bfloat16",
        skip_load_weights=False,
    )
    dataflow_config = DataFlowConfig(
        env="test",
        max_concurrent=args.max_concurrent,
        prompt_repeat_k=args.prompt_repeat_k,
        global_batch_size=args.global_batch_size
    )
    return rollout_config, dataflow_config

def main():
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    ray.init(num_cpus=80)
    resources = AcceleratorResourcesConfig(
        accelerator="GPU",
        num_accelerators_per_worker=1,
        num_cpus_per_worker=8,
        num_workers=args.num_workers,
        cpu_memory_per_worker=16 * 1024**3,  # 16 GB
    )
    pg = AutoAcceleratorWorkers.build_placement_group(resources)
    rollout_config, dataflow_config = init_config(args)
    rollout_workers_map = AutoAcceleratorWorkers.from_placement_group(
        LMDeployWorker, rollout_config, pg
    )
    rollout_controller = RolloutController.remote(rollout_config, rollout_workers_map)
    judger_controller = build_math500_judger_controller(pg)
    test_env, test_flow = build_math500_flow(args.model_path, args.data_path, dataflow_config, rollout_controller, judger_controller)

    reward_list = []
    for i in range(args.repeat_times):
        ray.get(test_env.restart.remote())
        rollout_data = ray.get(test_flow.run.remote())
        for data in rollout_data:
            reward_list.extend(data["reward"])
    print(f"Average reward: {sum(reward_list) / len(reward_list)}")

if __name__ == "__main__":
    main()