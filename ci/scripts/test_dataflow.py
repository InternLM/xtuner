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
import datasets

import ray
from xtuner.v1.ray.rollout import LMDeployWorker, RolloutController
from xtuner.v1.ray.environment import EnvController
from xtuner.v1.ray.config.worker import RolloutConfig
from xtuner.v1.ray.accelerator import AcceleratorResourcesConfig, AutoAcceleratorWorkers
from xtuner.v1.ray.dataflow import DataFlow, DataFlowConfig
from xtuner.v1.utils.math500_utils import build_math500_judger_controller, build_math500_flow
from xtuner.v1.datasets import RLTextTokenizeFnConfig, build_datasets, build_dataloader
from xtuner.v1.config import (
    DataloaderConfig,
    DatasetConfig,
)

MODEL_PATH = os.environ["ROLLOUT_MODEL_PATH"]
DATA_PATH = os.environ["ROLLOUT_DATA_PATH"]

def parse_args():
    parser = argparse.ArgumentParser(description="Env Generate Test Script")
    parser.add_argument("--work-dir", type=str, default="work_dir")
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--prompt-repeat-k", type=int, default=1)
    parser.add_argument("--repeat-times", type=int, default=1)
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    ray.init(num_cpus=80)
    resources_cfg = AcceleratorResourcesConfig(
        accelerator="GPU",
        num_workers=8,
        cpu_memory_per_worker=16 * 1024**3,  # 16 GB
    )
    rollout_cfg = RolloutConfig(
        env="test_env",
        model_path=MODEL_PATH,
        model_name=os.path.basename(MODEL_PATH).lower(),
        tokenizer_path=MODEL_PATH,
        tensor_parallel_size=8,
    )
    judger_cfg = {"judger_type": "xtuner.v1.ray.judger.gsm8k.GSM8KJudgerWorker"}

    dataflow_cfg = DataFlowConfig(
        env="test",
        prompt_repeat_k=args.prompt_repeat_k,
        global_batch_size=args.global_batch_size
    )
    dataset_cfg = [
        {
        "dataset": DatasetConfig(name="gsm8k",
                                 anno_path=DATA_PATH,
                                 sample_ratio=1.0),
        "tokenize_fn": RLTextTokenizeFnConfig(max_length=16386),
        },
    ]
    dataloader_cfg = DataloaderConfig(
        pack_max_length=16384,
        collator='fake_collator',
        pack_level='none',
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    pg = AutoAcceleratorWorkers.build_placement_group(resources_cfg)
    datasets = build_datasets(dataset_cfg, tokenizer)
    dataloader = build_dataloader(
        dataloader_config=dataloader_cfg,
        datasets=datasets,
        global_batch_size=1,
        micro_batch_size=1,
        seed=1,
    )
    test_env = EnvController.remote(
        "test_env",
        pg,
        rollout_cfg,
        judger_cfg
    )
    test_flow = DataFlow.remote("test_env", 
                                dataflow_cfg,
                                datasets,
                                dataloader,
                                test_env)

    reward_list = []
    for i in range(args.repeat_times):
        ray.get(test_flow.restart.remote())
        rollout_data = ray.get(test_flow.run.remote())
        for group in rollout_data:
            for data in group:
                reward_list.append(data["reward"])
    print(f"Average reward: {sum(reward_list) / len(reward_list)}")
    ray.get(test_flow.shutdown.remote())

if __name__ == "__main__":
    main()