import os
import time
import ray
import argparse

import torch 
from transformers import AutoTokenizer

from xtuner.v1.ray.config.worker import RolloutConfig
from xtuner.v1.ray.accelerator import ResourceMap, AcceleratorResourcesConfig, AutoAcceleratorWorkers
from xtuner.v1.ray.dataflow import DataFlow, DataFlowConfig, ReplayBufferConfig
from xtuner.v1.ray.environment import SingleTurnEnvironment
from xtuner.v1.ray.rollout import RolloutController, SampleParams
from xtuner.v1.datasets import RLTextTokenizeFnConfig, build_datasets, build_dataloader
from xtuner.v1.datasets.config import (
    DataloaderConfig,
    DatasetConfig,
)

TEST_TEXT_MESSAGES=[{"role": "user", "content": "Hello!"}]
os.environ['XTUNER_USE_FA3'] = "1"

def parse_args():
    parser = argparse.ArgumentParser(description="GRPO Training test scripts")
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--work-dir", type=str, default="work_dir")
    parser.add_argument("--rollout-tp-size", type=int, default=1)
    parser.add_argument("--rollout-ep-size", type=int, default=1)
    parser.add_argument("--max-concurrent", type=int, default=2048)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-response-length", type=int, default=1024)
    parser.add_argument("--ray-cluster-url", type=str, default="")
    return parser.parse_args()


def init_config(args):
    device_type = torch.accelerator.current_accelerator().type
    resources = AcceleratorResourcesConfig(
        accelerator=ResourceMap.get(device_type),
        num_accelerators_per_worker=1,
        num_cpus_per_worker=12,
        num_workers=ResourceMap.get_num_workers(device_type),
        cpu_memory_per_worker=16 * 1024**3,  # 16 GB
    )
    pg = AutoAcceleratorWorkers.build_placement_group(resources)
    rollout_config = RolloutConfig(
        env="test_env",
        model_path=args.model_path,
        model_name=os.path.basename(args.model_path).lower(),
        tokenizer_path=args.model_path,
        tensor_parallel_size=args.rollout_tp_size,
        expert_parallel_size=args.rollout_ep_size,
        rollout_max_batch_size=2048,
        extra_rollout_config={
            "lmdeploy_backend": "pytorch",
        }
    )
    dataflow_config = DataFlowConfig(
        env="test",
        max_concurrent=args.max_concurrent,
        prompt_repeat_k=1,
        global_batch_size=1,
        sample_params=SampleParams(max_tokens=args.max_response_length),
    )
    train_dataset_cfg = [
        {
        "dataset": DatasetConfig(name="gsm8k",
                                 anno_path=args.data_path,
                                 sample_ratio=1.0),
        "tokenize_fn": RLTextTokenizeFnConfig(max_length=args.max_prompt_length),
        },
    ]
    dataloader_cfg = DataloaderConfig(
        pack_max_length=32768,
        collator='fake_collator',
        pack_level='none',
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    replay_buffer_cfg = ReplayBufferConfig(
        dataset_cfg=train_dataset_cfg,
        dataloader_cfg=dataloader_cfg,
        tokenizer=tokenizer,
    )

    return pg, rollout_config, dataflow_config, replay_buffer_cfg

def main(args):
    if args.ray_cluster_url == "":
        ray.init(num_cpus=128, ignore_reinit_error=True)
    else:
        ray.init(address=args.ray_cluster_url, ignore_reinit_error=True)
    load_from = args.model_path
    pg, rollout_config, dataflow_config, replay_buffer_cfg = init_config(args)
    test_env = SingleTurnEnvironment.remote(
        "test_env",
        pg,
        rollout_cfg=rollout_config,
    )
    test_flow = DataFlow.remote(
        "test_env",
        dataflow_config,
        replay_buffer_cfg,
        test_env
    )
    # warm up
    responses = ray.get(test_flow.run.remote(num=10))

    # 统计每个response的token数量
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    start_time = time.time()
    responses = ray.get(test_flow.run.remote(num=1000))
    end_time = time.time()

    total_tokens = 0
    for data_group in responses:
        for data in data_group:
            resp = data["response_str"]
            tokens = tokenizer.encode(resp)
            total_tokens += len(tokens)
    
    duration = end_time - start_time
    tps = total_tokens / duration
    print(f"Total tokens: {total_tokens}, Duration: {duration:.2f} seconds, Throughput: {tps:.2f} tokens/second")





if __name__ == "__main__":
    args = parse_args()
    main(args)
