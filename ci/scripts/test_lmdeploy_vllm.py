import argparse
import json
import os
import time
from pathlib import Path

import ray
from transformers import AutoTokenizer

from xtuner.v1.datasets.config import (
    DataloaderConfig,
    DatasetConfig,
)
from xtuner.v1.ray.rollout import SampleParams
from xtuner.v1.datasets import RLTextTokenizeFnConfig
from xtuner.v1.ray.accelerator import AcceleratorResourcesConfig, AutoAcceleratorWorkers
from xtuner.v1.ray.config.worker import RolloutConfig
from xtuner.v1.ray.dataflow import DataFlow, DataFlowConfig, ReplayBufferConfig
from xtuner.v1.ray.environment import SingleTurnEnvironment
from xtuner.v1.ray.judger import JudgerConfig

import numpy as np

MODEL_PATH = os.environ["ROLLOUT_MODEL_PATH"]
TRAIN_DATA_PATH = os.environ["ROLLOUT_DATA_PATH"]
TEST_DATA_PATH = os.environ["ROLLOUT_TEST_DATA_PATH"]

def parse_args():
    parser = argparse.ArgumentParser(description="Env Generate Test Script")
    parser.add_argument("--work-dir", type=str, default="work_dir")
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=1)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--prompt-repeat-k", type=int, default=1)
    parser.add_argument("--max-prompt-length", type=int, default=1)
    parser.add_argument("--max-response-length", type=int, default=1)
    parser.add_argument("--repeat-times", type=int, default=1)
    parser.add_argument("--enable-partial-rollout", type=int, default=0)
    parser.add_argument("--vllm", action="store_true")
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
    if args.vllm:
        rollout_cfg = RolloutConfig(
            env="test_env",
            model_path=MODEL_PATH,
            model_name=os.path.basename(MODEL_PATH).lower(),
            tokenizer_path=MODEL_PATH,
            tensor_parallel_size=1,
            backend="vllm",
            launch_server_method="multiprocessing",
        )
    else:
        rollout_cfg = RolloutConfig(
            env="test_env",
            model_path=MODEL_PATH,
            model_name=os.path.basename(MODEL_PATH).lower(),
            tokenizer_path=MODEL_PATH,
            tensor_parallel_size=1,
        )
    from xtuner.v1.ray.judger.gsm8k import GSM8KJudgerConfig
    gsm8k_judger_config = GSM8KJudgerConfig()
    judger_cfg = JudgerConfig(
        reward_judger_configs={"openai/gsm8k": gsm8k_judger_config}
    )

    dataflow_cfg = DataFlowConfig(
        env="test",
        prompt_repeat_k=args.prompt_repeat_k,
        global_batch_size=args.global_batch_size,
        enable_partial_rollout=args.enable_partial_rollout,
        sample_params=SampleParams(
            max_tokens=args.max_response_length,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            min_tokens=0,
            ),
    )
    dataset_cfg = [
        {
        "dataset": DatasetConfig(name="gsm8k",
                                 anno_path=TRAIN_DATA_PATH,
                                 sample_ratio=1.0),
        "tokenize_fn": RLTextTokenizeFnConfig(max_length=args.max_prompt_length),
        },
    ]
    dataloader_cfg = DataloaderConfig(
        pack_max_length=16384,
        collator='fake_collator',
        pack_level='none',
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    replay_buffer_cfg = ReplayBufferConfig(
        dataset_cfg=dataset_cfg,
        dataloader_cfg=dataloader_cfg,
        tokenizer=tokenizer,
        postprocessor=None
    )
    pg = AutoAcceleratorWorkers.build_placement_group(resources_cfg)
    test_env = SingleTurnEnvironment.remote(
        "test_env",
        pg,
        rollout_cfg,
        judger_cfg
    )
    test_flow = DataFlow.remote("test_env", 
                                dataflow_cfg,
                                replay_buffer_cfg,
                                test_env)

    for i in range(6):
        rollout_data = ray.get(test_flow.run.remote())
        dataflow_state = ray.get(test_flow.state.remote())
        result_path = os.path.join(args.work_dir, f"vllm{args.vllm}_rollout_results_round{i}.jsonl")
        with open(result_path, "w") as f:
            for group in rollout_data:
                group_prompt_len = []
                group_response_len = []
                group_reward_list = []
                for data in group:
                    promt_str = group[0]["messages"][0]['content']
                    prompt_ids = tokenizer(promt_str, return_tensors="pt")["input_ids"].flatten().tolist()
                    group_prompt_len.append(len(prompt_ids))

                    response_str = data["response_str"]
                    response_ids = tokenizer(response_str, return_tensors="pt")["input_ids"].flatten().tolist()
                    group_response_len.append(len(response_ids))

                    group_reward_list.append(data["reward"])
                    item = {
                        "messages": group[0]["messages"],
                        "response": response_str,
                        "label": group[0]["reward_model"]["ground_truth"],
                        "reward": data["reward"],
                    }
                    json.dump(item, f)
                    f.write('\n')

        print(f"test_time{i}===============================================")
        print(f"prompt_len: mean {np.mean(group_prompt_len)} max {np.max(group_prompt_len)} min {np.min(group_prompt_len)} std {np.std(group_prompt_len)}")
        print(f"response_len: mean {np.mean(group_response_len)} max {np.max(group_response_len)} min {np.min(group_response_len)} std {np.std(group_response_len)}")
        print(f"Average reward: {np.mean(group_reward_list)}")
        time.sleep(2)
    ray.get(test_env.shutdown.remote())

if __name__ == "__main__":
    main()
