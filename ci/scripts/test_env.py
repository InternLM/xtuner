import os
import argparse
from pathlib import Path

import ray
from xtuner.v1.ray.environment import EnvController
from xtuner.v1.ray.config.worker import RolloutConfig
from xtuner.v1.ray.accelerator import AcceleratorResourcesConfig, AutoAcceleratorWorkers
from xtuner.v1.ray.judger.controller import JudgerConfig

MODEL_PATH = os.environ["ROLLOUT_MODEL_PATH"]
DATA_PATH = os.environ["ROLLOUT_DATA_PATH"]

FAKE_ROLLOUT_INPUT_STR = '<|im_start|>user\nNatalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Let\'s think step by step and output the final answer after "####".<|im_end|>\n<|im_start|>assistant\n'

FAKE_INPUT_DATA_ITEM = {
    'prompt_str': '<|im_start|>user\nNatalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Let\'s think step by step and output the final answer after "####".<|im_end|>\n<|im_start|>assistant\n', 
    'num_tokens': 62, 
    'reward_model': {'ground_truth': '72', 'style': 'rule'}, 
    'ability': 'math', 
    'data_source': 'openai/gsm8k', 
    'extra_info': {'answer': 'Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72', 'index': 0, 'question': 'Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?', 'split': 'train', 'raw_prompt': '<|im_start|>user\nNatalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Let\'s think step by step and output the final answer after "####".<|im_end|>\n<|im_start|>assistant\n'}, 
    'env': 'test_env', 
    'group_id': 255971142656329732139546771377476227093, 
    'prompt_id': 22175756018538642401581407443664245296, 
    'retry_times': 0}

def parse_args():
    parser = argparse.ArgumentParser(description="Env Generate Test Script")
    parser.add_argument("--work-dir", type=str, default="work_dir")
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--prompt-repeat-k", type=int, default=1)
    parser.add_argument("--repeat-times", type=int, default=1)
    parser.add_argument("--enable-partial-rollout", type=int, default=0)
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
    from xtuner.v1.ray.judger.gsm8k import compute_reward
    judger_cfg = JudgerConfig(
        reward_functions={"math": compute_reward},
        extra_info={"math": {"score": 1, "format_score": 0.5}}
    )
    pg = AutoAcceleratorWorkers.build_placement_group(resources_cfg)
    test_rollout_env = EnvController.remote(
        "test_rollout",
        pg,
        rollout_cfg,
    )
    test_rollout_judger_env = EnvController.remote(
        "test_env",
        pg,
        rollout_cfg,
        judger_cfg
    )

    res1 = ray.get(test_rollout_env.run.remote(FAKE_ROLLOUT_INPUT_STR))
    res2 = ray.get(test_rollout_env.run.remote(FAKE_INPUT_DATA_ITEM))
    res3 = ray.get(test_rollout_judger_env.run.remote(FAKE_INPUT_DATA_ITEM))
    print(res1, res2, res3)



if __name__ == "__main__":
    main()