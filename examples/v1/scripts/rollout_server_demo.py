import os
import time
from functools import wraps
import ray

from transformers import AutoTokenizer
from xtuner.v1.utils.rl_common_config import (
    get_resources_config,
    get_rollout_config,
    get_dataflow_config,
    get_replay_buffer_config
)
from xtuner.v1.ray.rollout.controller import RolloutController
from xtuner.v1.ray.base import AutoAcceleratorWorkers
from xtuner.v1.data_proto.rl_data import SampleParams
from xtuner.v1.ray.environment import SingleTurnEnvironment
from xtuner.v1.ray.dataflow import DataFlow


ray.init(num_cpus=128, ignore_reinit_error=True)
params = {
    "work_dir": os.environ.get("WORK_DIR", ""),
    "model_path": os.environ.get("MODEL_PATH", ""),
    "data_path": os.environ.get("DATA_PATH", ""),
    # resource setting
    "num_workers": 8,
    # model settings
    "max_prompt_length": 2048,
    "max_response_length": 8192,
    # rollout settings
    "tensor_parallel_size": 8,
    "expert_parallel_size": 1,
    "rollout_max_batch_size_per_instance": 1024,
    # dataflow settings
    "global_batch_size": 3000,
    "prompt_repeat_k": 1,
    "enable_partial_rollout": 0,
    "partial_rollout_step": 0,
    # "max_concurrent": 512,
}

resource_cfg = get_resources_config(**params)
rollout_cfg = get_rollout_config(**params)
pg = AutoAcceleratorWorkers.build_placement_group(resource_cfg)
tokenizer = AutoTokenizer.from_pretrained(params["model_path"], trust_remote_code=True)

def init_rollout_controller():
    sample_params = SampleParams()
    rollout_controller = RolloutController.remote(rollout_cfg, pg)
    return rollout_controller

def init_dataflow_controller():
    dataflow_cfg = get_dataflow_config(**params)
    replay_buffer_cfg = get_replay_buffer_config(tokenizer=tokenizer, **params)
    env =  SingleTurnEnvironment.remote("test", pg, rollout_cfg = rollout_cfg)
    test_dataflow = DataFlow.remote("test", dataflow_cfg, replay_buffer_cfg, env)
    return test_dataflow

def benchmark_throughput():
    test_dataflow = init_dataflow_controller()
    statr_time = time.time()
    response = ray.get(test_dataflow.run.remote())[0]
    end_time = time.time()
    total_length = 0
    for group_response in response:
        for data in group_response:
            total_length += data.env.rollout.num_return_tokens
    
    print(f"throughput: {total_length / (end_time - statr_time)} tokens/s")

if __name__ == "__main__":
    benchmark_throughput()