import json

import ray
import torch

from xtuner.v1.ray.accelerator import AutoAcceleratorWorkers
from xtuner.v1.ray.judger import JudgerController, Math500JudgerWorker


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


def mapping_math500_dataset_func(data_str):
    rollout_input = (
        json.loads(data_str)["problem"] + " Let's think step by step and output the final answer within \\boxed{}."
    )
    reward_input = json.loads(data_str)["answer"]
    return rollout_input, reward_input


def build_math500_judger_controller(pg):
    judger_config = dict()
    judger_workers_map = AutoAcceleratorWorkers.from_placement_group(Math500JudgerWorker, judger_config, pg)
    judger_controller = JudgerController.remote(judger_workers_map, judger_config)
    ray.get(judger_controller.__ray_ready__.remote())
    return judger_controller


def build_math500_flow(model_path, data_path, dataflow_config, env_controller):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dataset = Math500Dataset(data_path, tokenizer=tokenizer)
    from xtuner.v1.ray.dataflow import DataFlow, ReplayBuffer

    replay_buffer = ReplayBuffer.remote(dataset, mapping_math500_dataset_func)
    test_flow = DataFlow.remote(
        "test_env",
        dataflow_config,
        replay_buffer,
        env_controller,
    )
    return test_flow
