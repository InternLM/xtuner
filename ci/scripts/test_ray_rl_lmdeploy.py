import os
import torch
import argparse
import json
import time
import unittest
from transformers import AutoTokenizer
import copy
import random
from pathlib import Path

import ray
from xtuner.v1.model.moe.qwen3 import Qwen3MoE30BA3Config
from xtuner.v1.config import (
    AdamWConfig,
    DataloaderConfig,
    DatasetConfig,
    Float8Config,
    FSDPConfig,
    LRConfig,
    BalancingLossConfig,
    ZLossConfig,
)
from xtuner.v1.ray.rollout.controller import RolloutController
from xtuner.v1.ray.config.worker import RolloutConfig
from xtuner.v1.ray.accelerator import AcceleratorResourcesConfig, AutoAcceleratorWorkers
from xtuner.v1.ray import find_master_addr_and_port
from xtuner.v1.ray.judger import JudgerController, Math500JudgerWorker
from xtuner.v1.ray.environment import SampleParams, EnvController
from xtuner.v1.ray.dataflow import Flow, DataFlowConfig, DataProcessor, ReplayBuffer
from xtuner.v1.data_proto.sequence_context import SequenceContext

from xtuner.v1.rl.grpo.controller import TrainingController
from xtuner.v1.rl.grpo.config import WorkerConfig, LossConfig
from xtuner.v1.rl.grpo.loss import GRPOLossContext
from xtuner.v1.rl.grpo.worker import TrainingWorker
from xtuner.v1.ray.rollout import LMDeployWorker
from xtuner.v1.utils import get_torch_device_module


MODEL_PATH = os.environ["ROLLOUT_MODEL_PATH"]
DATA_PATH = os.environ["ROLLOUT_DATA_PATH"]
DEVICE_MODULE = get_torch_device_module()


def parse_args():
    parser = argparse.ArgumentParser(description="VLLM Rollout Test Script")
    parser.add_argument("--work-dir", type=str, default="work_dir")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--data-path", type=str, default=DATA_PATH)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--rollout-global-batch-size", type=int, default=128)
    parser.add_argument("--train-global-batch-size", type=int, default=8)
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--prompt-repeat-k", type=int, default=8)
    parser.add_argument("--debug-train-only", action="store_true")
    parser.add_argument("--debug-rollout-only", action="store_true")
    parser.add_argument("--pack-max-length", type=int, default=8192)

    parser.add_argument("--offload-optimizer", action="store_true")  # save optimizer memory but will slow down training
    return parser.parse_args()


def bind_train_rollout(
    train_controller,
    env_controller,
) -> None:
    """Bind the training and rollout workers for update weights."""
    info_dict = ray.get(env_controller.get_rollout_info.remote())  # type: ignore[attr-defined]
    ray.get(train_controller.update_rollout_info.remote(info_dict))
    return


def get_eos_token_ids(model_path: str):
    config_path = os.path.join(model_path, "generation_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    eos_token_ids = config.get("eos_token_id")
    return eos_token_ids


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
    

def build_train_controller(args, pg):
    moe_cfg = Qwen3MoE30BA3Config(
        ep_size=1,
        balancing_loss_cfg=BalancingLossConfig(),
        z_loss_cfg=ZLossConfig(),
    )
    optim_cfg: AdamWConfig = AdamWConfig(lr=5e-7)
    fsdp_cfg: FSDPConfig = FSDPConfig(
        torch_compile=True,
        cpu_offload=False,
        ep_size=1,

    )
    lr_cfg = LRConfig(lr_type="constant", warmup_ratio=0, lr_min=5e-7)
    worker_cfg: WorkerConfig = WorkerConfig(
        model_cfg=moe_cfg,
        optim_cfg=optim_cfg,
        loss_cfg=LossConfig(cliprange_high=0.28, cliprange_low=0.2),
        lr_cfg=lr_cfg,
        fsdp_cfg=fsdp_cfg,
        load_from=args.model_path,
        tokenizer_path=args.model_path,
        sp_size=1,
        global_batch_size=args.train_global_batch_size,
        work_dir=args.work_dir,
        offload_optimizer=args.offload_optimizer,
    )
    train_workers = AutoAcceleratorWorkers.from_placement_group(
        TrainingWorker, worker_cfg, pg
    )
    ray.get([worker.__ray_ready__.remote() for worker in train_workers])
    train_workers = list(train_workers.keys())
    train_controller = TrainingController.remote(
        workers=train_workers,
    )
    ray.get(train_controller.__ray_ready__.remote())
    return train_controller


def build_judger_controller(pg):
    math500_workers = []
    print("pg.bundle_count: ", pg.bundle_count)
    for i in range(pg.bundle_count):
        master_addr, master_port = ray.get(find_master_addr_and_port.remote())
        worker = Math500JudgerWorker.remote(
            config=dict(),
            rank=i,
            master_addr=master_addr,
            master_port=master_port,
            world_size=pg.bundle_count,
        )
        math500_workers.append(worker)
    judger_controller = JudgerController.remote(math500_workers)
    ray.get(judger_controller.__ray_ready__.remote())
    return judger_controller


def build_rollout_controller(rollout_config, pg):
    rollout_workers_map = AutoAcceleratorWorkers.from_placement_group(
        LMDeployWorker, rollout_config, pg
    )
    rollout_controller = RolloutController.remote(rollout_config, rollout_workers_map)
    ray.get(rollout_controller.__ray_ready__.remote())
    return rollout_controller


def build_env_and_flow(args, pg, rollout_config, dataflow_config):
    rollout_controller = build_rollout_controller(rollout_config, pg)
    judger_controller = build_judger_controller(pg)
    test_env = EnvController.remote(
        environment="test",
        rollout_controller=rollout_controller,
        judger_controller=judger_controller
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    dataset = Math500Dataset(args.data_path, tokenizer=tokenizer)
    replay_buffer = ReplayBuffer.remote(dataset)
    data_processor = DataProcessor()
    test_flow = Flow.remote(dataflow_config, test_env, replay_buffer, data_processor, mapping_dataset_func)
    return test_env, test_flow

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
        skip_load_weights=True,
    )
    dataflow_config = DataFlowConfig(
        env="test",
        max_concurrent=args.max_concurrent,
        prompt_repeat_k=args.prompt_repeat_k,
        global_batch_size=args.rollout_global_batch_size
    )
    return rollout_config, dataflow_config

def main(args):
    args.work_dir = Path(args.work_dir)
    os.makedirs(args.work_dir, exist_ok=True)
    ray.init(num_cpus=80, ignore_reinit_error=True)

    resources = AcceleratorResourcesConfig(
        accelerator="GPU",
        num_accelerators_per_worker=1,
        num_cpus_per_worker=8,
        num_workers=args.num_workers,
        cpu_memory_per_worker=16 * 1024**3,  # 16 GB
    )
    pg = AutoAcceleratorWorkers.build_placement_group(resources)
    rollout_config, dataflow_config = init_config(args)
    
    train_controller = build_train_controller(args, pg)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if args.debug_rollout_only:
        test_env, test_flow = build_env_and_flow(args, pg, rollout_config, dataflow_config)
        bind_train_rollout(train_controller=train_controller, env_controller=test_env)

        # update weights
        ray.get(train_controller.update_weights.remote())
        print("update weights done!!!")
        ray.get(test_env.onload.remote(tags=["kv_cache"]))
        print("rollout load kvcache")
        ray.get(train_controller.offload.remote())
        
        rollout_data = ray.get(test_flow.run.remote())
        return
    
    if args.debug_train_only:
        file = '/cpfs01/shared/llm_razor/caoweihan/projects/xtuner_refactor/work_dirs/debug_lmdeploy/rollout_results_copy.jsonl'
        with open(file, 'r') as f:
            data = [json.loads(line) for line in f]
        data_groups = [data[i:i + args.prompt_repeat_k] for i in range(0, len(data), args.prompt_repeat_k)]
        # for debug
        data_groups = [
            json.dumps(
                {
                    "prompt": group[0]["prompt"],
                    "response": [item["response"] for item in group],
                    "reward": [item["reward"] for item in group],
                }
            )
            for group in data_groups
        ]
    else:
        test_env, test_flow = build_env_and_flow(args, pg, rollout_config, dataflow_config)
        bind_train_rollout(train_controller=train_controller, env_controller=test_env)

        # update weights
        ray.get(train_controller.update_weights.remote())
        print("update weights done!!!")
        ray.get(test_env.onload.remote(tags=["kv_cache"]))
        print("rollout load kvcache")
        ray.get(train_controller.offload.remote())
        data_groups = ray.get(test_flow.run.remote())

        result_path = args.work_dir / f"rollout_results.jsonl"
        with open(result_path, "w") as f:
            for group in data_groups:
                group = json.loads(group)
                for response, reward in zip(group["response"], group["reward"]):
                    item = {
                        "prompt": group["prompt"],
                        "response": response,
                        "label": group["label"],
                        "reward": reward,
                    }
                    json.dump(item, f)
                    f.write('\n')

        ray.get(test_env.pause.remote())
        time.sleep(5)
        ray.get(test_env.offload.remote())
        ray.get(train_controller.onload.remote())
    
    data_batches = []
    for group in data_groups:
        group = json.loads(group)
        prompt_ids = tokenizer(group['prompt'], return_tensors='pt')['input_ids'].flatten().tolist()
        rewards = [item for item in group["reward"]]
        rewards = torch.tensor(rewards, dtype=torch.float32)
        advantages = (rewards - rewards.mean(0)) / (rewards.std(0) + 1e-8)

        for i in range(args.prompt_repeat_k):
            item = group['response'][i]
            response_ids = tokenizer(item, return_tensors='pt')['input_ids'].flatten().tolist()
            input_ids = prompt_ids + response_ids
            shift_labels = [-100] * (len(prompt_ids) - 1) + response_ids + [-100]
            if len(input_ids) > args.pack_max_length:
                input_ids = input_ids[:args.pack_max_length]
                shift_labels = shift_labels[:args.pack_max_length]
            input_ids = torch.tensor(input_ids, dtype=torch.int64).unsqueeze(0)
            shift_labels = torch.tensor(shift_labels, dtype=torch.int64).unsqueeze(0)
            data_batches.append(
                dict(
                    seq_ctx=SequenceContext.from_input_ids((input_ids, ), device="cpu"),
                    shift_labels=shift_labels,
                    advantage=advantages[i].item(),
                )
            )
    random.shuffle(data_batches)
    print(f"data_batches size: {len(data_batches)}")
    ray.get(train_controller.fit.remote(data_batches, pack_max_length=args.pack_max_length))


if __name__ == "__main__":
    args = parse_args()
    main(args)
