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
from xtuner.v1.ray.dataflow import DataFlowConfig
from xtuner.v1.data_proto.sequence_context import SequenceContext

from xtuner.v1.rl.grpo.controller import GRPOTrainingController
from xtuner.v1.rl.grpo.config import WorkerConfig, LossConfig
from xtuner.v1.rl.grpo.worker import GRPOTrainingWorker
from xtuner.v1.ray.rollout import LMDeployWorker
from xtuner.v1.ray.environment import EnvController
from xtuner.v1.utils import get_torch_device_module
from xtuner.v1.ray.dataflow import DataFlow
from xtuner.v1.datasets import RLTextTokenizeFnConfig, build_datasets, build_dataloader


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
    parser.add_argument("--optimizer-disable-foreach", action="store_true")  # save memory usage during opt.step()
    return parser.parse_args()


def bind_train_rollout(
    train_controller,
    env_controller,
) -> None:
    """Bind the training and rollout workers for update weights."""
    info_dict = ray.get(env_controller.get_rollout_info.remote())  # type: ignore[attr-defined]
    ray.get(train_controller.update_rollout_info.remote(info_dict))
    return

def build_train_controller(args, pg):
    moe_cfg = Qwen3MoE30BA3Config(
        ep_size=1,
        balancing_loss_cfg=BalancingLossConfig(),
        z_loss_cfg=ZLossConfig(),
    )
    optim_cfg: AdamWConfig = AdamWConfig(lr=5e-7, foreach=False if args.optimizer_disable_foreach else None)
    fsdp_cfg: FSDPConfig = FSDPConfig(
        torch_compile=False,
        cpu_offload=False,
        ep_size=1,

    )
    lr_cfg = LRConfig(lr_type="constant", warmup_ratio=0, lr_min=5e-7)
    worker_cfg: WorkerConfig = WorkerConfig(
        model_cfg=moe_cfg,
        optim_cfg=optim_cfg,
        loss_cfg=LossConfig(
            policy_loss_cfg=dict(
                cliprange_high=0.28,
                cliprange_low=0.2,
                loss_type="vanilla",
            ),
            ignore_idx=-100,
            use_kl_loss=False,
            mode="eager"),
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
        GRPOTrainingWorker, worker_cfg, pg
    )
    ray.get([worker.__ray_ready__.remote() for worker in train_workers])
    train_workers = list(train_workers.keys())
    train_controller = GRPOTrainingController.remote(
        workers=train_workers,
    )
    ray.get(train_controller.__ray_ready__.remote())
    return train_controller

def prepare_train_data(data_groups, tokenizer, prompt_repeat_k, pack_max_length):
    data_batches = []
    for group in data_groups:
        prompt_ids = tokenizer(group[0]["prompt_str"], return_tensors='pt')['input_ids'].flatten().tolist()
        rewards = [data["reward"] for data in group]
        rewards = torch.tensor(rewards, dtype=torch.float32)
        advantages = (rewards - rewards.mean(0)) / (rewards.std(0) + 1e-8)

        for i in range(prompt_repeat_k):
            item = group[i]["response_str"]
            response_ids = tokenizer(item, return_tensors='pt')['input_ids'].flatten().tolist()
            input_ids = prompt_ids + response_ids
            shifted_labels = [-100] * (len(prompt_ids) - 1) + response_ids + [-100]
            if len(input_ids) > pack_max_length:
                input_ids = input_ids[:pack_max_length]
                shifted_labels = shifted_labels[:pack_max_length]
            input_ids = torch.tensor(input_ids, dtype=torch.int64).unsqueeze(0)
            shifted_labels = torch.tensor(shifted_labels, dtype=torch.int64).unsqueeze(0)
            data_batches.append(
                dict(
                    seq_ctx=SequenceContext.from_input_ids((input_ids, ), device="cpu"),
                    shifted_labels=shifted_labels,
                    advantage=advantages[i].item(),
                )
            )
    random.shuffle(data_batches)
    return data_batches


def save_trajectories(data_groups, save_path):
    with open(save_path, "w") as f:
        for group in data_groups:
            response_list = []
            reward_list = []
            for data in group:
                response_list.append(data["response_str"])
                reward_list.append(data["reward"])
            item = {
                "prompt": group[0]["prompt_str"],
                "response": response_list,
                "label": group[0]["reward_model"]["ground_truth"],
                "reward": reward_list,
            }
            json.dump(item, f)
            f.write('\n')


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
    judger_config = {"judger_type": "xtuner.v1.ray.judger.gsm8k.GSM8KJudgerWorker"}
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
    datasets = build_datasets(dataset_cfg, tokenizer)
    dataloader = build_dataloader(
        dataloader_config=dataloader_cfg,
        datasets=datasets,
        global_batch_size=1,
        micro_batch_size=1,
        seed=1,
    )
    test_env = EnvController.remote(
        "grpo",
        pg,
        rollout_config,
        judger_config)
    train_controller = build_train_controller(args, pg)
    if args.debug_rollout_only:
        test_flow = DataFlow.remote("grpo",dataflow_config, datasets, dataloader, tokenizer,test_env)
        bind_train_rollout(train_controller=train_controller, env_controller=test_env)

        # update weights
        ray.get(train_controller.update_weights.remote())
        print("update weights done!!!")
        ray.get(test_env.onload.remote(tags=["kv_cache"]))
        print("rollout load kvcache")
        # ray.get(train_controller.offload.remote())
        ray.get(train_controller.offload_model.remote())
        ray.get(train_controller.offload_optimizer.remote())

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
        data_batches = prepare_train_data(data_groups, tokenizer, args.prompt_repeat_k, args.pack_max_length)
        print(f"data_batches size: {len(data_batches)}")
        ray.get(train_controller.fit.remote(data_batches, pack_max_length=args.pack_max_length))
        return
    
    test_flow = DataFlow.remote("grpo",dataflow_config, datasets, dataloader, tokenizer, test_env)
    bind_train_rollout(train_controller=train_controller, env_controller=test_env)

    # update weights
    ray.get(train_controller.update_weights.remote())
    print("update weights done!!!")
    ray.get(test_env.onload.remote(tags=["kv_cache"]))
    print("rollout load kvcache")
    # ray.get(train_controller.offload.remote())
    ray.get(train_controller.offload_model.remote())
    ray.get(train_controller.offload_optimizer.remote())
    for step in range(2):
        data_groups = ray.get(test_flow.run.remote())
        result_path = args.work_dir / f"rollout_results_step{step}.jsonl"
        save_trajectories(data_groups, result_path)
        time.sleep(5)
        ray.get(test_env.offload.remote())
        ray.get(train_controller.onload.remote())
        data_batches = prepare_train_data(data_groups, tokenizer, args.prompt_repeat_k, args.pack_max_length)
        print(f"data_batches size: {len(data_batches)}")
        ray.get(train_controller.fit.remote(data_batches, pack_max_length=args.pack_max_length))
        ray.get(train_controller.offload_optimizer.remote())
        ray.get(test_env.onload.remote())
        ray.get(train_controller.update_weights.remote())
        print("update weights done!!!")
        ray.get(train_controller.offload_model.remote())


if __name__ == "__main__":
    args = parse_args()
    main(args)
