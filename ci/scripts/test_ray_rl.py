import os
import json
import argparse
import copy
from pathlib import Path

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
from xtuner.v1.data_proto.policy_loss_context import GRPOLossLossContext
import ray
# from xtuner.v1.ray.train import TrainingWorker
from xtuner.v1.ray.accelerator import AutoAcceleratorWorkers, AcceleratorResourcesConfig
from xtuner.v1.data_proto.sequence_context import SequenceContext
from transformers import AutoTokenizer
import random
import torch
from mmengine.runner import set_random_seed
from xtuner.v1.rl.grpo.controller import TrainingController
from xtuner.v1.rl.grpo.config import WorkerConfig, LossConfig
from xtuner.v1.rl.grpo.loss import GRPOLossContext
from xtuner.v1.rl.grpo.worker import TrainingWorker

from xtuner.v1.ray.rollout.controller import RolloutController
from xtuner.v1.ray.config.worker import RolloutConfig
from xtuner.v1.ray import find_master_addr_and_port
from xtuner.v1.ray.judger import Math500JudgerWorker, JudgerController
from xtuner.v1.ray.dataflow import SampleParams
from xtuner.v1.ray.rollout import vLLMWorker
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
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--max-concurrent", type=int, default=128)
    parser.add_argument("--repeat-times", type=int, default=1)
    parser.add_argument("--prompt-repeat-k", type=int, default=8)
    parser.add_argument("--debug-train-only", action="store_true")
    parser.add_argument("--debug-rollout-only", action="store_true")
    parser.add_argument("--pack-max-length", type=int, default=8192)
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


def build_rollout_controller(rollout_config, pg, outqueue):
    rollout_workers = AutoAcceleratorWorkers.from_placement_group(vLLMWorker, rollout_config, pg)

    rollout_controller = RolloutController.remote(rollout_config, rollout_workers, outqueue=outqueue)
    ray.get(rollout_controller.__ray_ready__.remote())
    return rollout_controller


def build_judger_controller(pg):
    math500_workers = []
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
    
    judger_controller = JudgerController.remote(dict(), math500_workers)
    ray.get(judger_controller.__ray_ready__.remote())
    return judger_controller


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
        global_batch_size=8,
        work_dir=args.work_dir,
    )
    train_workers = AutoAcceleratorWorkers.from_placement_group(
        TrainingWorker, worker_cfg, pg
    )
    futures = [ worker.test_all_reduce.remote() for worker in train_workers ]
    print(ray.get(futures))
    train_controller = TrainingController.remote(
        workers=train_workers,
    )
    ray.get(train_controller.__ray_ready__.remote())
    return train_controller


def rollout(
        dataset,
        rollout_controller, 
        judger_controller, 
        envqueue, 
        outqueue, 
        sample_params, 
        num_rollout_seqs,
        max_running_requests,
        prompt_repeat_k,
        work_dir,
    ):
    send_samples = 0
    data_iter = iter(dataset)
    judger_controller.judge.remote(envqueue, outqueue)
    while outqueue.qsize() < num_rollout_seqs:
        if (send_samples - envqueue.qsize() - outqueue.qsize()) < max_running_requests:
            try:
                prompt, label = next(data_iter)
            except StopIteration:
                continue
            rollout_controller.rollout.remote(prompt, label, sample_params)
            send_samples += 1      
    print(f"Sent {send_samples} samples, rollout controller received {envqueue.qsize()} samples. judger controller received {outqueue.qsize()} samples.")
    ray.get(rollout_controller.pause.remote())
    ray.get(judger_controller.pause.remote())
    # Collect 
    response_length = outqueue.qsize()
    data = []
    for _ in range(response_length):
        response_data = outqueue.get()
        response = ray.get(response_data[0][0])
        reward = response_data[1]
        data.append(
            dict(
                prompt=response.prompt,
                response=response.response,
                label=response.label,
                reward=reward,
            )
        )
    data_new = []
    for item in data:
        group = []
        group.append(item)
        for _ in range(prompt_repeat_k - 1):
            item_copy = copy.deepcopy(item)
            item_copy['reward'] = random.randint(0, 1)
            group.append(item_copy)
        data_new.append(group)
    
    result_path = Path(work_dir) / f"rollout_results.jsonl"
    with open(result_path, "w") as f:
        for group in data_new:
            for item in group:
                json.dump(item, f)
                f.write('\n')
    
    # TODO: use sleep instead
    ray.get(ray.get(rollout_controller.shutdown.remote()))
    DEVICE_MODULE.empty_cache()

    return data_new 


def main(args):
    os.makedirs(args.work_dir, exist_ok=True)
    ray.init(num_cpus=70, ignore_reinit_error=True)

    resources = AcceleratorResourcesConfig(
        accelerator="GPU",
        num_accelerators_per_worker=1,
        num_cpus_per_worker=8,
        num_workers=8,
        cpu_memory_per_worker=16 * 1024**3,  # 16 GB
    )

    pg = AutoAcceleratorWorkers.build_placement_group(resources)

    outqueue = ray.util.queue.Queue(maxsize=1000)
    envqueue = ray.util.queue.Queue(maxsize=1000)
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

    sample_params = SampleParams(
        top_p=0.95,
        temperature=0.6,
        max_tokens=2048,
        stop_token_ids=get_eos_token_ids(args.model_path),
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    dataset = Math500Dataset(args.data_path, tokenizer=tokenizer)
    
    if args.debug_rollout_only:
        rollout_controller = build_rollout_controller(rollout_config, pg, envqueue)
        judger_controller = build_judger_controller(pg)
        rollout_data = rollout(
            dataset,
            rollout_controller,
            judger_controller,
            envqueue,
            outqueue,
            sample_params,
            num_rollout_seqs=args.global_batch_size,
            max_running_requests=rollout_config.max_running_requests,
            prompt_repeat_k=args.prompt_repeat_k,
            work_dir=args.work_dir,
        )
        return
    
    if args.debug_train_only:
        file = '/cpfs01/shared/llm_razor/caoweihan/projects/xtuner_refactor/work_dirs/debug_ray_rl/rollout_results_0.jsonl'
        with open(file, 'r') as f:
            data = [json.loads(line) for line in f]
        data_groups = [data[i:i + args.prompt_repeat_k] for i in range(0, len(data), args.prompt_repeat_k)]
        # for debug
        data_groups = data_groups[:8]
    else:
        rollout_controller = build_rollout_controller(rollout_config, pg, envqueue)
        judger_controller = build_judger_controller(pg)
        data_groups = rollout(
            dataset,
            rollout_controller,
            judger_controller,
            envqueue,
            outqueue,
            sample_params,
            num_rollout_seqs=args.global_batch_size,
            max_running_requests=rollout_config.max_running_requests,
            prompt_repeat_k=args.prompt_repeat_k,
            work_dir=args.work_dir,
        )
    
    data_batches = []
    for group in data_groups:
        prompt_ids = tokenizer(group[0]['prompt'], return_tensors='pt')['input_ids'].flatten().tolist()
        rewards = [item['reward'] for item in group]
        rewards = torch.tensor(rewards, dtype=torch.float32)
        advantages = (rewards - rewards.mean(0)) / (rewards.std(0) + 1e-8)

        for i in range(args.prompt_repeat_k):
            item = group[i]
            response_ids = tokenizer(item['response'], return_tensors='pt')['input_ids'].flatten().tolist()
            input_ids = prompt_ids + response_ids
            shift_labels = [-100] * (len(prompt_ids) - 1) + response_ids + [-100]
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

    train_controller = build_train_controller(args, pg)
    ray.get(train_controller.fit.remote(data_batches, pack_max_length=args.pack_max_length))
    breakpoint()


if __name__ == "__main__":
    args = parse_args()
    main(args)
