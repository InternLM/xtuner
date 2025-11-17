import os
import torch
import json
import time
import unittest
from transformers import AutoTokenizer
import shutil
import tempfile

import ray
from xtuner.v1.ray.base import AcceleratorResourcesConfig, AutoAcceleratorWorkers
from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.config import (
    AdamWConfig,
    FSDPConfig,
    LRConfig,
)
from xtuner.v1.model.moe.moe import BalancingLossConfig, ZLossConfig
# from xtuner.v1.rl.grpo.config import WorkerConfig, LossConfig
from xtuner.v1.rl.base import WorkerConfig, TrainingController, TrainingWorker as BaseTrainingWorker
from xtuner.v1.rl.grpo.loss import GRPOLossConfig as LossConfig
from xtuner.v1.model.dense.qwen3 import Qwen3Dense8BConfig


# Qwen3 30B A3
QWEN3_PATH = os.environ["QWEN3_PATH"]

class TestGRPOTrain(unittest.TestCase):
    def setUp(self):
        ray.init(num_cpus=80, ignore_reinit_error=True)

        resources = AcceleratorResourcesConfig(
            accelerator="GPU",
            num_accelerators_per_worker=1,
            num_cpus_per_worker=8,
            num_workers=8,
            cpu_memory_per_worker=16 * 1024**3,  # 16 GB
        )

        pg = AutoAcceleratorWorkers.build_placement_group(resources)
        self.pg = pg

        self.temp_dir = tempfile.mkdtemp()
        tokenizer = AutoTokenizer.from_pretrained(QWEN3_PATH, trust_remote_code=True)
        self.tokenizer = tokenizer
        self.prompt_repeat_k = 8
        file = './tests/ray/rollout_output.jsonl'
        with open(file, 'r') as f:
            data = [json.loads(line) for line in f]
        data_groups = [data[i:i + self.prompt_repeat_k] for i in range(0, len(data), self.prompt_repeat_k)]
        data_groups = data_groups[:8]
        data_batches = []
        for group in data_groups:
            prompt_ids = tokenizer(group[0]['prompt'], return_tensors='pt')['input_ids'].flatten().tolist()
            rewards = [item['reward'] for item in group]
            rewards = torch.tensor(rewards, dtype=torch.float32)
            advantages = (rewards - rewards.mean(0)) / (rewards.std(0) + 1e-8)

            for i in range(self.prompt_repeat_k):
                item = group[i]
                response_ids = tokenizer(item['response'], return_tensors='pt')['input_ids'].flatten().tolist()
                input_ids = prompt_ids + response_ids
                shifted_labels = [-100] * (len(prompt_ids) - 1) + response_ids + [-100]
                input_ids = torch.tensor(input_ids, dtype=torch.int64).unsqueeze(0)
                shifted_labels = torch.tensor(shifted_labels, dtype=torch.int64).unsqueeze(0)
                data_batches.append(
                    dict(
                        seq_ctx=SequenceContext.from_input_ids((input_ids, ), device="cpu"),
                        shifted_labels=shifted_labels,
                        advantage=advantages[i].item(),
                    )
                )
        self.data_batches = data_batches
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        ray.shutdown()
    
    def build_train_controller(self):
        model_cfg = Qwen3Dense8BConfig()
        optim_cfg: AdamWConfig = AdamWConfig(lr=5e-7, foreach=False)
        fsdp_cfg: FSDPConfig = FSDPConfig(
            torch_compile=True,
            cpu_offload=False,
            ep_size=1,

        )
        lr_cfg = LRConfig(lr_type="constant", warmup_ratio=0, lr_min=5e-7)
        worker_cfg: WorkerConfig = WorkerConfig(
            model_cfg=model_cfg,
            optim_cfg=optim_cfg,
            loss_cfg=LossConfig(
                policy_loss_cfg=dict(
                    cliprange_high=0.28,
                    cliprange_low=0.2,
                    loss_type="vanilla",
                ),
                ignore_idx=-100,
                use_kl_loss=True,
                kl_loss_coef=0.001, 
                kl_loss_type="low_var_kl",
                mode="eager"),
            lr_cfg=lr_cfg,
            fsdp_cfg=fsdp_cfg,
            load_from=QWEN3_PATH,
            sp_size=1,
            pack_max_length=8192,
        )
        
        TrainingWorker = ray.remote(
            runtime_env={
                "env_vars": {
                    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
                }
            },
        )(BaseTrainingWorker)
        train_workers, _ = AutoAcceleratorWorkers.from_placement_group(
            TrainingWorker, worker_cfg, self.pg
        )
        futures = [ worker.test_all_reduce.remote() for worker in train_workers ]
        print(ray.get(futures))
        train_controller = TrainingController.remote(
            workers=train_workers,
        )
        ray.get(train_controller.__ray_ready__.remote())
        return train_controller

    def test_grpo_train_and_save(self):
        train_controller = self.build_train_controller()
        ray.get(train_controller.fit.remote(self.data_batches, pack_max_length=1024, rollout_idx=0))
        save_path = os.path.join(self.temp_dir, "hf_test")
        ray.get(train_controller.save_hf.remote(str(save_path)))
