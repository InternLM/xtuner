import os
import unittest
from transformers import AutoTokenizer
import shutil
import tempfile
import json
import torch
from xtuner.v1.data_proto.sequence_context import SequenceContext
import ray
from xtuner.v1.ray.base import AcceleratorResourcesConfig, AutoAcceleratorWorkers
from xtuner.v1.config import (
    AdamWConfig,
    FSDPConfig,
    LRConfig,
)
from xtuner.v1.rl.base import WorkerConfig, TrainingController, TrainingWorker as BaseTrainingWorker
from xtuner.v1.rl.grpo.loss import GRPOLossConfig as LossConfig
from xtuner.v1.model.dense.qwen3 import Qwen3Dense8BConfig
from xtuner.v1.datasets.sft_tokenize_fn import OpenaiTokenizeFunctionConfig
from xtuner.v1.loss import CELossConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.train.trainer import LoadCheckpointConfig

QWEN3_PATH = os.environ["QWEN3_PATH"]
ALPACA_PATH = os.environ["ALPACA_PATH"]


class TestRLTrainWithSFT(unittest.TestCase):
    def setUp(self):
        ray.init(num_cpus=80, ignore_reinit_error=True)

        resources = AcceleratorResourcesConfig(
            accelerator="GPU",
            num_accelerators_per_worker=1,
            num_cpus_per_worker=8,
            num_workers=8,
            cpu_memory_per_worker=16 * 1024 ** 3,  # 16 GB
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
                        seq_ctx=SequenceContext.from_input_ids((input_ids,), device="cpu"),
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

        dataset_config = []
        _data_cfg = {"dataset": DatasetConfig(name='apach',
                                              anno_path=ALPACA_PATH),
                     "tokenize_fn": OpenaiTokenizeFunctionConfig(
                         chat_template='qwen3',
                         max_length=32768
                     )
                     }
        dataset_config.append(_data_cfg)

        sft_dataloader_cfg = DataloaderConfig(
            dataset_config_list=dataset_config,
            pack_max_length=32768,
            pack_to_max_length=True,
            num_workers=0,
        )
        sft_global_batch_size = 8
        loss_reduction = "square"
        sft_loss_cfg = CELossConfig(mode="chunk", chunk_size=1024, loss_reduction=loss_reduction)

        worker_cfg: WorkerConfig = WorkerConfig(
            sft_dataloader_cfg=sft_dataloader_cfg,
            sft_global_batch_size=sft_global_batch_size,
            sft_loss_cfg=sft_loss_cfg,
            seed=42,
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
        futures = [worker.test_all_reduce.remote() for worker in train_workers]
        print(ray.get(futures))
        train_controller = TrainingController.remote(
            workers=train_workers,
        )
        ray.get(train_controller.__ray_ready__.remote())
        return train_controller

    def test_rl_train_with_sft(self):
        train_controller = self.build_train_controller()

        ray.get(train_controller.fit.remote(self.data_batches, pack_max_length=1024, rollout_idx=0))
        ray.get(train_controller.save.remote(os.path.join(self.temp_dir, "save_test"), no_save_optimizer=True))

        log_infos = ray.get(train_controller.fit.remote(self.data_batches, pack_max_length=1024, rollout_idx=1))
        efficient_attn_ratio_list = []
        for log_info in log_infos:
            efficient_attn_ratio_list.append(log_info['sft_train_metrics']['efficient_attn_ratio'])
        assert all([efficient_attn_ratio > 0 for efficient_attn_ratio in efficient_attn_ratio_list])

        ray.kill(train_controller)
        train_controller = self.build_train_controller()
        load_checkpoint_cfg = LoadCheckpointConfig(checkpoint_path=os.path.join(self.temp_dir, "save_test"),
                                                   load_optimizer_states=False,
                                                   load_optimizer_args=False
                                                   )
        ray.get(train_controller.resume.remote(load_checkpoint_cfg))

        log_infos = ray.get(train_controller.fit.remote(self.data_batches, pack_max_length=1024, rollout_idx=1))
        new_efficient_attn_ratio_list = []
        for log_info in log_infos:
            new_efficient_attn_ratio_list.append(log_info['sft_train_metrics']['efficient_attn_ratio'])

        efficient_attn_ratio_list.sort()
        new_efficient_attn_ratio_list.sort()
        self.assertEqual(efficient_attn_ratio_list, new_efficient_attn_ratio_list)
