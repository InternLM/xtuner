import os
import unittest
import shutil
import tempfile
import ray
from pathlib import Path

from xtuner.v1.rl.utils import AcceleratorResourcesConfig
from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.model import get_model_config_from_hf
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.datasets.rl_tokenize_fn import RLTextTokenizeFnConfig
from xtuner.v1.train.trainer import LoadCheckpointConfig
from xtuner.v1.train.rl_trainer import RLColocateTrainerConfig
from xtuner.v1.rl.trainer import WorkerConfig
from xtuner.v1.rl.loss import GRPOLossConfig
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.judger import GSM8KJudgerConfig
from xtuner.v1.loss import CELossConfig
from xtuner.v1.datasets.sft_tokenize_fn import OpenaiTokenizeFunctionConfig
from xtuner.v1.rl.replay_buffer import SyncReplayBufferConfig
from xtuner.v1.rl.agent_loop import SingleTurnAgentLoopConfig
from xtuner.v1.rl.agent_loop_manager import (
    AgentLoopManagerConfig,
    SamplerConfig,
    SyncProduceStrategyConfig,
    TaskSpecConfig,
)
from xtuner.v1.rl.evaluator import EvaluatorConfig
from xtuner.v1.data_proto import SampleParams
from xtuner.v1.data_proto.sequence_context import SequenceContext
from transformers import AutoTokenizer
import torch

QWEN3_PATH = os.environ["QWEN3_PATH"]
ALPACA_PATH = os.environ["ALPACA_PATH"]
ROLLOUT_DATA_PATH = os.environ["ROLLOUT_DATA_PATH"]


class TestRLColocateTrainerIntegration(unittest.TestCase):
    """Integration test for RLColocateTrainer with checkpoint save/resume."""

    def setUp(self):
        ray.init(num_cpus=80, num_gpus=8, ignore_reinit_error=True)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        ray.shutdown()

    def build_trainer_config(self, work_dir, checkpoint_interval=1, checkpoint_maxkeep=2, auto_resume=False):
        """Build RLColocateTrainerConfig for testing."""
        model_path = QWEN3_PATH
        data_path = ALPACA_PATH

        # Resources
        resources = AcceleratorResourcesConfig(
            accelerator="GPU",
            num_workers=8,
            num_cpus_per_worker=4,
            cpu_memory_per_worker=8 * 1024**3,
        )

        # Rollout config
        rollout_config = RolloutConfig(
            env="test_rl",
            device="GPU",
            model_path=model_path,
            dtype="bfloat16",
            tensor_parallel_size=1,
            expert_parallel_size=1,
            gpu_memory_utilization=0.5,
            context_length=1536,
        )

        # Judger
        judger_config = GSM8KJudgerConfig(judger_name="openai/gsm8k", num_ray_actors=1)

        # Train worker
        lr_cfg = LRConfig(lr_type="constant", warmup_ratio=0, lr_min=1e-6)
        fsdp_cfg = FSDPConfig(torch_compile=False, cpu_offload=False, ep_size=1)
        model_cfg = get_model_config_from_hf(Path(model_path))
        if hasattr(model_cfg, "balancing_loss_cfg"):
            model_cfg.balancing_loss_cfg = None
        if hasattr(model_cfg, "z_loss_cfg"):
            model_cfg.z_loss_cfg = None

        optim_cfg = AdamWConfig(lr=1e-6, foreach=False, weight_decay=0.1)
        loss_cfg = GRPOLossConfig(
            policy_loss_cfg=dict(
                cliprange_high=0.28,
                cliprange_low=0.2,
                loss_type="vanilla",
                clip_ratio_c=10.0,
                log_prob_diff_min=-20.0,
                log_prob_diff_max=20.0,
            ),
            ignore_idx=-100,
            use_kl_loss=False,
            kl_loss_coef=0.0,
            kl_loss_type="low_var_kl",
            mode="chunk",
            chunk_size=512,
        )

        # SFT configs for WorkerConfig
        sft_dataset_config = [{
            "dataset": DatasetConfig(name='alpaca', anno_path=data_path),
            "tokenize_fn": OpenaiTokenizeFunctionConfig(
                chat_template='qwen3',
                max_length=32768
            )
        }]
        sft_dataloader_cfg = DataloaderConfig(
            dataset_config_list=sft_dataset_config,
            pack_max_length=32768,
            pack_to_max_length=True,
            num_workers=0,
        )
        sft_global_batch_size = 8
        sft_loss_cfg = CELossConfig(mode="chunk", chunk_size=1024, loss_reduction="square")

        train_worker_cfg = WorkerConfig(
            model_cfg=model_cfg,
            load_from=model_path,
            optim_cfg=optim_cfg,
            loss_cfg=loss_cfg,
            lr_cfg=lr_cfg,
            fsdp_cfg=fsdp_cfg,
            sp_size=1,
            optimizer_steps=1,
            pack_max_length=2048,
            sft_dataloader_cfg=sft_dataloader_cfg,
            sft_global_batch_size=sft_global_batch_size,
            sft_loss_cfg=sft_loss_cfg,
        )

        # Agent loop manager
        train_dataset = DatasetConfig(name="test_rl", anno_path=ROLLOUT_DATA_PATH)
        tokenizer_config = RLTextTokenizeFnConfig(max_length=512)
        train_dataset_cfg = [{"dataset": train_dataset, "tokenize_fn": tokenizer_config}]
        dataloader_cfg = DataloaderConfig(
            dataset_config_list=train_dataset_cfg,
            pack_max_length=2048,
            collator="fake_collator",
            pack_level="none",
        )
        sampler_config = SamplerConfig(
            dataloader_cfg=dataloader_cfg,
            prompt_repeat_k=2,
        )
        training_sample_params = SampleParams(
            max_tokens=512,
            top_k=0,
            top_p=1.0,
            temperature=1.0,
            min_tokens=0,
        )
        agent_loop_config = SingleTurnAgentLoopConfig(
            hf_checkpoint=model_path,
            sample_params=training_sample_params,
        )
        produce_strategy_config = SyncProduceStrategyConfig()
        agent_loop_manager_cfg = AgentLoopManagerConfig(
            tasks=[
                TaskSpecConfig(
                    task_name="train_task",
                    agent_loop_config=agent_loop_config,
                    judger_config=judger_config,
                    produce_strategy_config=produce_strategy_config,
                    sampler_config=sampler_config,
                )
            ],
        )

        # Eval agent loop manager (minimal)
        eval_sampler_config = SamplerConfig(
            dataloader_cfg=dataloader_cfg,
            prompt_repeat_k=1,
        )
        eval_agent_loop_config = SingleTurnAgentLoopConfig(
            hf_checkpoint=model_path,
            sample_params=SampleParams(max_tokens=512, top_k=1, temperature=0.0),
        )
        eval_agent_loop_manager_cfg = AgentLoopManagerConfig(
            tasks=[
                TaskSpecConfig(
                    task_name="eval_task",
                    agent_loop_config=eval_agent_loop_config,
                    judger_config=judger_config,
                    sampler_config=eval_sampler_config,
                )
            ],
        )

        # Evaluator
        evaluator_config = EvaluatorConfig(compute_metric_func=None)

        return RLColocateTrainerConfig(
            resources=resources,
            train_worker_cfg=train_worker_cfg,
            rollout_config=rollout_config,
            tokenizer_path=model_path,
            replay_buffer_config=SyncReplayBufferConfig(),
            agent_loop_manager_cfg=agent_loop_manager_cfg,
            eval_agent_loop_manager_cfg=eval_agent_loop_manager_cfg,
            evaluator_config=evaluator_config,
            load_from=model_path,
            total_train_steps=2,
            train_batch_size=4,
            enable_evaluate=False,
            enable_initial_evaluate=False,
            work_dir=work_dir,
            checkpoint_interval=checkpoint_interval,
            checkpoint_maxkeep=checkpoint_maxkeep,
            auto_resume=auto_resume,
            seed=42,
            debug_rollout=False,
        )

    def test_rl_train_with_sft(self):
        """Test train_controller save/resume with efficient_attn_ratio verification."""
        work_dir = Path(self.temp_dir) / "work_dir_sft"
        work_dir.mkdir(parents=True, exist_ok=True)

        # Build trainer to get train_controller
        trainer_cfg = self.build_trainer_config(
            work_dir=str(work_dir),
            checkpoint_interval=1,
            checkpoint_maxkeep=2,
            auto_resume=False,
        )
        trainer = trainer_cfg.build()
        train_controller = trainer.train_controller

        # Prepare synthetic data batches
        tokenizer = AutoTokenizer.from_pretrained(QWEN3_PATH, trust_remote_code=True)

        # Create simple prompts and responses
        prompts = ["What is 2+2?", "What is the capital of France?"]
        responses = [
            ["4", "Four", "2+2=4", "The answer is 4"],
            ["Paris", "The capital is Paris", "Paris, France", "It's Paris"]
        ]

        data_batches = []
        for prompt, response_list in zip(prompts, responses):
            prompt_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].flatten().tolist()
            rewards = torch.tensor([1.0, 0.8, 0.9, 0.7], dtype=torch.float32)
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            for i, response in enumerate(response_list):
                response_ids = tokenizer(response, return_tensors='pt')['input_ids'].flatten().tolist()
                # Align with RLColocateTrainer._prepare_train_data():
                # - input_ids excludes last token (usually eos) of response_ids
                # - shifted_labels aligns to input_ids length
                input_ids = prompt_ids + response_ids[:-1]
                shifted_labels = [-100] * (len(prompt_ids) - 1) + response_ids
                input_ids_tensor = torch.tensor(input_ids, dtype=torch.int64).unsqueeze(0)
                shifted_labels_tensor = torch.tensor(shifted_labels, dtype=torch.int64).unsqueeze(0)

                adv_val = advantages[i].item()
                # Controller._packing expects `advantage` as a list and will flatten it.
                # Keep the length consistent with shifted_labels/input_ids.
                advantage_list = [adv_val] * (len(prompt_ids) - 1) + [adv_val] * len(response_ids)

                data_batches.append(dict(
                    seq_ctx=SequenceContext.from_input_ids((input_ids_tensor,), device="cpu"),
                    shifted_labels=shifted_labels_tensor,
                    advantage=advantage_list,
                ))

        # RLColocateTrainer initializes by offloading train workers to CPU.
        # Align with RLColocateTrainer.fit() which onloads before training.
        train_controller.onload(target="all")

        # First fit and save
        train_controller.fit(data_batches, pack_max_length=1024, rollout_idx=0)
        checkpoint_path = str(work_dir / "save_test")
        train_controller.save(checkpoint_path, no_save_optimizer=True)

        # Second fit and collect metrics
        train_controller.onload(target="all")
        log_infos = train_controller.fit(data_batches, pack_max_length=1024, rollout_idx=1)
        efficient_attn_ratio_list = []
        for log_info in log_infos:
            efficient_attn_ratio_list.append(log_info['sft_train_metrics']['efficient_attn_ratio'])
        self.assertTrue(all([ratio > 0 for ratio in efficient_attn_ratio_list]))

        # Kill and rebuild
        del trainer
        ray.shutdown()
        # Re-init Ray with enough resources for AcceleratorResourcesConfig(num_workers=8, num_cpus_per_worker=4).
        ray.init(num_cpus=80, num_gpus=8, ignore_reinit_error=True)

        trainer_cfg = self.build_trainer_config(
            work_dir=str(work_dir),
            checkpoint_interval=1,
            checkpoint_maxkeep=2,
            auto_resume=False,
        )
        trainer = trainer_cfg.build()
        train_controller = trainer.train_controller

        # Resume and verify
        load_checkpoint_cfg = LoadCheckpointConfig(
            checkpoint_path=checkpoint_path,
            load_optimizer_states=False,
            load_optimizer_args=False
        )
        train_controller.resume(load_checkpoint_cfg)

        train_controller.onload(target="all")
        log_infos = train_controller.fit(data_batches, pack_max_length=1024, rollout_idx=1)
        new_efficient_attn_ratio_list = []
        for log_info in log_infos:
            new_efficient_attn_ratio_list.append(log_info['sft_train_metrics']['efficient_attn_ratio'])

        efficient_attn_ratio_list.sort()
        new_efficient_attn_ratio_list.sort()
        self.assertEqual(efficient_attn_ratio_list, new_efficient_attn_ratio_list)


if __name__ == "__main__":
    unittest.main()
