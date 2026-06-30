"""Two-step real Qwen3.5 MoE VLM async RL training check.

This test intentionally runs a real two-step ``RLColocateTrainer.fit()`` with
LMDeploy, Qwen3.5 35B-A3B VLM MoE, Geometry3K multimodal data, async replay
buffer, and async produce strategy.

Checks:
- train/infer mismatch metrics stay finite and below PR thresholds;
- each train step produces timing, batch-size, and async failure metrics;
- async completed/aborted group accounting matches produced samples plus
  previous completed leftovers;
- train weights are synchronized once per step;
- rollout states keep VLM fields, token logprobs, rewards, and response ids;
- trajectory artifacts contain two 64-sample train rollout dumps;
- response lengths are valid per step and diverse across the full smoke run.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any

# XTUNER_DETERMINISTIC is read during xtuner imports. Keep it disabled for
# this 2-step test because FA3 deterministic backward does not support hdim=256.
os.environ["XTUNER_DETERMINISTIC"] = "false"

import ray

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.datasets.rl_tokenize_fn import RLQwen3VLTokenizeFnConfig
from xtuner.v1.model import Qwen3_5_VLMoE35BA3Config
from xtuner.v1.rl.advantage import GRPOAdvantageConfig
from xtuner.v1.rl.agent_loop import SingleTurnAgentLoopConfig
from xtuner.v1.rl.agent_loop_manager import (
    AgentLoopManagerConfig,
    AsyncProduceStrategyConfig,
    ProduceBatchResult,
    SamplerConfig,
    TaskSpecConfig,
)
from xtuner.v1.rl.judger import GEO3KJudgerConfig
from xtuner.v1.rl.loss import GRPOLossConfig
from xtuner.v1.rl.replay_buffer import AsyncReplayBufferConfig
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.trainer import RolloutImportanceSampling, WorkerConfig
from xtuner.v1.rl.utils import AcceleratorResourcesConfig, CPUResourcesConfig
from xtuner.v1.train.rl_trainer import RLColocateTrainerConfig


MODEL_PATH = Path(os.environ["QWEN3_5_MOE_PATH"])
MEDIA_ROOT = Path(os.environ["GEO3K_MEDIA_ROOT"])
DATA_PATH = Path(os.environ["GEO3K_LONGTAIL_DATA_PATH"])
BACKEND_LMDEPLOY = os.environ["XTUNER_USE_LMDEPLOY"]

EXPERIMENT_NAME = "qwen35_vl_moe_async_train_2step"

TRAIN_BATCH_SIZE = 32
PROMPT_REPEAT_K = 2
EXPECTED_TRAIN_SAMPLES = TRAIN_BATCH_SIZE * PROMPT_REPEAT_K
TOTAL_TRAIN_STEPS = 2
MAX_PROMPT_LENGTH = 4096
MAX_RESPONSE_LENGTH = 2048
PACK_MAX_LENGTH = 8192
MISMATCH_KL_MAX = 0.005
MISMATCH_K3_KL_MAX = 0.005

REQUIRED_STEP_METRICS = (
    "mismatch/mismatch_kl",
    "mismatch/mismatch_k3_kl",
    "response/batch_size",
    "response/response_len/min",
    "response/response_len/max",
    "time/produce_batch",
    "time/training",
    "timing/task_n",
    "timing/pause_s",
    "async/failed_samples",
    "async/filtered_samples",
    "async/expired_samples",
)


class TestQwen35VLMoEAsyncTrain2Step(unittest.TestCase):
    def setUp(self):
        if BACKEND_LMDEPLOY != "1":
            raise RuntimeError("XTUNER_USE_LMDEPLOY=1 is required for Qwen3.5 VLM MoE async train 2-step.")
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"QWEN3_5_MOE_PATH does not exist: {MODEL_PATH}")
        if not MEDIA_ROOT.exists():
            raise FileNotFoundError(f"GEO3K_MEDIA_ROOT does not exist: {MEDIA_ROOT}")
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Long-tail training dataset does not exist: {DATA_PATH}")

        self.temp_dir = tempfile.TemporaryDirectory(
            prefix=f"{EXPERIMENT_NAME}_{time.strftime('%Y%m%d%H%M%S')}_{os.getpid()}_",
        )
        self.addCleanup(self.temp_dir.cleanup)
        self.temp_dir_path = Path(self.temp_dir.name)
        print(f"qwen35 vl moe async train 2-step temp dir: {self.temp_dir_path}")
        self.produce_calls: list[dict[str, Any]] = []
        self.produce_results: list[ProduceBatchResult] = []
        self.update_weight_calls = 0
        self._patch_env(
            {
                "XTUNER_USE_LMDEPLOY": "1",
                "XTUNER_USE_SGLANG": "0",
                "XTUNER_USE_VLLM": "0",
                "XTUNER_USE_FA3": "1",
                "XTUNER_DETERMINISTIC": "false",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            },
            unset=("RAY_ADDRESS",),
        )
        ray.init(address="local", num_cpus=128, num_gpus=8, ignore_reinit_error=True)

    def tearDown(self):
        if ray.is_initialized():
            ray.shutdown()
        if hasattr(self, "_old_env"):
            self._restore_env()

    def test_qwen35_vl_moe_async_train_2step_and_metrics(self):
        work_dir = self.temp_dir_path / "work_dir"
        work_dir.mkdir(parents=True, exist_ok=True)

        start_s = time.perf_counter()
        trainer = self.build_config(work_dir).build()
        self._record_produce_batch(trainer)
        self._record_update_weights(trainer)
        try:
            trainer.fit()
        finally:
            trainer._exp_tracker.close()
        elapsed_s = time.perf_counter() - start_s

        step_metrics = self._load_step_metrics(work_dir)
        self._assert_step_metrics(step_metrics)
        self._assert_async_produce_results()
        self._assert_weight_sync()
        self._assert_vlm_rollout_states()
        self._assert_trajectory_artifacts(work_dir)

        diagnostics = {
            "elapsed_s": round(elapsed_s, 3),
            "steps": step_metrics,
            "produce_calls": self.produce_calls,
        }
        print("qwen35 vl moe async train 2-step diagnostics:\n" + json.dumps(diagnostics, ensure_ascii=False, indent=2))

    def build_config(self, work_dir: Path) -> RLColocateTrainerConfig:
        resources = AcceleratorResourcesConfig(
            accelerator="GPU",
            num_workers=8,
            num_cpus_per_worker=12,
            cpu_memory_per_worker=16 * 1024**3,
        )
        rollout_config = RolloutConfig(
            env=EXPERIMENT_NAME,
            device=resources.accelerator,
            model_path=str(MODEL_PATH),
            tokenizer_path=str(MODEL_PATH),
            dtype="bfloat16",
            tensor_parallel_size=1,
            expert_parallel_size=2,
            gpu_memory_utilization=0.8,
            context_length=MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH,
            rollout_max_batch_size_per_instance=128,
            # Deprecated compatibility option; fixed rollout constants now
            # keep Ray/httpx from queueing before the inference engine.
            allow_over_concurrency_ratio=1.0,
            enable_return_routed_experts=True,
            extra_rollout_config={
                "lmdeploy_backend": "pytorch",
                "lmdeploy_log_level": "ERROR",
                "lmdeploy_uvicorn_log_level": "ERROR",
            },
        )

        model_cfg = Qwen3_5_VLMoE35BA3Config(freeze_vision=True, freeze_projector=True)
        optim_cfg = AdamWConfig(lr=1e-6, betas=(0.9, 0.999), max_grad_norm=1.0, weight_decay=0.1, foreach=False)
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
            rollout_is=RolloutImportanceSampling(
                rollout_is_level="token",
                rollout_is_mode="both",
                rollout_is_threshold=(5, 0.5),
                rollout_is_mask_threshold=(5, 0.5),
                rollout_is_veto_threshold=(20, 0),
            ),
        )
        lr_cfg = LRConfig(lr_type="constant", warmup_ratio=0, lr_min=1e-6)
        fsdp_cfg = FSDPConfig(torch_compile=False, cpu_offload=False, ep_size=1, fp32_lm_head=True)
        train_worker_cfg = WorkerConfig(
            model_cfg=model_cfg,
            load_from=str(MODEL_PATH),
            optim_cfg=optim_cfg,
            loss_cfg=loss_cfg,
            lr_cfg=lr_cfg,
            fsdp_cfg=fsdp_cfg,
            sp_size=1,
            optimizer_steps=8,
            pack_max_length=PACK_MAX_LENGTH,
        )

        dataloader_cfg = DataloaderConfig(
            dataset_config_list=[
                {
                    "dataset": DatasetConfig(
                        name=EXPERIMENT_NAME,
                        anno_path=DATA_PATH,
                        class_name="VLMJsonlDataset",
                        media_root=str(MEDIA_ROOT),
                    ),
                    "tokenize_fn": RLQwen3VLTokenizeFnConfig(
                        processor_path=str(MODEL_PATH),
                        max_length=MAX_PROMPT_LENGTH,
                        chat_template="qwen3.5-vl",
                        add_generation_prompt=True,
                        enable_thinking=True,
                    ),
                }
            ],
            pack_max_length=PACK_MAX_LENGTH,
            collator="fake_collator",
            pack_level="none",
        )
        agent_loop_manager_cfg = AgentLoopManagerConfig(
            tasks=[
                TaskSpecConfig(
                    task_name="geo3k_longtail",
                    agent_loop_config=SingleTurnAgentLoopConfig(
                        hf_checkpoint=str(MODEL_PATH),
                        sample_params=SampleParams(
                            max_tokens=MAX_RESPONSE_LENGTH,
                            top_k=0,
                            top_p=1.0,
                            temperature=0.0,
                            min_tokens=0,
                            return_logprob=True,
                            return_token_ids=True,
                            return_routed_experts=True,
                        ),
                    ),
                    judger_config=GEO3KJudgerConfig(
                        judger_name="hiyouga/geometry3k",
                        cpu_resources=CPUResourcesConfig(num_workers=1, num_cpus_per_worker=1),
                    ),
                    produce_strategy_config=AsyncProduceStrategyConfig(
                        over_sample_threshold=1.0,
                        enable_partial_rollout=True,
                        max_staleness=1,
                    ),
                    sampler_config=SamplerConfig(
                        dataloader_cfg=dataloader_cfg,
                        prompt_repeat_k=PROMPT_REPEAT_K,
                    ),
                )
            ],
        )

        return RLColocateTrainerConfig(
            resources=resources,
            train_worker_cfg=train_worker_cfg,
            rollout_config=rollout_config,
            tokenizer_path=str(MODEL_PATH),
            replay_buffer_config=AsyncReplayBufferConfig(),
            agent_loop_manager_cfg=agent_loop_manager_cfg,
            load_from=str(MODEL_PATH),
            total_train_steps=TOTAL_TRAIN_STEPS,
            train_batch_size=TRAIN_BATCH_SIZE,
            advantage_estimator_config=GRPOAdvantageConfig(eps=1e-8),
            sync_weights_interval=1,
            enable_evaluate=False,
            enable_initial_evaluate=False,
            evaluate_step=1,
            work_dir=str(work_dir),
            checkpoint_interval=-1,
            checkpoint_maxkeep=-1,
            hf_interval=-1,
            hf_max_keep=-1,
            seed=123,
            debug_rollout=False,
            exp_tracker="jsonl",
        )

    def _record_produce_batch(self, trainer) -> None:
        original_produce_batch = trainer.agent_loop_manager.produce_batch

        async def produce_batch_wrapper(batch_size: int, train_step: int, *, model_step: int):
            result = await original_produce_batch(batch_size, train_step, model_step=model_step)
            self.produce_calls.append(
                {
                    "batch_size": batch_size,
                    "train_step": train_step,
                    "model_step": model_step,
                    "produced_samples": result.produced_samples,
                    "rollout_groups": len(result.rollout_states),
                    "leftover_completed": result.leftover_completed,
                    "leftover_aborted": result.leftover_aborted,
                    "failed_samples": result.failed_samples,
                    "filtered_samples": result.filtered_samples,
                    "leftover_expired": result.leftover_expired,
                }
            )
            self.produce_results.append(result)
            return result

        trainer.agent_loop_manager.produce_batch = produce_batch_wrapper

    def _record_update_weights(self, trainer) -> None:
        original_update_weights = trainer.train_controller.update_weights

        def update_weights_wrapper(*args, **kwargs):
            self.update_weight_calls += 1
            return original_update_weights(*args, **kwargs)

        trainer.train_controller.update_weights = update_weights_wrapper

    def _load_step_metrics(self, work_dir: Path) -> list[dict[str, float]]:
        rows: list[dict[str, float]] = []
        metrics_files = sorted(work_dir.rglob("tracker.jsonl"))
        for metrics_path in metrics_files:
            with metrics_path.open(encoding="utf-8") as file:
                for line in file:
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if all(key in payload for key in REQUIRED_STEP_METRICS):
                        row = {"step": int(payload["step"])}
                        for key in REQUIRED_STEP_METRICS:
                            row[key] = float(payload[key])
                        rows.append(row)

        rows.sort(key=lambda item: int(item["step"]))
        actual_steps = [int(row["step"]) for row in rows]
        self.assertEqual(actual_steps, [1, 2], f"unexpected tracker steps from {metrics_files}")
        return rows

    def _assert_step_metrics(self, step_metrics: list[dict[str, float]]) -> None:
        for row in step_metrics:
            step = int(row["step"])
            mismatch_kl = row["mismatch/mismatch_kl"]
            mismatch_k3_kl = row["mismatch/mismatch_k3_kl"]
            self.assertTrue(math.isfinite(mismatch_kl), f"mismatch_kl is not finite at step {step}")
            self.assertTrue(math.isfinite(mismatch_k3_kl), f"mismatch_k3_kl is not finite at step {step}")
            self.assertLess(mismatch_kl, MISMATCH_KL_MAX, f"mismatch_kl too high at step {step}")
            self.assertLess(mismatch_k3_kl, MISMATCH_K3_KL_MAX, f"mismatch_k3_kl too high at step {step}")
            self.assertEqual(int(row["response/batch_size"]), EXPECTED_TRAIN_SAMPLES)
            self.assertEqual(int(row["timing/task_n"]), TRAIN_BATCH_SIZE)
            self.assertGreater(row["time/produce_batch"], 0.0)
            self.assertGreater(row["time/training"], 0.0)
            self.assertGreaterEqual(row["timing/pause_s"], 0.0)
            self.assertEqual(int(row["async/failed_samples"]), 0)
            self.assertEqual(int(row["async/filtered_samples"]), 0)
            self.assertEqual(int(row["async/expired_samples"]), 0)
            self.assertGreater(row["response/response_len/min"], 0.0)
            self.assertGreaterEqual(row["response/response_len/max"], row["response/response_len/min"])

    def _assert_async_produce_results(self) -> None:
        self.assertEqual(len(self.produce_results), TOTAL_TRAIN_STEPS)
        previous_leftover_completed_groups = 0
        for step_idx, result in enumerate(self.produce_results, start=1):
            self.assertEqual(len(result.rollout_states), TRAIN_BATCH_SIZE, f"step {step_idx}")
            self.assertEqual(result.produced_samples % PROMPT_REPEAT_K, 0, f"step {step_idx}")
            sent_groups = result.produced_samples // PROMPT_REPEAT_K
            completed_groups = len(result.rollout_states) + result.leftover_completed
            aborted_groups = result.leftover_aborted
            self.assertEqual(
                completed_groups + aborted_groups,
                sent_groups + previous_leftover_completed_groups,
                f"step {step_idx}",
            )
            self.assertEqual(result.failed_samples, 0, f"step {step_idx}")
            self.assertEqual(result.filtered_samples, 0, f"step {step_idx}")
            self.assertEqual(result.leftover_expired, 0, f"step {step_idx}")
            previous_leftover_completed_groups = result.leftover_completed
            for group in result.rollout_states:
                self.assertEqual(len(group), PROMPT_REPEAT_K, f"step {step_idx}")

    def _assert_weight_sync(self) -> None:
        self.assertEqual(self.update_weight_calls, TOTAL_TRAIN_STEPS - 1)
        self.assertEqual([call["model_step"] for call in self.produce_calls], [0, 1])

    def _assert_vlm_rollout_states(self) -> None:
        for result in self.produce_results:
            for group in result.rollout_states:
                for sample in group:
                    self.assertIsInstance(sample, RolloutState)
                    self.assertIsNotNone(sample.mm_info)
                    self.assertIn("train_prompt_ids", sample.extra_fields)
                    self.assertIn("image_data", sample.extra_fields)
                    self.assertTrue(sample.response_ids)
                    self.assertIsNotNone(sample.logprobs)
                    self.assertEqual(len(sample.logprobs), len(sample.response_ids))
                    self.assertIsNotNone(sample.reward)
                    self.assertIn("score", sample.reward)

    def _assert_trajectory_artifacts(self, work_dir: Path) -> None:
        trajectory_files = sorted(work_dir.rglob("train_rollout_*.jsonl"))
        self.assertEqual(len(trajectory_files), TOTAL_TRAIN_STEPS)
        self.assertEqual([path.name for path in trajectory_files], ["train_rollout_1.jsonl", "train_rollout_2.jsonl"])

        response_lens: list[int] = []
        finish_reasons: list[str | None] = []
        for path in trajectory_files:
            objects = self._load_trajectory_objects(path)
            summary = objects[0]
            rows = objects[1:]
            self.assertEqual(summary["total_len"], EXPECTED_TRAIN_SAMPLES, path)
            self.assertEqual(len(rows), EXPECTED_TRAIN_SAMPLES, path)
            for row in rows:
                self.assertTrue(row["response"])
                self.assertGreater(row["response_len"], 0)
                self.assertIn("reward", row)
                response_lens.append(int(row["response_len"]))
                finish_reasons.append(row.get("finish_reason"))

        self.assertGreater(max(response_lens), min(response_lens), "expected response length diversity across rollout artifacts")
        self.assertIn("stop", finish_reasons, "expected at least one rollout sample to stop before max_tokens")

    def _load_trajectory_objects(self, path: Path) -> list[dict[str, Any]]:
        text = path.read_text(encoding="utf-8")
        decoder = json.JSONDecoder()
        objects: list[dict[str, Any]] = []
        pos = 0
        while pos < len(text):
            while pos < len(text) and text[pos].isspace():
                pos += 1
            if pos >= len(text):
                break
            item, pos = decoder.raw_decode(text, pos)
            objects.append(item)
        self.assertTrue(objects, path)
        return objects

    def _patch_env(self, updates: dict[str, str], unset: tuple[str, ...] = ()) -> None:
        keys = set(updates) | set(unset)
        self._old_env = {key: os.environ.get(key) for key in keys}
        for key in unset:
            os.environ.pop(key, None)
        os.environ.update(updates)

    def _restore_env(self) -> None:
        for key, value in self._old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
