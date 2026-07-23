"""Real Qwen3.5 VLM MoE async-HF recovery E2E test.

This test focuses only on the recovery protocol:

1. train step 1 starts an asynchronous recovery-HF export;
2. the test waits until that export has been published as ready;
3. while train step 2 rollout is running, rank 0's backend is crashed;
4. RolloutHealthManager detects the failure and reloads the ready HF;
5. train step 2 and the post-recovery train step 3 both complete.

Run in the same 8-GPU environment used by the Qwen3.5 VLM MoE
async-training E2E test.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
import unittest
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Callable


# These values are consumed while XTuner modules are imported.
os.environ["XTUNER_DETERMINISTIC"] = "false"
os.environ["XTUNER_USE_LMDEPLOY"] = "1"
os.environ["XTUNER_USE_SGLANG"] = "0"
os.environ["XTUNER_USE_VLLM"] = "0"
os.environ["XTUNER_TEST_IMMEDIATE_RECOVERY"] = "1"

import ray

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.data_proto.rl_data import SampleParams
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.datasets.rl_tokenize_fn import RLQwen3VLTokenizeFnConfig
from xtuner.v1.model import Qwen3_5_VLMoE35BA3Config
from xtuner.v1.rl.advantage import GRPOAdvantageConfig
from xtuner.v1.rl.agent_loop import SingleTurnAgentLoopConfig
from xtuner.v1.rl.agent_loop_manager import (
    AgentLoopManagerConfig,
    AsyncProduceStrategyConfig,
    SamplerConfig,
    TaskSpecConfig,
)
from xtuner.v1.rl.judger import GEO3KJudgerConfig
from xtuner.v1.rl.loss import GRPOLossConfig
from xtuner.v1.rl.replay_buffer import AsyncReplayBufferConfig
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.rollout.worker_registry import WorkerLifecycleState
from xtuner.v1.rl.trainer import RolloutImportanceSampling, WorkerConfig
from xtuner.v1.rl.utils import AcceleratorResourcesConfig, CPUResourcesConfig
from xtuner.v1.train.rl_trainer import RLColocateTrainerConfig


EXPERIMENT_NAME = "qwen35_vl_moe_async_hf_recovery_e2e"
TOTAL_TRAIN_STEPS = 3
TRAIN_BATCH_SIZE_BY_STEP = {1: 8, 2: 128, 3: 8}
PROMPT_REPEAT_K = 2
MAX_PROMPT_LENGTH = 4096
MAX_RESPONSE_LENGTH = 2048
PACK_MAX_LENGTH = 8192
RECOVERY_TIMEOUT_S = 600.0
RAY_GET_TIMEOUT_S = 600.0
POLL_INTERVAL_S = 0.5


class TestQwen35VLMoEAsyncHFRecoveryE2E(unittest.TestCase):
    def setUp(self) -> None:
        self.model_path = self._required_path("QWEN3_5_MOE_PATH")
        self.media_root = self._required_path("GEO3K_MEDIA_ROOT")
        self.data_path = self._required_path("GEO3K_LONGTAIL_DATA_PATH")

        default_work_dir = (
            Path.cwd() / "work_dirs" / f"{EXPERIMENT_NAME}_{time.strftime('%Y%m%d%H%M%S')}_{os.getpid()}"
        )
        self.work_dir = Path(os.environ.get("WORK_DIR", str(default_work_dir)))
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self._events: list[str] = []
        self._events_lock = threading.Lock()
        self._pending_export_captured = threading.Event()
        self._rollout_step_2_started = threading.Event()
        self._rollout_step_2_finished = threading.Event()
        self._recovery_finished = threading.Event()
        self._first_recovery_hf_export: Future[Path | None] | None = None
        self._fault_injection_error: Exception | None = None
        self._recovery_hf_observation: dict[str, Any] | None = None
        self._rank_0_lifecycle_states: list[str] = []
        self._produce_calls: list[dict[str, int]] = []

        self._patch_env(
            {
                "XTUNER_USE_LMDEPLOY": "1",
                "XTUNER_USE_SGLANG": "0",
                "XTUNER_USE_VLLM": "0",
                "XTUNER_USE_FA3": "1",
                "XTUNER_DETERMINISTIC": "false",
                "XTUNER_TEST_IMMEDIATE_RECOVERY": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            },
            unset=("RAY_ADDRESS",),
        )
        ray.init(address="local", num_cpus=128, num_gpus=8, ignore_reinit_error=True)

    def tearDown(self) -> None:
        if ray.is_initialized():
            ray.shutdown()
        if hasattr(self, "_old_env"):
            self._restore_env()

    def test_async_hf_save_and_backend_failure_recovery(self) -> None:
        trainer = self._build_config().build()
        self._install_rollout_probe(trainer)
        self._install_async_hf_probe(trainer)

        fault_injection_thread = threading.Thread(
            target=self._inject_failure_after_recovery_hf_ready,
            args=(trainer,),
            name="async-hf-recovery-fault-injector",
            daemon=True,
        )
        fault_injection_thread.start()

        try:
            trainer.fit()
        finally:
            fault_injection_thread.join(timeout=10)

        self.assertFalse(fault_injection_thread.is_alive(), "Fault-injection coordinator did not exit.")
        if self._fault_injection_error is not None:
            raise AssertionError("Fault-injection coordinator failed.") from self._fault_injection_error

        observation = self._recovery_hf_observation
        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertTrue(observation["path_existed"])
        self.assertGreater(observation["file_count"], 0)
        unavailable_states = {
            WorkerLifecycleState.INACTIVE.value,
            WorkerLifecycleState.RECOVERING.value,
        }
        self.assertTrue(unavailable_states.intersection(self._rank_0_lifecycle_states))
        self.assertEqual(self._rank_0_lifecycle_states[-1], WorkerLifecycleState.ACTIVE.value)
        self.assertEqual(
            [call["train_step"] for call in self._produce_calls],
            [1, 2, 3],
        )
        self.assertEqual(
            [call["batch_size"] for call in self._produce_calls],
            [TRAIN_BATCH_SIZE_BY_STEP[step] for step in range(1, TOTAL_TRAIN_STEPS + 1)],
        )
        self._assert_recovery_event_order()

    def _install_rollout_probe(self, trainer: Any) -> None:
        original_produce_batch = trainer.agent_loop_manager.produce_batch

        async def produce_batch_wrapper(batch_size: int, train_step: int, *, model_step: int) -> Any:
            batch_size = TRAIN_BATCH_SIZE_BY_STEP.get(train_step, batch_size)
            self._record_event(f"rollout_{train_step}_started")
            if train_step == 2:
                self._rollout_step_2_started.set()

            try:
                result = await original_produce_batch(batch_size, train_step, model_step=model_step)
                self._produce_calls.append(
                    {
                        "batch_size": batch_size,
                        "train_step": train_step,
                        "model_step": model_step,
                    }
                )
                if train_step == 2:
                    recovered = await asyncio.to_thread(
                        self._recovery_finished.wait,
                        RECOVERY_TIMEOUT_S,
                    )
                    if not recovered:
                        raise TimeoutError("Timed out waiting for rank 0 recovery during train step 2 rollout.")
                return result
            finally:
                if train_step == 2:
                    self._rollout_step_2_finished.set()
                self._record_event(f"rollout_{train_step}_finished")

        trainer.agent_loop_manager.produce_batch = produce_batch_wrapper

    def _install_async_hf_probe(self, trainer: Any) -> None:
        original_maybe_save_recovery_hf = trainer._maybe_save_recovery_hf

        def maybe_save_recovery_hf_wrapper(cur_step: int) -> None:
            original_maybe_save_recovery_hf(cur_step)
            if cur_step != 1:
                return

            pending_export = trainer._pending_hf_export
            if pending_export is None:
                raise AssertionError("Train step 1 did not schedule an asynchronous recovery-HF export.")
            self._first_recovery_hf_export = pending_export
            self._record_event("async_hf_1_scheduled")
            self._pending_export_captured.set()

        trainer._maybe_save_recovery_hf = maybe_save_recovery_hf_wrapper

    def _inject_failure_after_recovery_hf_ready(self, trainer: Any) -> None:
        try:
            if not self._pending_export_captured.wait(timeout=RECOVERY_TIMEOUT_S):
                raise TimeoutError("Timed out waiting for the train step 1 recovery-HF export to be scheduled.")

            pending_export = self._first_recovery_hf_export
            if pending_export is None:
                raise AssertionError("Recovery-HF export event was set without a Future.")
            recovery_hf_path = pending_export.result(timeout=RECOVERY_TIMEOUT_S)
            if recovery_hf_path is None:
                raise RuntimeError("Train step 1 recovery-HF export failed.")

            self._recovery_hf_observation = {
                "path": str(recovery_hf_path),
                "path_existed": recovery_hf_path.is_dir(),
                "file_count": sum(1 for path in recovery_hf_path.rglob("*") if path.is_file()),
            }
            self._record_event("async_hf_1_ready")

            if not self._rollout_step_2_started.wait(timeout=RECOVERY_TIMEOUT_S):
                raise TimeoutError("Timed out waiting for train step 2 rollout to start.")
            if self._rollout_step_2_finished.is_set():
                raise RuntimeError("Train step 2 rollout finished before backend failure injection.")

            initial_state = self._get_rank_0_lifecycle_state(trainer)
            if initial_state != WorkerLifecycleState.ACTIVE.value:
                raise RuntimeError(f"Rank 0 was not active before fault injection: state={initial_state}.")
            self._record_rank_0_state(initial_state)

            ray.get(
                trainer.rollout_controller.inject_backend_crash_for_test.remote(rank=0),
                timeout=RAY_GET_TIMEOUT_S,
            )
            self._record_event("backend_crash_injected")

            self._wait_for_rank_0_state(
                trainer,
                expected=lambda state: state != WorkerLifecycleState.ACTIVE.value,
                description="become inactive",
            )
            self._record_event("rank_0_unavailable")
            self._wait_for_rank_0_state(
                trainer,
                expected=lambda state: state == WorkerLifecycleState.ACTIVE.value,
                description="recover to active",
            )
            self._record_event("rank_0_recovered")
        except Exception as error:
            self._fault_injection_error = error
        finally:
            self._recovery_finished.set()

    def _wait_for_rank_0_state(
        self,
        trainer: Any,
        *,
        expected: Callable[[str], bool],
        description: str,
    ) -> str:
        deadline = time.monotonic() + RECOVERY_TIMEOUT_S
        while time.monotonic() < deadline:
            state = self._get_rank_0_lifecycle_state(trainer)
            self._record_rank_0_state(state)
            if expected(state):
                return state
            time.sleep(POLL_INTERVAL_S)
        raise TimeoutError(
            f"Timed out waiting for rank 0 to {description}; observed states={self._rank_0_lifecycle_states}."
        )

    @staticmethod
    def _get_rank_0_lifecycle_state(trainer: Any) -> str:
        targets = ray.get(
            trainer.rollout_controller.get_weight_update_targets.remote(),
            timeout=RAY_GET_TIMEOUT_S,
        )
        for target in targets:
            if target.endpoint_rank == 0:
                return target.lifecycle_state
        raise RuntimeError(f"Rank 0 weight-update target was not found: targets={targets}.")

    def _record_rank_0_state(self, state: str) -> None:
        if not self._rank_0_lifecycle_states or self._rank_0_lifecycle_states[-1] != state:
            self._rank_0_lifecycle_states.append(state)

    def _assert_recovery_event_order(self) -> None:
        required_events = (
            "async_hf_1_scheduled",
            "async_hf_1_ready",
            "rollout_2_started",
            "backend_crash_injected",
            "rank_0_unavailable",
            "rank_0_recovered",
            "rollout_2_finished",
            "rollout_3_started",
            "rollout_3_finished",
        )
        for event in required_events:
            self.assertEqual(self._events.count(event), 1, f"Unexpected event count for {event}: {self._events}")

        positions = {event: self._events.index(event) for event in required_events}
        ordered_pairs = (
            ("async_hf_1_scheduled", "async_hf_1_ready"),
            ("async_hf_1_ready", "backend_crash_injected"),
            ("rollout_2_started", "backend_crash_injected"),
            ("backend_crash_injected", "rank_0_unavailable"),
            ("rank_0_unavailable", "rank_0_recovered"),
            ("rank_0_recovered", "rollout_2_finished"),
            ("rollout_2_finished", "rollout_3_started"),
            ("rollout_3_started", "rollout_3_finished"),
        )
        for first, second in ordered_pairs:
            self.assertLess(positions[first], positions[second], f"Expected {first} before {second}: {self._events}")

    def _record_event(self, event: str) -> None:
        with self._events_lock:
            self._events.append(event)

    def _build_config(self) -> RLColocateTrainerConfig:
        resources = AcceleratorResourcesConfig(
            accelerator="GPU",
            num_workers=8,
            num_cpus_per_worker=12,
            cpu_memory_per_worker=16 * 1024**3,
        )
        rollout_config = RolloutConfig(
            env=EXPERIMENT_NAME,
            device=resources.accelerator,
            model_path=str(self.model_path),
            tokenizer_path=str(self.model_path),
            dtype="bfloat16",
            tensor_parallel_size=1,
            expert_parallel_size=2,
            gpu_memory_utilization=0.8,
            context_length=MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH,
            rollout_max_batch_size_per_instance=128,
            allow_over_concurrency_ratio=1.0,
            enable_return_routed_experts=True,
            health_check_interval_seconds=5.0,
            health_check_failure_threshold=1,
            extra_rollout_config={
                "lmdeploy_backend": "pytorch",
                "lmdeploy_log_level": "ERROR",
                "lmdeploy_uvicorn_log_level": "ERROR",
            },
        )

        train_worker_cfg = WorkerConfig(
            model_cfg=Qwen3_5_VLMoE35BA3Config(freeze_vision=True, freeze_projector=True),
            load_from=str(self.model_path),
            optim_cfg=AdamWConfig(
                lr=1e-6,
                betas=(0.9, 0.999),
                max_grad_norm=1.0,
                weight_decay=0.1,
                foreach=False,
            ),
            loss_cfg=GRPOLossConfig(
                policy_loss_cfg={
                    "cliprange_high": 0.28,
                    "cliprange_low": 0.2,
                    "loss_type": "vanilla",
                    "clip_ratio_c": 10.0,
                    "log_prob_diff_min": -20,
                    "log_prob_diff_max": 20,
                },
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
            ),
            lr_cfg=LRConfig(lr_type="constant", warmup_ratio=0, lr_min=1e-6),
            fsdp_cfg=FSDPConfig(torch_compile=False, cpu_offload=False, ep_size=1, fp32_lm_head=True),
            sp_size=1,
            optimizer_steps=8,
            pack_max_length=PACK_MAX_LENGTH,
        )

        dataloader_cfg = DataloaderConfig(
            dataset_config_list=[
                {
                    "dataset": DatasetConfig(
                        name=EXPERIMENT_NAME,
                        anno_path=self.data_path,
                        class_name="VLMJsonlDataset",
                        media_root=str(self.media_root),
                    ),
                    "tokenize_fn": RLQwen3VLTokenizeFnConfig(
                        processor_path=str(self.model_path),
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
                        hf_checkpoint=str(self.model_path),
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
            tokenizer_path=str(self.model_path),
            replay_buffer_config=AsyncReplayBufferConfig(),
            agent_loop_manager_cfg=agent_loop_manager_cfg,
            load_from=str(self.model_path),
            total_train_steps=TOTAL_TRAIN_STEPS,
            train_batch_size=TRAIN_BATCH_SIZE_BY_STEP[1],
            advantage_estimator_config=GRPOAdvantageConfig(eps=1e-8),
            sync_weights_interval=1,
            enable_evaluate=False,
            enable_initial_evaluate=False,
            evaluate_step=1,
            work_dir=str(self.work_dir),
            checkpoint_interval=-1,
            checkpoint_maxkeep=-1,
            hf_interval=-1,
            hf_max_keep=-1,
            enable_immediate_recovery=True,
            seed=123,
            debug_rollout=False,
            exp_tracker="jsonl",
        )

    @staticmethod
    def _required_path(env_name: str) -> Path:
        value = os.environ.get(env_name)
        if not value:
            raise RuntimeError(f"{env_name} must be set for the async-HF recovery E2E test.")
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"{env_name} does not exist: {path}")
        return path

    def _patch_env(self, updates: dict[str, str], *, unset: tuple[str, ...] = ()) -> None:
        keys = set(updates) | set(unset)
        self._old_env = {key: os.environ.get(key) for key in keys}
        for key, value in updates.items():
            os.environ[key] = value
        for key in unset:
            os.environ.pop(key, None)

    def _restore_env(self) -> None:
        for key, value in self._old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
