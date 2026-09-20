"""Real Qwen3.5 VLM MoE checkpoint-engine recovery E2E test.

This test focuses only on the recovery protocol:

1. train step 1 registers and broadcasts a checkpoint-engine weight update;
2. while train step 2 rollout is running, rank 0's backend is crashed;
3. RolloutHealthManager restarts the worker into pending_weight_update;
4. the train step 2 checkpoint-engine sync updates the pending worker;
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
from pathlib import Path
from typing import Any, Callable

import ray

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.data_proto.rl_data import SampleParams
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.datasets.rl_tokenize_fn import RLQwen3VLTokenizeFnConfig
from xtuner.v1.model import Qwen3_5_VLMoE35BA3Config
from xtuner.v1.module.mtp import MTPConfig
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
from xtuner.v1.utils import get_logger


EXPERIMENT_NAME = "qwen35_vl_moe_checkpoint_engine_recovery_e2e"
TOTAL_TRAIN_STEPS = 3
TRAIN_BATCH_SIZE_BY_STEP = {1: 8, 2: 256, 3: 8}
PROMPT_REPEAT_K = 2
MAX_PROMPT_LENGTH = 4096
MAX_RESPONSE_LENGTH = 2048
PACK_MAX_LENGTH = 8192
RECOVERY_TIMEOUT_S = 600.0
RAY_GET_TIMEOUT_S = 600.0
POLL_INTERVAL_S = 0.5
logger = get_logger()


class TestQwen35VLMoECheckpointEngineRecoveryE2E(unittest.TestCase):
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
        self._step_1_weight_update_finished = threading.Event()
        self._rollout_step_2_started = threading.Event()
        self._rollout_step_2_finished = threading.Event()
        self._rank_0_pending_weight_update = threading.Event()
        self._recovery_finished = threading.Event()
        self._fault_injection_error: Exception | None = None
        self._rank_0_lifecycle_states: list[str] = []
        self._produce_calls: list[dict[str, int]] = []
        self._weight_update_calls: list[dict[str, int | bool]] = []

        self._patch_env(
            {
                "XTUNER_USE_LMDEPLOY": "0",
                "XTUNER_USE_SGLANG": "1",
                "XTUNER_USE_VLLM": "0",
                "XTUNER_USE_FA3": "1",
                "XTUNER_DETERMINISTIC": "false",
                "XTUNER_TEST_IMMEDIATE_RECOVERY": "1",
            },
            unset=("RAY_ADDRESS","PYTORCH_CUDA_ALLOC_CONF"),
        )
        ray.init(address="local", num_cpus=256, num_gpus=8, ignore_reinit_error=True)

    def tearDown(self) -> None:
        if ray.is_initialized():
            ray.shutdown()
        if hasattr(self, "_old_env"):
            self._restore_env()

    @unittest.skipIf(os.environ.get("XTUNER_USE_SGLANG", "0") == "0", "sglang backend is not enabled")
    def test_checkpoint_engine_backend_failure_recovery(self) -> None:
        trainer = self._build_config().build()
        self._install_rollout_probe(trainer)
        self._install_checkpoint_engine_probe(trainer)

        fault_injection_thread = threading.Thread(
            target=self._inject_failure_after_checkpoint_engine_ready,
            args=(trainer,),
            name="checkpoint-engine-recovery-fault-injector",
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

        unavailable_states = {
            WorkerLifecycleState.INACTIVE.value,
            WorkerLifecycleState.PENDING_WEIGHTS.value,
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
        self.assertEqual(
            [call["train_step"] for call in self._weight_update_calls],
            [1, 2],
        )
        self.assertTrue(all(call["weights_synced"] for call in self._weight_update_calls))
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
                    pending_weight_update = await asyncio.to_thread(
                        self._rank_0_pending_weight_update.wait,
                        RECOVERY_TIMEOUT_S,
                    )
                    if not pending_weight_update:
                        raise TimeoutError(
                            "Timed out waiting for rank 0 to restart into pending_weight_update during train step 2 "
                            "rollout."
                        )
                return result
            finally:
                if train_step == 2:
                    self._rollout_step_2_finished.set()
                self._record_event(f"rollout_{train_step}_finished")

        trainer.agent_loop_manager.produce_batch = produce_batch_wrapper

    def _install_checkpoint_engine_probe(self, trainer: Any) -> None:
        original_sync_weights_and_save = trainer._sync_weights_and_save

        def sync_weights_and_save_wrapper(train_step: int, step_timer_dict: dict) -> bool:
            logger.info(f"[recovery-test] sync_weights_and_save enter train_step={train_step}")
            weights_synced = original_sync_weights_and_save(train_step, step_timer_dict)
            logger.info(
                f"[recovery-test] sync_weights_and_save exit train_step={train_step} "
                f"weights_synced={weights_synced}"
            )
            if weights_synced:
                has_registered_checkpoint = trainer.train_controller.has_registered_weight_checkpoint()
                self._weight_update_calls.append(
                    {
                        "train_step": train_step,
                        "weights_synced": weights_synced,
                        "has_registered_checkpoint": has_registered_checkpoint,
                    }
                )
                self._record_event(f"checkpoint_engine_{train_step}_updated")
                if train_step == 1:
                    if not has_registered_checkpoint:
                        raise AssertionError("Train step 1 did not register a checkpoint-engine checkpoint.")
                    rank_0_state = self._get_rank_0_lifecycle_state(trainer)
                    logger.info(
                        f"[recovery-test] set step_1_weight_update_finished "
                        f"rank0_state={rank_0_state}"
                    )
                    self._step_1_weight_update_finished.set()
            return weights_synced

        trainer._sync_weights_and_save = sync_weights_and_save_wrapper

    def _inject_failure_after_checkpoint_engine_ready(self, trainer: Any) -> None:
        try:
            logger.info("[recovery-test] fault injector waiting for step 1 checkpoint-engine update")
            if not self._step_1_weight_update_finished.wait(timeout=RECOVERY_TIMEOUT_S):
                raise TimeoutError("Timed out waiting for the train step 1 checkpoint-engine update.")

            logger.info("[recovery-test] fault injector waiting for train step 2 rollout start")
            if not self._rollout_step_2_started.wait(timeout=RECOVERY_TIMEOUT_S):
                raise TimeoutError("Timed out waiting for train step 2 rollout to start.")
            if self._rollout_step_2_finished.is_set():
                raise RuntimeError("Train step 2 rollout finished before backend failure injection.")

            initial_state = self._get_rank_0_lifecycle_state(trainer)
            if initial_state != WorkerLifecycleState.ACTIVE.value:
                raise RuntimeError(f"Rank 0 was not active before fault injection: state={initial_state}.")
            self._record_rank_0_state(initial_state)
            logger.info(f"[recovery-test] before backend crash injection rank0_state={initial_state}")

            ray.get(
                trainer.rollout_controller.inject_backend_crash_for_test.remote(rank=0),
                timeout=RAY_GET_TIMEOUT_S,
            )
            logger.info("[recovery-test] backend crash injection returned")
            self._record_event("backend_crash_injected")

            self._wait_for_rank_0_state(
                trainer,
                expected=lambda state: state != WorkerLifecycleState.ACTIVE.value,
                description="become inactive",
            )
            self._record_event("rank_0_unavailable")
            self._wait_for_rank_0_state(
                trainer,
                expected=lambda state: state == WorkerLifecycleState.PENDING_WEIGHTS.value,
                description="wait for checkpoint-engine weights",
            )
            self._record_event("rank_0_pending_weight_update")
            self._rank_0_pending_weight_update.set()
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
            "checkpoint_engine_1_updated",
            "rollout_2_started",
            "backend_crash_injected",
            "rank_0_unavailable",
            "rank_0_pending_weight_update",
            "checkpoint_engine_2_updated",
            "rank_0_recovered",
            "rollout_2_finished",
            "rollout_3_started",
            "rollout_3_finished",
        )
        for event in required_events:
            self.assertEqual(self._events.count(event), 1, f"Unexpected event count for {event}: {self._events}")

        positions = {event: self._events.index(event) for event in required_events}
        ordered_pairs = (
            ("checkpoint_engine_1_updated", "backend_crash_injected"),
            ("rollout_2_started", "backend_crash_injected"),
            ("backend_crash_injected", "rank_0_unavailable"),
            ("rank_0_unavailable", "rank_0_pending_weight_update"),
            ("rank_0_pending_weight_update", "rank_0_recovered"),
            ("rank_0_recovered", "rollout_2_finished"),
            ("rollout_2_finished", "checkpoint_engine_2_updated"),
            ("checkpoint_engine_2_updated", "rollout_3_started"),
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
            cpu_memory_per_worker=24 * 1024**3,
        )
        rollout_config = RolloutConfig(
            env=EXPERIMENT_NAME,
            device=resources.accelerator,
            model_path=str(self.model_path),
            tokenizer_path=str(self.model_path),
            dtype="bfloat16",
            tensor_parallel_size=1,
            expert_parallel_size=4,
            gpu_memory_utilization=0.8,
            context_length=MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH,
            rollout_max_batch_size_per_instance=128,
            allow_over_concurrency_ratio=1.0,
            enable_return_routed_experts=False,
            weight_transport_type="checkpoint_engine",
            skip_load_weights=True,
            checkpoint_name_prefix=EXPERIMENT_NAME,
            checkpoint_engine_timeout=RECOVERY_TIMEOUT_S,
            # 更快的发现错误并且重启
            health_check_interval_seconds=5.0,
            health_check_failure_threshold=1,
            extra_rollout_config={
                "sglang_log_level": "error",
            },
        )
        model_cfg = Qwen3_5_VLMoE35BA3Config(freeze_vision=True, freeze_projector=True)
        model_cfg.text_config.mtp_config = MTPConfig(num_layers=1)
        train_worker_cfg = WorkerConfig(
            model_cfg=model_cfg,
            load_from=str(self.model_path),
            optim_cfg=AdamWConfig(
                lr=1e-6,
                betas=(0.9, 0.999),
                max_grad_norm=1.0,
                weight_decay=0.1,
                foreach=False,
                swap_optimizer=True,
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
            fsdp_cfg=FSDPConfig(torch_compile=False, cpu_offload=False, ep_size=1, fp32_lm_head=False),
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
                            return_routed_experts=False,
                        ),
                    ),
                    judger_config=GEO3KJudgerConfig(
                        judger_name="hiyouga/geometry3k",
                        cpu_resources=CPUResourcesConfig(num_workers=1, num_cpus_per_worker=1),
                    ),
                    produce_strategy_config=AsyncProduceStrategyConfig(
                        over_sample_threshold=1.0,
                        enable_partial_rollout=False,
                        max_staleness=1,
                        max_pending_tasks=16,
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
            seed=123,
            debug_rollout=False,
            exp_tracker="jsonl",
        )

    @staticmethod
    def _required_path(env_name: str) -> Path:
        value = os.environ.get(env_name)
        if not value:
            raise RuntimeError(f"{env_name} must be set for the checkpoint-engine recovery E2E test.")
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
