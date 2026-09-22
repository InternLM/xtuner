"""Two-step OPD accuracy checks with in-test training launch.

Each test fully defines its OPD training config (Qwen3-VL-2B student +
GSM8K/Geo3K teachers), launches a real 2-step ``RLColocateTrainer.fit()``
in-process, and compares the step-1/2 metrics read from ``tracker.jsonl``
(``exp_tracker="jsonl"``) against the golden baselines below. Training
shape matches the reference 50-step runs (batch size 128, seed 1234,
1024/2048 sequence lengths), and data order is seed-deterministic, so the
2-step run must reproduce the first two steps of the reference runs
within per-tag tolerances.

Golden baselines (headline values; full precision in the constants below):

  topk (train-teacher forward_kl_topk):
    distillation/reduced_topk_opd_kl      step1=0.3049  step2=0.3286
    response/rewards/mean                 step1=0.3828  step2=0.3633
  mopd (sampled-token k1 rollout teachers):
    distillation/reduced_distillation_kl  step1=0.3442  step2=0.3924
    response/rewards/mean                 step1=0.4477  step2=0.4703

``TOPK_GOLDEN`` / ``MOPD_GOLDEN`` are self-contained constants measured
from the reference smoke runs with ``exp_tracker="tensorboard"``; metric
values are writer-independent, so reading them back from the jsonl
tracker is equivalent.

Requirements: 8 GPUs. Model paths come from the CI environment
(``QWEN3_VL_2B_PATH``, ``QWEN3_4B_PATH``, ``QWEN3_VL_DENSE_PATH``; see
``ci/scripts/CI_ENV.sh``) and LMDeploy comes from the image's
``PYTHONPATH``. The tests skip cleanly when these are unavailable.
Approximate wall time is
dominated by model loading / engine warmup rather than the 2 training
steps: topk ~10 min, mopd ~15-25 min (mopd additionally boots two
LMDeploy teacher HTTP servers on GPUs 7/6 while student workers use
GPUs 0-5, mirroring ``run_sampled_token_mopd.sh``).

Workflow:

    pytest tests/rl/test_opd_two_step_accuracy.py::TestOpdTopKTrainTeacherTwoStep -x -s
    pytest tests/rl/test_opd_two_step_accuracy.py::TestOpdMopdRolloutTeacherTwoStep -x -s
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from copy import deepcopy
from pathlib import Path

import ray
import torch

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.data_proto.rl_data import SampleParams
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.datasets.rl_tokenize_fn import RLQwen3VLTokenizeFnConfig
from xtuner.v1.model import Qwen3Dense4BConfig, Qwen3VLDense2BConfig, Qwen3VLDense4BConfig
from xtuner.v1.module.rope import RopeParametersConfig
from xtuner.v1.rl.advantage import GRPOAdvantageConfig
from xtuner.v1.rl.agent_loop import SingleTurnAgentLoopConfig
from xtuner.v1.rl.agent_loop_manager import (
    AgentLoopManagerConfig,
    SamplerConfig,
    SyncProduceStrategyConfig,
    TaskSpecConfig,
)
from xtuner.v1.rl.distillation import (
    DistillationConfig,
    RolloutTeacherConfig,
    RolloutTeacherLaunchConfig,
    TrainTeacherConfig,
)
from xtuner.v1.rl.judger import ComposedJudgerConfig, GEO3KJudgerConfig, GSM8KJudgerConfig
from xtuner.v1.rl.loss import DistillationLossConfig
from xtuner.v1.rl.replay_buffer import SyncReplayBufferConfig
from xtuner.v1.rl.rollout import trace_store
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.trainer import WorkerConfig
from xtuner.v1.rl.utils import AcceleratorResourcesConfig, CPUResourcesConfig
from xtuner.v1.train.rl_trainer import RLColocateTrainerConfig


REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_path(env_name: str) -> Path | None:
    value = os.environ.get(env_name)
    return Path(value) if value else None


# Model paths are provided by the CI environment (ci/scripts/CI_ENV.sh), the
# same variables other RL tests consume. LMDeploy comes from the image's
# PYTHONPATH; this file must stay free of machine-specific paths.
STUDENT_MODEL_PATH = _env_path("QWEN3_VL_2B_PATH")
GSM8K_TEACHER_MODEL_PATH = _env_path("QWEN3_4B_PATH")
GEO3K_TEACHER_MODEL_PATH = _env_path("QWEN3_VL_DENSE_PATH")
DATA_PATH = REPO_ROOT / "recipe/on_policy_distillation/data/gsm8k_geo3k_train.json"

TRAIN_BATCH_SIZE = 128
TOTAL_TRAIN_STEPS = 2
MAX_PROMPT_LENGTH = 1024
MAX_RESPONSE_LENGTH = 2048
PACK_MAX_LENGTH = MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH
SEED = 1234
TOP_K = 64
GSM8K_TEACHER_PORT = 13141
GEO3K_TEACHER_PORT = 13142
TEACHER_STARTUP_TIMEOUT_S = 1200


TOPK_GOLDEN = {
    "metrics": {
        "distillation/reduced_topk_opd_kl": {1: 0.30486780405044556, 2: 0.3286096453666687},
        "distillation/reduced_topk_opd_loss": {1: 0.30486780405044556, 2: 0.3286095857620239},
        "distillation/reduced_topk_opd_overlap_fraction": {1: 0.6676791906356812, 2: 0.6546056866645813},
        "distillation/reduced_topk_opd_student_selected_mass": {1: 0.9960947632789612, 2: 0.9966292381286621},
        "distillation/reduced_topk_opd_teacher_selected_mass": {1: 0.9986172318458557, 2: 0.999037504196167},
        "response/rewards/mean": {1: 0.3828125298023224, 2: 0.36328125},
    },
    "invariants": {"num_train_steps": 2, "response/batch_size": 128},
    "tolerance": {
        "default": {"rtol": 0.2, "atol": 0.0},
        "response/rewards/mean": {"rtol": 0.0, "atol": 0.15},
    },
}

MOPD_GOLDEN = {
    "metrics": {
        "distillation/reduced_distillation_kl": {1: 0.34421640634536743, 2: 0.3924480080604553},
        "distillation/reduced_distillation_abs_loss": {1: 0.4967445433139801, 2: 0.5339475274085999},
        "response/rewards/mean": {1: 0.4476562440395355, 2: 0.4703125059604645},
    },
    "invariants": {"num_train_steps": 2, "response/batch_size": 128},
    "tolerance": {
        "default": {"rtol": 0.2, "atol": 0.0},
        "response/rewards/mean": {"rtol": 0.0, "atol": 0.15},
    },
}


def read_tracker_scalars(work_dir: Path, required_tags: tuple[str, ...]) -> dict[str, dict[int, float]]:
    """Read step-indexed scalar series from trainer ``tracker.jsonl`` files.

    Only rows containing every tag in ``required_tags`` are kept; the trainer
    also writes unrelated mini-batch rows that must not pollute the golden
    comparison.

    Args:
        work_dir (Path): WORK_DIR of a finished run, searched recursively.
        required_tags (tuple[str, ...]): Tags that must all be present in a row.

    Returns:
        dict[str, dict[int, float]]: Mapping of tag to step-indexed values.
    """
    scalars: dict[str, dict[int, float]] = {tag: {} for tag in required_tags}
    for metrics_path in sorted(work_dir.rglob("tracker.jsonl")):
        with metrics_path.open(encoding="utf-8") as file:
            for line in file:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not all(tag in payload for tag in required_tags):
                    continue
                step = int(payload["step"])
                for tag in required_tags:
                    scalars[tag][step] = float(payload[tag])
    return scalars


def compare_metrics(
    golden: dict,
    scalars: dict[str, dict[int, float]],
    steps: list[int],
) -> list[str]:
    """Check golden metrics against actually-read scalar series.

    Args:
        golden (dict): Golden baseline (``metrics`` + ``tolerance``).
        scalars (dict[str, dict[int, float]]): Output of :func:`read_tracker_scalars`.
        steps (list[int]): Training steps that must be present and compared.

    Returns:
        list[str]: Human-readable failure descriptions; empty when all
        metrics match within tolerance.
    """
    failures: list[str] = []
    tolerance = golden["tolerance"]
    default_tol = tolerance.get("default", {"rtol": 0.0, "atol": 0.0})
    for tag, expected in golden["metrics"].items():
        series = scalars.get(tag)
        if series is None:
            failures.append(f"missing scalar tag: {tag}")
            continue
        tol = tolerance.get(tag, default_tol)
        for step in steps:
            if step not in series:
                failures.append(f"missing {tag} at step {step}")
                continue
            expected_value = expected[step]
            actual_value = series[step]
            bound = max(tol["atol"], tol["rtol"] * abs(expected_value))
            if abs(actual_value - expected_value) > bound:
                failures.append(
                    f"{tag} step {step}: actual {actual_value:.6f} outside golden {expected_value:.6f} +/- {bound:.6f}"
                )
    return failures


def check_invariants(golden: dict, scalars: dict[str, dict[int, float]]) -> list[str]:
    """Check structural invariants (step count, batch size) of the run.

    Args:
        golden (dict): Golden baseline (``invariants`` section).
        scalars (dict[str, dict[int, float]]): Output of :func:`read_tracker_scalars`.

    Returns:
        list[str]: Human-readable failure descriptions; empty when all
        invariants hold.
    """
    failures: list[str] = []
    invariants = golden.get("invariants", {})
    num_steps = invariants.get("num_train_steps")
    batch_series = scalars.get("response/batch_size", {})
    if num_steps is not None and sorted(batch_series) != list(range(1, num_steps + 1)):
        failures.append(f"expected response/batch_size at steps 1..{num_steps}, got steps {sorted(batch_series)}")
    batch_size = invariants.get("response/batch_size")
    if batch_size is not None:
        for step, value in batch_series.items():
            if int(value) != batch_size:
                failures.append(f"response/batch_size at step {step}: {int(value)} != {batch_size}")
    return failures


class _BaseOpdTwoStepAccuracyTest(unittest.TestCase):
    num_total_gpus = 8
    num_student_gpus = 8
    num_workers = 8
    student_cuda_visible_devices: str | None = None
    rollout_gpu_memory_utilization = 0.5
    experimental_name = "qwen3_vl_2b_opd_two_step_accuracy"

    def setUp(self) -> None:
        self._skip_when_env_unavailable()

        self.temp_dir = tempfile.TemporaryDirectory(
            prefix=f"opd_two_step_accuracy_{time.strftime('%Y%m%d%H%M%S')}_{os.getpid()}_",
        )
        self.addCleanup(self.temp_dir.cleanup)
        self.temp_dir_path = Path(self.temp_dir.name)
        print(f"opd two-step accuracy temp dir: {self.temp_dir_path}")

        env_updates = {
            "XTUNER_USE_LMDEPLOY": "1",
            "XTUNER_USE_SGLANG": "0",
            "XTUNER_USE_VLLM": "0",
            "XTUNER_USE_FA3": "1",
            "XTUNER_DETERMINISTIC": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
        if self.student_cuda_visible_devices is not None:
            env_updates["CUDA_VISIBLE_DEVICES"] = self.student_cuda_visible_devices
        env_updates.update(self.extra_env_updates())
        self._patch_env(env_updates, unset=("RAY_ADDRESS",))

        # ray.shutdown() in a previous test invalidated this module-global named-actor
        # handle; a stale handle would fail with "actor from a different cluster".
        trace_store._handle_cache = None

        self._setup_before_ray()

        ray.init(address="local", num_cpus=128, num_gpus=self.num_student_gpus, ignore_reinit_error=True)

    def tearDown(self) -> None:
        if ray.is_initialized():
            ray.shutdown()
        if hasattr(self, "_old_env"):
            self._restore_env()

    def build_config(self, work_dir: Path) -> RLColocateTrainerConfig:
        """Build the 2-step OPD trainer config mirrored from the recipe.

        Args:
            work_dir (Path): Directory the trainer writes logs/metrics into.

        Returns:
            RLColocateTrainerConfig: The trainer config (not yet built into
            a trainer instance).
        """
        resources = AcceleratorResourcesConfig(
            accelerator="GPU",
            num_workers=self.num_workers,
            num_cpus_per_worker=12,
            cpu_memory_per_worker=16 * 1024**3,
        )
        rollout_config = RolloutConfig(
            env=self.experimental_name,
            device=resources.accelerator,
            model_path=str(STUDENT_MODEL_PATH),
            dtype="bfloat16",
            tensor_parallel_size=1,
            expert_parallel_size=1,
            gpu_memory_utilization=self.rollout_gpu_memory_utilization,
            context_length=PACK_MAX_LENGTH,
            enable_return_routed_experts=False,
            rollout_max_batch_size_per_instance=2048,
        )
        model_cfg = self._student_model_cfg()
        loss_cfg = self._loss_cfg()
        train_worker_cfg = WorkerConfig(
            model_cfg=model_cfg,
            load_from=str(STUDENT_MODEL_PATH),
            optim_cfg=AdamWConfig(lr=1e-6, foreach=False, weight_decay=0.1, betas=(0.9, 0.98)),
            loss_cfg=loss_cfg,
            lr_cfg=LRConfig(lr_type="constant", warmup_ratio=0, lr_min=1e-6),
            fsdp_cfg=FSDPConfig(torch_compile=False, cpu_offload=False, ep_size=1, reduce_dtype="float32"),
            sp_size=1,
            optimizer_steps=1,
            pack_max_length=PACK_MAX_LENGTH,
        )
        return RLColocateTrainerConfig(
            resources=resources,
            train_worker_cfg=train_worker_cfg,
            rollout_config=rollout_config,
            tokenizer_path=str(STUDENT_MODEL_PATH),
            replay_buffer_config=SyncReplayBufferConfig(),
            agent_loop_manager_cfg=self._agent_loop_manager_cfg(),
            eval_agent_loop_manager_cfg=None,
            evaluator_config=None,
            load_from=str(STUDENT_MODEL_PATH),
            train_batch_size=TRAIN_BATCH_SIZE,
            advantage_estimator_config=GRPOAdvantageConfig(eps=1e-8),
            distillation_config=self._distillation_config(loss_cfg),
            enable_evaluate=False,
            enable_initial_evaluate=False,
            evaluate_step=5,
            total_train_steps=TOTAL_TRAIN_STEPS,
            total_epochs=15,
            work_dir=str(work_dir),
            seed=SEED,
            debug_rollout=False,
            exp_tracker="jsonl",
        )

    def extra_env_updates(self) -> dict[str, str]:
        return {}

    def _student_model_cfg(self) -> Qwen3VLDense2BConfig:
        model_cfg = Qwen3VLDense2BConfig()
        if hasattr(model_cfg, "balancing_loss_cfg"):
            model_cfg.balancing_loss_cfg = None
        if hasattr(model_cfg, "z_loss_cfg"):
            model_cfg.z_loss_cfg = None
        return model_cfg

    def _loss_cfg(self) -> DistillationLossConfig:
        raise NotImplementedError

    def _distillation_config(self, loss_cfg: DistillationLossConfig) -> DistillationConfig:
        raise NotImplementedError

    def _sample_params(self) -> SampleParams:
        return SampleParams(
            max_tokens=MAX_RESPONSE_LENGTH,
            top_k=0,
            top_p=1.0,
            temperature=1.0,
            min_tokens=0,
            skip_special_tokens=False,
            return_logprob=True,
            return_token_ids=True,
            return_routed_experts=False,
        )

    def _agent_loop_manager_cfg(self) -> AgentLoopManagerConfig:
        return AgentLoopManagerConfig(
            tasks=TaskSpecConfig(
                task_name="train_task",
                agent_loop_config=SingleTurnAgentLoopConfig(
                    hf_checkpoint=str(STUDENT_MODEL_PATH),
                    sample_params=self._sample_params(),
                ),
                judger_config=self._judger_config(),
                produce_strategy_config=SyncProduceStrategyConfig(),
                sampler_config=SamplerConfig(
                    dataloader_cfg=self._dataloader_cfg(),
                    prompt_repeat_k=1,
                ),
            ),
        )

    def _judger_config(self) -> ComposedJudgerConfig:
        return ComposedJudgerConfig(
            branches={
                "openai/gsm8k": GSM8KJudgerConfig(
                    judger_name="openai/gsm8k",
                    cpu_resources=CPUResourcesConfig(num_workers=1, num_cpus_per_worker=1),
                ),
                "hiyouga/geometry3k": GEO3KJudgerConfig(
                    cpu_resources=CPUResourcesConfig(num_workers=1, num_cpus_per_worker=1),
                ),
            }
        )

    def _dataloader_cfg(self) -> DataloaderConfig:
        with open(DATA_PATH, encoding="utf-8") as file:
            ds_collections = json.load(file)
        dataset_config_list = []
        for name, data in ds_collections.items():
            annotations = data["annotation"] if isinstance(data["annotation"], list) else [data["annotation"]]
            for annotation in annotations:
                dataset_config_list.append(
                    {
                        "dataset": DatasetConfig(
                            name=name,
                            anno_path=annotation,
                            media_root=data.get("media_root", ""),
                            sample_ratio=data.get("sample_ratio", 1.0),
                            class_name="VLMJsonlDataset",
                        ),
                        "tokenize_fn": RLQwen3VLTokenizeFnConfig(
                            processor_path=str(STUDENT_MODEL_PATH),
                            max_length=MAX_PROMPT_LENGTH,
                            system_message=data.get("system_message", None),
                            chat_template="qwen3-vl",
                            add_generation_prompt=True,
                            enable_thinking=True,
                        ),
                    }
                )
        return DataloaderConfig(
            dataset_config_list=dataset_config_list,
            num_workers=8,
            pack_max_length=PACK_MAX_LENGTH,
            collator="fake_collator",
            pack_level="none",
        )

    def _run_and_check(self, golden: dict) -> None:
        work_dir = self.temp_dir_path / "work_dir"
        work_dir.mkdir(parents=True, exist_ok=True)

        trainer = self.build_config(work_dir).build()
        try:
            trainer.fit()
        finally:
            trainer._exp_tracker.close()

        required_tags = tuple(golden["metrics"]) + ("response/batch_size",)
        scalars = read_tracker_scalars(work_dir, required_tags)
        batch_steps = sorted(scalars.get("response/batch_size", {}))
        self.assertEqual(batch_steps, [1, 2], f"expected batch_size series at steps [1, 2], got {batch_steps}")
        metric_steps = sorted({int(step) for expected in golden["metrics"].values() for step in expected})
        failures = compare_metrics(golden, scalars, metric_steps) + check_invariants(golden, scalars)
        self.assertEqual(failures, [], "\n".join(failures))

    def _skip_when_env_unavailable(self) -> None:
        required_paths = {
            "QWEN3_VL_2B_PATH": STUDENT_MODEL_PATH,
            "QWEN3_4B_PATH": GSM8K_TEACHER_MODEL_PATH,
            "QWEN3_VL_DENSE_PATH": GEO3K_TEACHER_MODEL_PATH,
        }
        for env_name, path in required_paths.items():
            if path is None:
                raise unittest.SkipTest(f"{env_name} is not set; source ci/scripts/CI_ENV.sh before running this test")
            if not path.exists():
                raise unittest.SkipTest(f"{env_name} does not exist: {path}")
        if not DATA_PATH.exists():
            raise unittest.SkipTest(f"data file does not exist: {DATA_PATH}")
        visible_gpus = torch.cuda.device_count()
        if visible_gpus < self.num_total_gpus:
            raise unittest.SkipTest(f"requires {self.num_total_gpus} GPUs, found {visible_gpus}")

    def _setup_before_ray(self) -> None:
        return

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


class TestOpdTopKTrainTeacherTwoStep(_BaseOpdTwoStepAccuracyTest):
    """2-step train-teacher forward_kl_topk run vs ``TOPK_GOLDEN``."""

    experimental_name = "qwen3_vl_2b_train_teacher_topk_e2e"
    rollout_gpu_memory_utilization = 0.5

    def test_topk_train_teacher_two_step_matches_golden(self) -> None:
        self._run_and_check(TOPK_GOLDEN)

    def _loss_cfg(self) -> DistillationLossConfig:
        return DistillationLossConfig(
            policy_loss_cfg={
                "cliprange_high": 0.2,
                "cliprange_low": 0.2,
                "loss_type": "vanilla",
                "clip_ratio_c": 10.0,
                "log_prob_diff_min": -20.0,
                "log_prob_diff_max": 20.0,
            },
            ignore_idx=-100,
            use_kl_loss=False,
            kl_loss_coef=0.0,
            kl_loss_type="low_var_kl",
            mode="chunk",
            chunk_size=512,
            loss_mode="forward_kl_topk",
            use_policy_gradient=False,
            top_k=TOP_K,
            log_prob_min_clamp=-10.0,
            loss_max_clamp=10.0,
            task_adv_weight=0.0,
            distillation_loss_weight=1.0,
        )

    def _distillation_config(self, loss_cfg: DistillationLossConfig) -> DistillationConfig:
        teacher_fsdp_cfg = FSDPConfig(
            torch_compile=False,
            cpu_offload=False,
            ep_size=1,
            reduce_dtype="float32",
            recompute_ratio=0,
            vision_recompute_ratio=0,
            requires_grad=False,
        )
        gsm8k_teacher_model_cfg = self._teacher_model_cfg(Qwen3Dense4BConfig())
        geo3k_teacher_model_cfg = self._teacher_model_cfg(Qwen3VLDense4BConfig())
        # Ensure VL rope section matches Qwen3-VL HF checkpoints.
        geo3k_teacher_model_cfg.text_config.rope_parameters_cfg = RopeParametersConfig(
            rope_theta=5000000.0,
            rope_type="qwen3_vl",
            mrope_section=[24, 20, 20],
        )
        return DistillationConfig(
            loss_config=loss_cfg,
            teachers=[
                TrainTeacherConfig(
                    name="gsm8k_teacher",
                    model_path=str(GSM8K_TEACHER_MODEL_PATH),
                    model_cfg=gsm8k_teacher_model_cfg,
                    fsdp_cfg=deepcopy(teacher_fsdp_cfg),
                ),
                TrainTeacherConfig(
                    name="geo3k_teacher",
                    model_path=str(GEO3K_TEACHER_MODEL_PATH),
                    model_cfg=geo3k_teacher_model_cfg,
                    fsdp_cfg=deepcopy(teacher_fsdp_cfg),
                ),
            ],
            data_source_teacher_map={
                "openai/gsm8k": "gsm8k_teacher",
                "hiyouga/geometry3k": "geo3k_teacher",
            },
        )

    @staticmethod
    def _teacher_model_cfg(
        model_cfg: Qwen3Dense4BConfig | Qwen3VLDense4BConfig,
    ) -> Qwen3Dense4BConfig | Qwen3VLDense4BConfig:
        if hasattr(model_cfg, "balancing_loss_cfg"):
            model_cfg.balancing_loss_cfg = None
        if hasattr(model_cfg, "z_loss_cfg"):
            model_cfg.z_loss_cfg = None
        model_cfg.compile_cfg = False
        return model_cfg


class TestOpdMopdRolloutTeacherTwoStep(_BaseOpdTwoStepAccuracyTest):
    """2-step sampled-token k1 rollout-teacher run vs ``MOPD_GOLDEN``.

    Mirrors ``run_sampled_token_mopd.sh``: two LMDeploy teacher HTTP
    servers are started outside Ray (GPUs 7/6) and the student uses the
    remaining GPUs via ``CUDA_VISIBLE_DEVICES``.
    """

    experimental_name = "qwen3_vl_2b_mopd_e2e"
    num_student_gpus = 6
    num_workers = 6
    student_cuda_visible_devices = "0,1,2,3,4,5"
    rollout_gpu_memory_utilization = 0.6

    def test_mopd_rollout_teacher_two_step_matches_golden(self) -> None:
        self._run_and_check(MOPD_GOLDEN)

    def extra_env_updates(self) -> dict[str, str]:
        endpoint_map = {
            "gsm8k_teacher": [f"http://127.0.0.1:{GSM8K_TEACHER_PORT}"],
            "geo3k_teacher": [f"http://127.0.0.1:{GEO3K_TEACHER_PORT}"],
        }
        return {"XTUNER_OPD_TEACHER_ENDPOINTS_JSON": json.dumps(endpoint_map)}

    def _setup_before_ray(self) -> None:
        specs = [
            ("gsm8k_teacher", GSM8K_TEACHER_MODEL_PATH, GSM8K_TEACHER_PORT, "7"),
            ("geo3k_teacher", GEO3K_TEACHER_MODEL_PATH, GEO3K_TEACHER_PORT, "6"),
        ]
        teachers = [self._launch_teacher(name, model_path, port, device) for name, model_path, port, device in specs]
        for name, proc, port, log_path in teachers:
            self._wait_teacher_healthy(name, proc, port, log_path)

    def _launch_teacher(
        self,
        name: str,
        model_path: Path,
        port: int,
        device: str,
    ) -> tuple[str, subprocess.Popen, int, Path]:
        log_path = self.temp_dir_path / f"teacher_{name}.log"
        log_file = log_path.open("w", encoding="utf-8")
        self.addCleanup(log_file.close)
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = device
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            self._teacher_server_command(model_path, port),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.addCleanup(self._stop_teacher, proc)
        return name, proc, port, log_path

    def _teacher_server_command(self, model_path: Path, port: int) -> list[str]:
        # Mirrors build_teacher_server_commands._build_lmdeploy_command for
        # the RolloutTeacherLaunchConfig used by this test (tp=ep=dp=1).
        return [
            sys.executable,
            "-m",
            "lmdeploy",
            "serve",
            "api_server",
            str(model_path),
            "--backend",
            "pytorch",
            "--role",
            "Hybrid",
            "--logprobs-mode",
            "raw_logprobs",
            "--server-name",
            "0.0.0.0",
            "--server-port",
            str(port),
            "--dtype",
            "bfloat16",
            "--tp",
            "1",
            "--ep",
            "1",
            "--dp",
            "1",
            "--cache-max-entry-count",
            "0.8",
            "--session-len",
            str(PACK_MAX_LENGTH),
            "--max-batch-size",
            str(PACK_MAX_LENGTH),
            "--max-prefill-token-num",
            "4096",
        ]

    def _wait_teacher_healthy(
        self,
        name: str,
        proc: subprocess.Popen,
        port: int,
        log_path: Path,
    ) -> None:
        health_url = f"http://127.0.0.1:{port}/health"
        deadline = time.monotonic() + TEACHER_STARTUP_TIMEOUT_S
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                self.fail(
                    f"teacher {name} exited early (code {proc.returncode}); log tail:\n{self._log_tail(log_path)}"
                )
            try:
                with urllib.request.urlopen(health_url, timeout=10) as resp:
                    if resp.status == 200:
                        print(f"teacher {name} healthy at {health_url}")
                        return
            except OSError:
                time.sleep(5)
        self.fail(
            f"teacher {name} not healthy within {TEACHER_STARTUP_TIMEOUT_S}s at {health_url}; "
            f"log tail:\n{self._log_tail(log_path)}"
        )

    def _stop_teacher(self, proc: subprocess.Popen) -> None:
        if proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError, subprocess.TimeoutExpired):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=60)

    @staticmethod
    def _log_tail(log_path: Path, lines: int = 40) -> str:
        try:
            content = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return "<unavailable>"
        return "\n".join(content[-lines:])

    def _sample_params(self) -> SampleParams:
        # PG-OPD (sampled-token k1) requires identity sampling; enforced by
        # DistillationConfig.validate_trainer via validate_opd_sample_params.
        return SampleParams(
            max_tokens=MAX_RESPONSE_LENGTH,
            top_k=0,
            top_p=1.0,
            temperature=1.0,
            repetition_penalty=1.0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            min_tokens=0,
            skip_special_tokens=False,
            return_logprob=True,
            return_token_ids=True,
            return_routed_experts=False,
        )

    def _loss_cfg(self) -> DistillationLossConfig:
        return DistillationLossConfig(
            policy_loss_cfg={
                "cliprange_high": 0.2,
                "cliprange_low": 0.2,
                "loss_type": "vanilla",
                "clip_ratio_c": 10.0,
                "log_prob_diff_min": -20.0,
                "log_prob_diff_max": 20.0,
            },
            ignore_idx=-100,
            use_kl_loss=False,
            kl_loss_coef=0.0,
            kl_loss_type="low_var_kl",
            mode="chunk",
            chunk_size=512,
            loss_mode="k1",
            use_policy_gradient=True,
            task_adv_weight=0.0,
            distillation_loss_weight=1.0,
        )

    def _distillation_config(self, loss_cfg: DistillationLossConfig) -> DistillationConfig:
        return DistillationConfig(
            loss_config=loss_cfg,
            teachers=[
                RolloutTeacherConfig(
                    name="gsm8k_teacher",
                    num_replicas=1,
                    enable_prefix_caching=False,
                    launch_config=RolloutTeacherLaunchConfig(
                        model_path=str(GSM8K_TEACHER_MODEL_PATH),
                        num_workers=1,
                        server_port=GSM8K_TEACHER_PORT,
                        dtype="bfloat16",
                        tensor_parallel_size=1,
                        expert_parallel_size=1,
                        context_length=PACK_MAX_LENGTH,
                        max_batch_size=PACK_MAX_LENGTH,
                        gpu_memory_utilization=0.8,
                    ),
                ),
                RolloutTeacherConfig(
                    name="geo3k_teacher",
                    num_replicas=1,
                    enable_prefix_caching=False,
                    launch_config=RolloutTeacherLaunchConfig(
                        model_path=str(GEO3K_TEACHER_MODEL_PATH),
                        num_workers=1,
                        server_port=GEO3K_TEACHER_PORT,
                        dtype="bfloat16",
                        tensor_parallel_size=1,
                        expert_parallel_size=1,
                        context_length=PACK_MAX_LENGTH,
                        max_batch_size=PACK_MAX_LENGTH,
                        gpu_memory_utilization=0.8,
                    ),
                ),
            ],
            data_source_teacher_map={
                "openai/gsm8k": "gsm8k_teacher",
                "hiyouga/geometry3k": "geo3k_teacher",
            },
        )


if __name__ == "__main__":
    unittest.main()
