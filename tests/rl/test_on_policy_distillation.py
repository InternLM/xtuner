import math
import os
import signal
import subprocess
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import torch

from recipe.on_policy_distillation.build_teacher_server_commands import (
    build_teacher_server_command,
)
from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.rl.loss import GRPOLossConfig
from xtuner.v1.rl.on_policy_distillation import (
    OPDConfig,
    OPDTeacherConfig,
    OPDTeacherLaunchConfig,
    TeacherLogprobClient,
    apply_opd_kl_to_advantages,
)
from xtuner.v1.rl.utils import find_free_ports


REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = os.getenv("XTUNER_OPD_BASELINE")
TEACHER_MODEL_PATH = os.getenv("XTUNER_OPD_TEACHER_MODEL")
STUDENT_MODEL_PATH = os.getenv("XTUNER_OPD_STUDENT_MODEL")
TEACHER_STARTUP_TIMEOUT_S = float(os.getenv("XTUNER_OPD_TEACHER_STARTUP_TIMEOUT_S", "1200"))
TRAINER_CONFIG_PATH = (
    REPO_ROOT / "recipe/on_policy_distillation/config/rl_dapo_math_opd.py"
)


def _wait_for_teacher(process: subprocess.Popen, endpoint: str, backend: str) -> None:
    deadline = time.monotonic() + TEACHER_STARTUP_TIMEOUT_S
    health_path = "health_generate" if backend == "sglang" else "health"
    with httpx.Client(timeout=1.0, trust_env=False) as client:
        while time.monotonic() < deadline:
            return_code = process.poll()
            if return_code is not None:
                raise RuntimeError(f"Teacher process exited during startup with code {return_code}")
            try:
                if client.get(f"{endpoint}/{health_path}").status_code == 200:
                    return
            except httpx.RequestError:
                pass
            time.sleep(1.0)
    raise TimeoutError(f"Teacher did not become ready within {TEACHER_STARTUP_TIMEOUT_S} seconds")


def _build_debug_rollout_batch(samples: list[dict]) -> list[list[RolloutState]]:
    train_batch = []
    for sample in samples:
        sample_index = int(sample["sample_index"])
        group_index = sample.get("group_index")
        prompt_ids = torch.as_tensor(sample["prompt_token_ids"], dtype=torch.long).tolist()
        response_ids = torch.as_tensor(sample["response_token_ids"], dtype=torch.long).tolist()
        rollout_logprobs = torch.as_tensor(sample["rollout_log_probs"], dtype=torch.float32).tolist()
        teacher_logprobs = torch.as_tensor(sample["teacher_log_probs"], dtype=torch.float32).tolist()
        response_mask = torch.as_tensor(sample["loss_mask"], dtype=torch.bool).int().tolist()
        response = str(sample["response"])

        train_batch.append(
            [
                RolloutState(
                    rollout_id=sample_index,
                    group_id=sample_index if group_index is None else int(group_index),
                    message=[],
                    prompt_ids=prompt_ids,
                    tokens=prompt_ids,
                    response=response,
                    response_ids=response_ids,
                    logprobs=rollout_logprobs,
                    teacher_tokens=response_ids,
                    teacher_logprobs=teacher_logprobs,
                    response_mask=response_mask,
                    reward={"score": 0.0},
                    finish_reason="stop",
                    status=Status.COMPLETED,
                    extra_fields={"origin_data_source": "baseline"},
                )
            ]
        )
    return train_batch


@unittest.skipUnless(BASELINE_PATH, "XTUNER_OPD_BASELINE is required")
class TestPGOPDAdvantage(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        baseline = torch.load(BASELINE_PATH, map_location="cpu", weights_only=False)
        cls.samples = baseline["samples"]

    def test_compute_advantages_matches_baseline(self) -> None:
        config = OPDConfig(
            teachers=[OPDTeacherConfig(name="teacher", endpoint="http://unused")],
            data_source_teacher_map={"baseline": "teacher"},
        )
        loss_cfg = GRPOLossConfig(policy_loss_cfg={"loss_type": "vanilla"})

        for sample in self.samples:
            with self.subTest(sample_index=sample["sample_index"]):
                old_logprobs = torch.as_tensor(sample["old_log_probs"], dtype=torch.float32)
                teacher_logprobs = torch.as_tensor(sample["teacher_log_probs"], dtype=torch.float32)
                loss_mask = torch.as_tensor(sample["loss_mask"], dtype=torch.float32)
                shifted_labels = torch.where(
                    loss_mask.bool(),
                    torch.zeros_like(loss_mask, dtype=torch.long),
                    torch.full_like(loss_mask, -100, dtype=torch.long),
                )
                loss_ctx = loss_cfg.build(
                    {
                        "shifted_labels": shifted_labels,
                        "advantages": torch.zeros_like(old_logprobs),
                        "old_logprobs": old_logprobs,
                        "teacher_logprobs": teacher_logprobs,
                    }
                )
                assert loss_ctx is not None
                apply_opd_kl_to_advantages(loss_ctx, config=config)
                actual_advantages = loss_ctx.loss_kwargs.advantages.cpu()
                expected_advantages = torch.as_tensor(sample["advantages"], dtype=torch.float32) * loss_mask
                try:
                    torch.testing.assert_close(
                        actual_advantages,
                        expected_advantages,
                        rtol=1e-5,
                        atol=1e-5,
                    )
                except AssertionError as error:
                    mismatch_indices = torch.nonzero(
                        ~torch.isclose(actual_advantages, expected_advantages, rtol=1e-5, atol=1e-5)
                    ).flatten()
                    mismatch_values = "\n".join(
                        (
                            f"index={index}: "
                            f"actual={actual_advantages[index].item()!r}, "
                            f"expected={expected_advantages[index].item()!r}, "
                            f"abs_diff={abs(actual_advantages[index] - expected_advantages[index]).item()!r}"
                        )
                        for index in mismatch_indices.tolist()
                    )
                    raise AssertionError(f"{error}\n\nMismatched advantages:\n{mismatch_values}") from None


def _load_trainer_samples(capture_dir: Path) -> dict[int, dict]:
    trainer_samples = {}
    for capture_file in sorted(capture_dir.glob("rank_*.pt")):
        for batch in torch.load(capture_file, map_location="cpu", weights_only=True):
            shifted_labels = torch.as_tensor(batch["shifted_labels"], dtype=torch.long).reshape(-1)
            old_log_probs = torch.as_tensor(batch["old_log_probs"], dtype=torch.float32).reshape(-1)
            advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32).reshape(-1)
            boundaries = torch.as_tensor(batch["cu_seq_lens_q"], dtype=torch.long).tolist()
            num_padding = int(batch["num_padding"])
            padding_start = shifted_labels.numel() - num_padding

            assert boundaries[len(batch["rollout_ids"])] == padding_start
            torch.testing.assert_close(
                shifted_labels[padding_start:],
                torch.full_like(shifted_labels[padding_start:], -100),
                rtol=0,
                atol=0,
            )

            for rollout_id, start, end in zip(batch["rollout_ids"], boundaries[:-1], boundaries[1:]):
                response_mask = shifted_labels[start:end] != -100
                trainer_samples[int(rollout_id)] = {
                    "response_token_ids": shifted_labels[start:end][response_mask],
                    "old_log_probs": old_log_probs[start:end][response_mask],
                    "advantages": advantages[start:end][response_mask],
                }

    return trainer_samples


def _run_trainer_once(
    baseline_samples: list[dict],
    *,
    baseline_path: Path,
    student_model_path: Path,
    debug_rollout_dir: Path,
    capture_dir: Path,
    run_dir: Path,
) -> dict[int, dict]:
    import ray

    from xtuner.v1.rl.trainer.controller import TrainingController
    from xtuner.v1.rl.trainer.worker import TrainingWorker, WorkerConfig
    from xtuner.v1.train.rl_trainer import RLColocateTrainer
    from xtuner.v1.utils import Config

    max_prompt_length = max(len(sample["prompt_token_ids"]) for sample in baseline_samples)
    max_response_length = max(len(sample["response_token_ids"]) for sample in baseline_samples)
    max_input_length = max(len(sample["token_ids"]) - 1 for sample in baseline_samples)
    pack_max_length = math.ceil(max_input_length / 512) * 512
    num_workers = 8

    class CapturingRLColocateTrainer(RLColocateTrainer):
        def _prepare_train_data(
            self,
            data_groups,
            pack_max_length,
            raw_rewards_sum=0.0,
            raw_rewards_count=0,
        ):
            data_batches, data_info = super()._prepare_train_data(
                data_groups,
                pack_max_length,
                raw_rewards_sum=raw_rewards_sum,
                raw_rewards_count=raw_rewards_count,
            )
            rollout_states = (state for group in data_groups for state in group)
            for data_batch, rollout_state in zip(data_batches, rollout_states):
                data_batch["rollout_id"] = rollout_state.rollout_id
            return data_batches, data_info

    class CapturingTrainingController(TrainingController):
        def _packing(self, data_batches, pack_max_length, language_cfg):
            pack_infos = self._get_pack_infos(
                data_batches,
                [data["seq_ctx"].input_ids.numel() for data in data_batches],
                pack_max_length,
            )
            packed_data_batches = super()._packing(data_batches, pack_max_length, language_cfg)
            for packed_data, pack_info in zip(packed_data_batches, pack_infos):
                packed_data["rollout_ids"] = [data_batches[index]["rollout_id"] for index in pack_info["indices"]]
            return packed_data_batches

    class CapturingTrainingWorker(TrainingWorker):
        trainer_capture_dir = capture_dir

        def fit(self, data_batches, rollout_idx):
            from unittest.mock import patch as mock_patch

            from xtuner.v1.rl.trainer import worker as worker_module

            captured_batches = [
                {
                    "rollout_ids": data.get("rollout_ids", []),
                    "cu_seq_lens_q": data["seq_ctx"].cu_seq_lens_q.detach().cpu(),
                    "num_padding": data["seq_ctx"].num_padding,
                    "shifted_labels": data["shifted_labels"].detach().cpu().reshape(-1),
                }
                for data in data_batches
            ]
            captured_batch_index = 0
            apply_opd = worker_module.apply_opd_kl_to_advantages

            def capture_opd_result(loss_ctx, *, config):
                nonlocal captured_batch_index
                reverse_kl_sum = apply_opd(loss_ctx, config=config)
                captured_batches[captured_batch_index]["old_log_probs"] = (
                    loss_ctx.loss_kwargs.old_logprobs.detach().cpu().reshape(-1)
                )
                captured_batches[captured_batch_index]["advantages"] = (
                    loss_ctx.loss_kwargs.advantages.detach().cpu().reshape(-1)
                )
                captured_batch_index += 1
                return reverse_kl_sum

            with mock_patch.object(worker_module, "apply_opd_kl_to_advantages", capture_opd_result):
                worker_log_item = TrainingWorker.fit(self, data_batches, rollout_idx)

            torch.save(captured_batches, self.trainer_capture_dir / f"rank_{self.rank}.pt")
            return worker_log_item

    def build_capturing_training_workers(self, placement_group):
        from xtuner.v1.rl.utils import AutoAcceleratorWorkers

        capturing_worker_cls = ray.remote(
            runtime_env={
                "env_vars": {
                    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
                    "HCCL_NPU_SOCKET_PORT_RANGE": "auto",
                }
            }
        )(CapturingTrainingWorker)
        train_workers, _ = AutoAcceleratorWorkers.from_placement_group(
            capturing_worker_cls,
            self,
            placement_group,
        )
        ray.wait([worker.ready.remote() for worker in train_workers])
        return CapturingTrainingController(workers=train_workers)

    trainer_environment = {
        "WORK_DIR": str(run_dir / "trainer"),
        "MODEL_PATH": str(student_model_path),
        "TEACHER_MODEL_PATH": os.getenv("XTUNER_OPD_TEACHER_MODEL", str(student_model_path)),
        "DATA_PATH": os.getenv("XTUNER_OPD_DATA_PATH", str(baseline_path)),
        "WORLD_SIZE": "1",
        "ONLY_CALC_MISMATCH_RATIO": "1",
        "XTUNER_DETERMINISTIC": "true",
        "XTUNER_USE_FA3": os.getenv("XTUNER_OPD_USE_FA3", "1"),
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
    }

    with patch.dict(os.environ, trainer_environment, clear=False):
        try:
            ray.init(
                num_cpus=12 * num_workers + 10,
                num_gpus=num_workers,
                include_dashboard=False,
                _temp_dir="/dev/shm/xtuner-opd-old-logprobs",
            )
            cfg = Config.fromfile(TRAINER_CONFIG_PATH)
            cfg.trainer.resources.num_workers = num_workers
            cfg.trainer.total_train_steps = 1
            cfg.trainer.train_batch_size = len(baseline_samples)
            cfg.trainer.train_worker_cfg.optimizer_steps = 1
            cfg.trainer.train_worker_cfg.pack_max_length = pack_max_length
            cfg.trainer.debug_train = True
            cfg.trainer.debug_rollout_dir = debug_rollout_dir
            with (
                patch("xtuner.v1.train.rl_trainer.XTUNER_DETERMINISTIC", True),
                patch(
                    "xtuner.v1.train.rl_trainer.RLColocateTrainer",
                    CapturingRLColocateTrainer,
                ),
                patch.object(WorkerConfig, "build", build_capturing_training_workers),
            ):
                trainer = cfg.trainer.build()
                trainer.fit()
        finally:
            if ray.is_initialized():
                ray.shutdown()

    return _load_trainer_samples(capture_dir)


@unittest.skipUnless(
    BASELINE_PATH and STUDENT_MODEL_PATH,
    "XTUNER_OPD_BASELINE and XTUNER_OPD_STUDENT_MODEL are required",
)
class TestPGOPDOldLogprobs(unittest.TestCase):
    def test_trainer_old_logprobs_and_advantage_error_propagation(self) -> None:
        baseline_path = Path(str(BASELINE_PATH)).expanduser().resolve()
        student_model_path = Path(str(STUDENT_MODEL_PATH)).expanduser().resolve()
        baseline = torch.load(baseline_path, map_location="cpu", weights_only=True)
        baseline_samples = baseline["samples"]

        work_root = Path(
            os.getenv(
                "XTUNER_OPD_OLD_LOGPROB_WORK_DIR",
                str(REPO_ROOT / "work_dirs/test_pg_opd_old_logprobs"),
            )
        ).expanduser()
        run_dir = work_root / f"run_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        debug_rollout_dir = run_dir / "debug_rollout"
        capture_dir = run_dir / "trainer_capture"
        debug_rollout_dir.mkdir(parents=True)
        capture_dir.mkdir()

        train_batch = _build_debug_rollout_batch(baseline_samples)
        torch.save(train_batch, debug_rollout_dir / "debug_rollout_1.pt")
        trainer_samples_by_rollout_id = _run_trainer_once(
            baseline_samples,
            baseline_path=baseline_path,
            student_model_path=student_model_path,
            debug_rollout_dir=debug_rollout_dir,
            capture_dir=capture_dir,
            run_dir=run_dir,
        )

        result_path = run_dir / "result.pt"
        result_samples = []

        for baseline_sample in baseline_samples:
            rollout_id = int(baseline_sample["sample_index"])
            trainer_sample = trainer_samples_by_rollout_id[rollout_id]
            loss_mask = torch.as_tensor(baseline_sample["loss_mask"], dtype=torch.bool)
            response_token_ids = torch.as_tensor(baseline_sample["response_token_ids"], dtype=torch.long)[loss_mask]
            torch.testing.assert_close(
                trainer_sample["response_token_ids"],
                response_token_ids,
                rtol=0,
                atol=0,
            )
            baseline_old_log_probs = torch.as_tensor(
                baseline_sample["old_log_probs"],
                dtype=torch.float32,
            )[loss_mask]
            trainer_old_log_probs = trainer_sample["old_log_probs"]
            baseline_advantages = torch.as_tensor(
                baseline_sample["advantages"],
                dtype=torch.float32,
            )[loss_mask]
            trainer_advantages = trainer_sample["advantages"]

            old_logprob_error = trainer_old_log_probs - baseline_old_log_probs
            advantage_error = trainer_advantages - baseline_advantages
            propagation_residual = advantage_error + old_logprob_error
            sample_num_tokens = old_logprob_error.numel()
            sample_summary = {
                "num_tokens": sample_num_tokens,
                "old_logprobs_mean_abs_error": old_logprob_error.abs().mean().item(),
                "old_logprobs_max_abs_error": old_logprob_error.abs().max().item(),
                "advantages_mean_abs_error": advantage_error.abs().mean().item(),
                "advantages_max_abs_error": advantage_error.abs().max().item(),
                "propagation_mean_abs_error": propagation_residual.abs().mean().item(),
                "propagation_max_abs_error": propagation_residual.abs().max().item(),
            }
            result_samples.append(
                {
                    "sample_index": rollout_id,
                    "response_token_ids": response_token_ids,
                    "baseline_old_log_probs": baseline_old_log_probs,
                    "trainer_old_log_probs": trainer_old_log_probs,
                    "old_logprob_error": old_logprob_error,
                    "baseline_advantages": baseline_advantages,
                    "trainer_advantages": trainer_advantages,
                    "advantage_error": advantage_error,
                    "propagation_residual": propagation_residual,
                    "summary": sample_summary,
                }
            )

            with self.subTest(sample_index=rollout_id):
                torch.testing.assert_close(
                    advantage_error,
                    -old_logprob_error,
                    rtol=1e-5,
                    atol=1e-5,
                    msg=f"Result: {result_path}",
                )

        global_result = {
            "response_token_ids": torch.cat([sample["response_token_ids"] for sample in result_samples]),
            "baseline_old_log_probs": torch.cat([sample["baseline_old_log_probs"] for sample in result_samples]),
            "trainer_old_log_probs": torch.cat([sample["trainer_old_log_probs"] for sample in result_samples]),
            "old_logprob_error": torch.cat([sample["old_logprob_error"] for sample in result_samples]),
            "baseline_advantages": torch.cat([sample["baseline_advantages"] for sample in result_samples]),
            "trainer_advantages": torch.cat([sample["trainer_advantages"] for sample in result_samples]),
            "advantage_error": torch.cat([sample["advantage_error"] for sample in result_samples]),
            "propagation_residual": torch.cat([sample["propagation_residual"] for sample in result_samples]),
        }
        summary = {
            "num_samples": len(result_samples),
            "num_tokens": global_result["old_logprob_error"].numel(),
            "old_logprobs_mean_abs_error": global_result["old_logprob_error"].abs().mean().item(),
            "old_logprobs_max_abs_error": global_result["old_logprob_error"].abs().max().item(),
            "advantages_mean_abs_error": global_result["advantage_error"].abs().mean().item(),
            "advantages_max_abs_error": global_result["advantage_error"].abs().max().item(),
            "propagation_mean_abs_error": global_result["propagation_residual"].abs().mean().item(),
            "propagation_max_abs_error": global_result["propagation_residual"].abs().max().item(),
        }
        torch.save(
            {
                "summary": summary,
                "global": global_result,
                "samples": result_samples,
            },
            result_path,
        )

        with self.subTest(scope="global"):
            torch.testing.assert_close(
                global_result["advantage_error"],
                -global_result["old_logprob_error"],
                rtol=1e-5,
                atol=1e-5,
                msg=f"Result: {result_path}",
            )

        print(
            f"old_logprobs: mean_abs={summary['old_logprobs_mean_abs_error']}, "
            f"max_abs={summary['old_logprobs_max_abs_error']}\n"
            f"advantages: mean_abs={summary['advantages_mean_abs_error']}, "
            f"max_abs={summary['advantages_max_abs_error']}\n"
            f"propagation: mean_abs={summary['propagation_mean_abs_error']}, "
            f"max_abs={summary['propagation_max_abs_error']}\n"
            f"result: {result_path}"
        )


@unittest.skipUnless(
    BASELINE_PATH and TEACHER_MODEL_PATH,
    "XTUNER_OPD_BASELINE and XTUNER_OPD_TEACHER_MODEL are required",
)
class TestTeacherLogprobClient(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        baseline = torch.load(BASELINE_PATH, map_location="cpu", weights_only=False)
        cls.samples = baseline["samples"]
        port = find_free_ports()[0]
        cls.teacher_endpoint = f"http://127.0.0.1:{port}"
        cls.teacher_backend = TeacherLogprobClient._resolve_backend_from_env()
        teacher_env = os.environ.copy()
        teacher_command = build_teacher_server_command(
            OPDTeacherConfig(
                name="teacher",
                endpoint=cls.teacher_endpoint,
                launch_config=OPDTeacherLaunchConfig(
                    model_path=str(TEACHER_MODEL_PATH),
                    cuda_visible_devices=teacher_env.get("CUDA_VISIBLE_DEVICES")
                    or "7",
                ),
            ),
            cls.teacher_backend,
        )
        if not teacher_command:
            raise ValueError("Teacher must define launch_config for local startup")
        cls.teacher_process = subprocess.Popen(
            teacher_command,
            cwd=REPO_ROOT,
            env=teacher_env,
            start_new_session=True,
        )
        cls.addClassCleanup(cls._stop_teacher)
        _wait_for_teacher(cls.teacher_process, cls.teacher_endpoint, cls.teacher_backend)

    @classmethod
    def _stop_teacher(cls) -> None:
        if cls.teacher_process.poll() is not None:
            return
        os.killpg(cls.teacher_process.pid, signal.SIGTERM)
        try:
            cls.teacher_process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(cls.teacher_process.pid, signal.SIGKILL)
            cls.teacher_process.wait()

    @unittest.skipUnless(os.getenv("XTUNER_USE_SGLANG", "0") == "1", "XTUNER_USE_SGLANG=1 is required")
    async def test_compute_logprobs_with_sglang_matches_baseline(self) -> None:
        self.assertEqual(self.teacher_backend, "sglang")
        await self._assert_compute_logprobs_matches_baseline()

    @unittest.skipUnless(os.getenv("XTUNER_USE_LMDEPLOY", "0") == "1", "XTUNER_USE_LMDEPLOY=1 is required")
    async def test_compute_logprobs_with_lmdeploy_matches_baseline(self) -> None:
        self.assertEqual(self.teacher_backend, "lmdeploy")
        await self._assert_compute_logprobs_matches_baseline()

    async def _assert_compute_logprobs_matches_baseline(self) -> None:
        client = TeacherLogprobClient(OPDTeacherConfig(name="teacher", endpoint=self.teacher_endpoint))
        self.addAsyncCleanup(client._client.aclose)

        for sample in self.samples:
            with self.subTest(sample_index=sample["sample_index"]):
                prompt_ids = torch.as_tensor(sample["prompt_token_ids"], dtype=torch.long).tolist()
                response_ids = torch.as_tensor(sample["response_token_ids"], dtype=torch.long).tolist()
                expected_logprobs = torch.as_tensor(sample["teacher_log_probs"], dtype=torch.float32)
                state = RolloutState(
                    rollout_id=int(sample["sample_index"]),
                    group_id=int(sample["group_index"]),
                    message=[],
                    prompt_ids=prompt_ids,
                    tokens=prompt_ids,
                    response="",
                    response_ids=response_ids,
                    status=Status.COMPLETED,
                )

                result = await client.compute_logprobs(state)

                self.assertEqual(result.status, Status.COMPLETED, result.error_msg)
                self.assertEqual(result.teacher_tokens, response_ids)
                actual_logprobs = torch.tensor(result.teacher_logprobs, dtype=torch.float32)
                self.assertEqual(actual_logprobs.shape, expected_logprobs.shape)
                absolute_errors = torch.abs(actual_logprobs - expected_logprobs)
                mae = absolute_errors.mean().item()
                self.assertLessEqual(
                    mae,
                    0.05,
                    (
                        f"Teacher logprob MAE {mae:.8f} exceeds threshold 0.05; "
                        f"max_abs_error={absolute_errors.max().item():.8f}"
                    ),
                )


if __name__ == "__main__":
    unittest.main()
