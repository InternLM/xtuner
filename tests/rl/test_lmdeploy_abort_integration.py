import asyncio
import math
import os
import tempfile
import time
import unittest
from pathlib import Path

import httpx
import ray
import torch

from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams, Status
from xtuner.v1.rl.rollout.worker import RolloutConfig
from xtuner.v1.rl.utils import (
    AcceleratorResourcesConfig,
    AutoAcceleratorWorkers,
    CPUResourceManager,
    PRODUCER_PAUSE_PENDING_TASK_TIMEOUT_S,
    clear_cpu_resource_manager,
    set_cpu_resource_manager,
)


MODEL_PATH = os.environ.get("ROLLOUT_MODEL_PATH") or os.environ.get("MODEL_PATH", "")
_RESOURCE_MAP = {"npu": "NPU", "cuda": "GPU"}


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _accelerator_type() -> str:
    return _RESOURCE_MAP[torch.accelerator.current_accelerator().type]


def _parse_prometheus_gauge(text: str, metric_name: str) -> float:
    total = 0.0
    for line in text.splitlines():
        if not line.startswith(metric_name):
            continue
        try:
            total += float(line.rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            continue
    return total


async def _fetch_lmdeploy_metrics(server_urls: list[str]) -> dict[str, float]:
    metric_names = {
        "api_routed": "lmdeploy:num_api_requests_routed",
        "api_waiting": "lmdeploy:num_api_requests_waiting",
        "engine_running": "lmdeploy:num_requests_running",
        "engine_waiting": "lmdeploy:num_requests_waiting",
        "failed": "lmdeploy:num_requests_failed",
    }
    totals = dict.fromkeys(metric_names, 0.0)
    async with httpx.AsyncClient(timeout=5.0) as client:
        for url in server_urls:
            response = await client.get(f"{url}/metrics")
            response.raise_for_status()
            text = response.text
            for key, metric_name in metric_names.items():
                totals[key] += _parse_prometheus_gauge(text, metric_name)
    return totals


async def _wait_for_lmdeploy_activity(
    server_urls: list[str],
    *,
    min_active: int,
    timeout_s: float,
) -> dict[str, float]:
    deadline = time.perf_counter() + timeout_s
    last_metrics: dict[str, float] = {}
    while time.perf_counter() < deadline:
        last_metrics = await _fetch_lmdeploy_metrics(server_urls)
        active = (
            last_metrics["api_routed"]
            + last_metrics["api_waiting"]
            + last_metrics["engine_running"]
            + last_metrics["engine_waiting"]
        )
        if active >= min_active:
            return last_metrics
        await asyncio.sleep(0.2)
    return last_metrics


async def _wait_for_lmdeploy_idle(server_urls: list[str], *, timeout_s: float) -> dict[str, float]:
    deadline = time.perf_counter() + timeout_s
    last_metrics: dict[str, float] = {}
    while time.perf_counter() < deadline:
        last_metrics = await _fetch_lmdeploy_metrics(server_urls)
        if (
            last_metrics["api_routed"] == 0
            and last_metrics["api_waiting"] == 0
            and last_metrics["engine_running"] == 0
            and last_metrics["engine_waiting"] == 0
        ):
            return last_metrics
        await asyncio.sleep(0.2)
    return last_metrics


@unittest.skipUnless(
    os.environ.get("XTUNER_USE_LMDEPLOY", "0") == "1" and MODEL_PATH,
    "real LMDeploy abort integration requires XTUNER_USE_LMDEPLOY=1 and ROLLOUT_MODEL_PATH or MODEL_PATH",
)
class TestLMDeployAbortIntegration(unittest.IsolatedAsyncioTestCase):
    """Regression test for real LMDeploy abort behavior under over-concurrency.

    This intentionally starts a real LMDeploy server through XTuner's
    RolloutController, sends more requests than the LMDeploy max batch size, and
    asserts that /abort_request drains both API-side waiting requests and engine
    requests within the configured pause cleanup deadline.
    """

    async def asyncSetUp(self):
        os.environ.setdefault("XTUNER_USE_FA3", "1")
        os.environ.setdefault("LMD_SKIP_WARMUP", "1")

        self.num_workers = 1
        self.tensor_parallel_size = self.num_workers
        self.max_batch_size = 8
        self.request_count = self.max_batch_size * 4
        self.max_tokens = 1024
        self.min_tokens = min(256, self.max_tokens)
        self.abort_after_s = 5.0
        self.startup_activity_timeout_s = _env_float("XTUNER_LMDEPLOY_ABORT_TEST_ACTIVITY_TIMEOUT_S", 60.0)

        ray.init(address="local", num_cpus=18, ignore_reinit_error=True)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.pg = None
        self.rollout_ctl = None

    async def asyncTearDown(self):
        if self.rollout_ctl is not None:
            try:
                await asyncio.wait_for(self.rollout_ctl.shutdown.remote(), timeout=120)
            except Exception:
                pass
        if self.pg is not None:
            try:
                ray.util.remove_placement_group(self.pg)
            except Exception:
                pass
        clear_cpu_resource_manager()
        ray.shutdown()
        self.temp_dir.cleanup()

    def _build_rollout_controller(self):
        resources_cfg = AcceleratorResourcesConfig(
            accelerator=_accelerator_type(),
            num_workers=self.num_workers,
            num_cpus_per_worker=8,
            cpu_memory_per_worker=16 * 1024**3,
        )
        self.pg = AutoAcceleratorWorkers.build_placement_group(resources_cfg, name="lmdeploy_abort_integration_pg")
        set_cpu_resource_manager(CPUResourceManager(accelerator_placement_groups=self.pg))

        over_concurrency = max(1.0, math.ceil(self.request_count / self.max_batch_size))
        rollout_config = RolloutConfig(
            env="lmdeploy_abort_integration",
            device=resources_cfg.accelerator,
            model_path=MODEL_PATH,
            model_name=Path(MODEL_PATH).name.lower(),
            tokenizer_path=MODEL_PATH,
            tensor_parallel_size=self.tensor_parallel_size,
            expert_parallel_size=1,
            context_length=self.max_tokens + 512,
            rollout_max_batch_size_per_instance=self.max_batch_size,
            allow_over_concurrency_ratio=over_concurrency,
            rollout_timeout=max(300.0, self.max_tokens),
            gpu_memory_utilization=_env_float("XTUNER_LMDEPLOY_ABORT_TEST_GPU_MEM", 0.8),
            worker_log_dir=Path(self.temp_dir.name) / "work_dirs",
            health_check_interval_seconds=60.0,
            health_check_failure_threshold=3,
            max_retry_per_sample=0,
            extra_rollout_config={
                "lmdeploy_backend": "pytorch",
                "lmdeploy_log_level": os.environ.get("XTUNER_LMDEPLOY_ABORT_TEST_LOG_LEVEL", "INFO"),
                "lmdeploy_uvicorn_log_level": "WARNING",
            },
        )
        self.rollout_ctl = rollout_config.build(self.pg)
        return self.rollout_ctl

    def _make_state(self, idx: int) -> RolloutState:
        return RolloutState(
            uid=idx,
            message_uid=idx,
            message=[
                {
                    "role": "user",
                    "content": (
                        "Write a long numbered list. Keep going until you are stopped. "
                        f"This is abort integration request {idx}."
                    ),
                }
            ],
            sample_params=SampleParams(
                min_tokens=self.min_tokens,
                max_tokens=self.max_tokens,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                return_token_ids=True,
            ),
        )

    async def test_large_overconcurrent_abort_drains_real_lmdeploy_quickly(self):
        rollout_ctl = self._build_rollout_controller()
        metadata = await rollout_ctl.get_rollout_metadata.remote()
        server_urls = list(metadata["server_url_dict"].values())
        self.assertGreater(len(server_urls), 0)

        refs = [rollout_ctl.generate.remote(self._make_state(i)) for i in range(self.request_count)]

        pre_abort_metrics = await _wait_for_lmdeploy_activity(
            server_urls,
            min_active=min(self.max_batch_size + 1, self.request_count),
            timeout_s=self.startup_activity_timeout_s,
        )
        self.assertGreater(
            pre_abort_metrics.get("api_routed", 0)
            + pre_abort_metrics.get("api_waiting", 0)
            + pre_abort_metrics.get("engine_running", 0)
            + pre_abort_metrics.get("engine_waiting", 0),
            0,
            msg=f"LMDeploy did not show active requests before abort. metrics={pre_abort_metrics}",
        )

        abort_start = time.perf_counter()
        await rollout_ctl.pause_generation.remote()  # type: ignore[attr-defined]
        post_abort_metrics = await _wait_for_lmdeploy_idle(
            server_urls,
            timeout_s=PRODUCER_PAUSE_PENDING_TASK_TIMEOUT_S,
        )
        abort_elapsed = time.perf_counter() - abort_start

        results = await asyncio.wait_for(
            asyncio.gather(*refs, return_exceptions=True),
            timeout=PRODUCER_PAUSE_PENDING_TASK_TIMEOUT_S,
        )
        exceptions = [result for result in results if isinstance(result, BaseException)]
        self.assertFalse(exceptions, msg=f"Rollout refs raised after abort: {exceptions[:3]}")

        statuses = [result.status for result in results]
        self.assertIn(Status.ABORTED, statuses, msg=f"Expected at least one aborted rollout, got statuses={statuses}")
        self.assertLess(
            abort_elapsed,
            PRODUCER_PAUSE_PENDING_TASK_TIMEOUT_S,
            msg=f"LMDeploy abort exceeded deadline. elapsed={abort_elapsed:.2f}s metrics={post_abort_metrics}",
        )
        self.assertEqual(post_abort_metrics["api_routed"], 0, msg=f"metrics after abort: {post_abort_metrics}")
        self.assertEqual(post_abort_metrics["api_waiting"], 0, msg=f"metrics after abort: {post_abort_metrics}")
        self.assertEqual(post_abort_metrics["engine_running"], 0, msg=f"metrics after abort: {post_abort_metrics}")
        self.assertEqual(post_abort_metrics["engine_waiting"], 0, msg=f"metrics after abort: {post_abort_metrics}")

    async def test_delayed_abort_returns_all_lmdeploy_requests_within_deadline(self):
        rollout_ctl = self._build_rollout_controller()
        refs = [rollout_ctl.generate.remote(self._make_state(i)) for i in range(self.request_count)]

        await asyncio.sleep(self.abort_after_s)
        abort_start = time.perf_counter()
        await rollout_ctl.pause_generation.remote()  # type: ignore[attr-defined]
        abort_elapsed = time.perf_counter() - abort_start
        if abort_elapsed > PRODUCER_PAUSE_PENDING_TASK_TIMEOUT_S:
            self.fail(
                f"LMDeploy pause_generation exceeded {PRODUCER_PAUSE_PENDING_TASK_TIMEOUT_S:.2f}s "
                f"after aborting {self.request_count} requests"
            )
        remaining_deadline_s = PRODUCER_PAUSE_PENDING_TASK_TIMEOUT_S - abort_elapsed

        async def _await_ref(idx: int, ref):
            result = await ref
            return idx, result, time.perf_counter() - abort_start

        try:
            completed = await asyncio.wait_for(
                asyncio.gather(*[_await_ref(i, ref) for i, ref in enumerate(refs)], return_exceptions=True),
                timeout=remaining_deadline_s,
            )
        except asyncio.TimeoutError:
            self.fail(
                f"LMDeploy requests did not all return within {PRODUCER_PAUSE_PENDING_TASK_TIMEOUT_S:.2f}s "
                f"after aborting {self.request_count} requests"
            )

        exceptions = [item for item in completed if isinstance(item, BaseException)]
        self.assertFalse(exceptions, msg=f"Rollout refs raised after delayed abort: {exceptions[:3]}")

        request_results = [(idx, result, elapsed) for idx, result, elapsed in completed]
        slow_requests = [
            (idx, elapsed)
            for idx, _result, elapsed in request_results
            if elapsed > PRODUCER_PAUSE_PENDING_TASK_TIMEOUT_S
        ]
        self.assertFalse(
            slow_requests,
            msg=f"Requests exceeded {PRODUCER_PAUSE_PENDING_TASK_TIMEOUT_S:.2f}s after abort: {slow_requests[:10]}",
        )
        statuses = [result.status for _idx, result, _elapsed in request_results]
        self.assertIn(Status.ABORTED, statuses, msg=f"Expected at least one aborted rollout, got statuses={statuses}")


if __name__ == "__main__":
    unittest.main()
