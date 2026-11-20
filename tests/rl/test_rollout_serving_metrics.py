"""Rollout serving 指标抓取测试。

Good Tests:
- 用 pin 版 lmdeploy 实际生成的 /metrics 文本锁定 parse_serving_metrics 的白名单行为。
- aggregate_serving_metrics 的 gauge 均值 / 计数求和，以及失败端点（空 dict）不稀释 gauge。
- 按端点差分、抓取失败保留 baseline、恢复跨窗口标记、counter reset 和加权直方图均值。
- worker scrape 与异步 controller collect 的 fail-open 行为（不阻塞后台生产）。

Bad Tests:
- 不启动真实推理服务，不用 Ray 集群；远程调用用可等待替身。
"""

import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from xtuner.v1.rl.agent_loop_manager import ProduceBatchResult
from xtuner.v1.rl.rollout.controller import RolloutController
from xtuner.v1.rl.rollout.worker import (
    SERVING_GAUGE_KEYS,
    RolloutWorker,
    aggregate_serving_metrics,
    parse_serving_metrics,
)
from xtuner.v1.train.rl_trainer import BaseRLTrainer


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "lmdeploy_metrics.txt"


class TestParseServingMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metrics = parse_serving_metrics(FIXTURE_PATH.read_text())

    def test_fixture_locks_allowlisted_values(self):
        # 行为：真实 lmdeploy exposition 中只取白名单样本，数值精确锁定
        self.assertEqual(self.metrics["requests_running"], 2.0)
        self.assertEqual(self.metrics["requests_waiting"], 1.0)
        self.assertEqual(self.metrics["gpu_cache_usage"], 0.35)
        self.assertEqual(self.metrics["prefix_cache_hit_rate"], 0.5)
        self.assertEqual(self.metrics["prompt_tokens"], 1024.0)
        self.assertEqual(self.metrics["generation_tokens"], 2048.0)
        self.assertEqual(self.metrics["cached_tokens"], 512.0)
        self.assertEqual(self.metrics["ttft_sum"], 0.25)
        self.assertEqual(self.metrics["ttft_count"], 1.0)
        self.assertEqual(self.metrics["tpot_sum"], 0.02)
        self.assertEqual(self.metrics["tpot_count"], 1.0)
        self.assertEqual(self.metrics["e2e_sum"], 3.5)
        self.assertEqual(self.metrics["e2e_count"], 1.0)

    def test_request_success_split_by_finish_reason(self):
        # 行为：request_success_total 聚合为 finished，并按 reason 拆分（去掉 Enum 前缀）
        self.assertEqual(self.metrics["finished"], 10.0)
        self.assertEqual(self.metrics["finished_finish"], 9.0)
        self.assertEqual(self.metrics["finished_cancel"], 1.0)

    def test_default_collectors_are_ignored(self):
        # 行为：python_gc / process 等默认 collector 不进结果
        for key in self.metrics:
            self.assertNotIn("python_", key)
            self.assertNotIn("process_", key)
        self.assertNotIn("iteration_tokens_sum", self.metrics)

    def test_gauge_keys_cover_all_non_cumulative_outputs(self):
        # 行为：除 gauge 集合外的 key 都是累计值，由消费方做差
        self.assertEqual(
            SERVING_GAUGE_KEYS,
            {"requests_running", "requests_waiting", "gpu_cache_usage", "prefix_cache_hit_rate"},
        )

    def test_empty_and_malformed_text_fail_open(self):
        self.assertEqual(parse_serving_metrics(""), {})
        self.assertEqual(parse_serving_metrics("not prometheus text {{{"), {})

    def test_missing_parser_returns_empty(self):
        # 行为：prometheus_client 不可用时返回空，不抛异常
        with patch.dict(sys.modules, {"prometheus_client.parser": None}):
            self.assertEqual(parse_serving_metrics(FIXTURE_PATH.read_text()), {})


class TestAggregateServingMetrics(unittest.TestCase):
    def test_gauges_average_and_counters_sum(self):
        merged = aggregate_serving_metrics(
            [
                {"requests_running": 2.0, "prompt_tokens": 100.0},
                {"requests_running": 4.0, "prompt_tokens": 50.0},
            ]
        )
        self.assertEqual(merged["requests_running"], 3.0)
        self.assertEqual(merged["prompt_tokens"], 150.0)

    def test_failed_endpoint_does_not_dilute_gauge_average(self):
        # 行为：抓取失败的端点贡献空 dict，gauge 均值只在有值的端点上算
        merged = aggregate_serving_metrics([{"gpu_cache_usage": 0.4}, {}, {"gpu_cache_usage": 0.8}])
        self.assertAlmostEqual(merged["gpu_cache_usage"], 0.6)


class TestServingStepValues(unittest.TestCase):
    def setUp(self) -> None:
        self.trainer = BaseRLTrainer.__new__(BaseRLTrainer)
        self.trainer._last_serving_metrics = {}

    def test_first_scrape_establishes_baseline_without_inventing_deltas(self) -> None:
        values = self.trainer._serving_step_values({"a": {"prompt_tokens": 100.0, "requests_running": 3.0}}, 0)
        self.assertNotIn("prompt_tokens", values)
        self.assertNotIn("ttft_mean", values)
        self.assertEqual(values["requests_running"], 3.0)
        self.assertEqual(values["window_steps"], 0.0)

    def test_counter_deltas_and_gauge_passthrough(self) -> None:
        self.trainer._serving_step_values({"a": {"prompt_tokens": 100.0, "requests_running": 3.0}}, 0)
        values = self.trainer._serving_step_values({"a": {"prompt_tokens": 130.0, "requests_running": 1.0}}, 1)
        self.assertEqual(values["prompt_tokens"], 30.0)
        self.assertEqual(values["requests_running"], 1.0)
        self.assertEqual(values["window_steps"], 1.0)

    def test_partial_failure_does_not_manufacture_counter_growth(self) -> None:
        self.trainer._serving_step_values({"a": {"prompt_tokens": 100.0}, "b": {"prompt_tokens": 100.0}}, 0)
        partial = self.trainer._serving_step_values({"a": {"prompt_tokens": 110.0}, "b": {}}, 1)
        recovered = self.trainer._serving_step_values(
            {"a": {"prompt_tokens": 120.0}, "b": {"prompt_tokens": 120.0}}, 2
        )
        self.assertEqual(partial["prompt_tokens"], 10.0)
        self.assertEqual(recovered["prompt_tokens"], 30.0)
        self.assertEqual(partial["prompt_tokens"] + recovered["prompt_tokens"], 40.0)
        self.assertEqual(recovered["window_steps"], 2.0)

    def test_failed_scrapes_preserve_tokens_requests_and_histograms(self) -> None:
        self.trainer._serving_step_values(
            {"a": {"prompt_tokens": 100.0, "finished": 10.0, "ttft_sum": 5.0, "ttft_count": 10.0}}, 0
        )
        self.assertEqual(self.trainer._serving_step_values(None, 1), {})
        self.assertEqual(self.trainer._serving_step_values({"a": {}}, 2), {})
        values = self.trainer._serving_step_values(
            {"a": {"prompt_tokens": 140.0, "finished": 14.0, "ttft_sum": 7.0, "ttft_count": 14.0}}, 3
        )
        self.assertEqual(values["prompt_tokens"], 40.0)
        self.assertEqual(values["finished"], 4.0)
        self.assertEqual(values["ttft_mean"], 0.5)
        self.assertEqual(values["window_steps"], 3.0)

    def test_counter_reset_does_not_cancel_other_endpoint_growth(self) -> None:
        self.trainer._serving_step_values({"a": {"prompt_tokens": 500.0}, "b": {"prompt_tokens": 100.0}}, 0)
        values = self.trainer._serving_step_values({"a": {"prompt_tokens": 20.0}, "b": {"prompt_tokens": 110.0}}, 1)
        self.assertEqual(values["prompt_tokens"], 30.0)

    def test_new_endpoint_and_new_key_only_establish_baselines(self) -> None:
        self.trainer._serving_step_values({"a": {"prompt_tokens": 10.0}}, 0)
        values = self.trainer._serving_step_values(
            {"a": {"prompt_tokens": 15.0, "finished": 4.0}, "b": {"prompt_tokens": 1000.0}}, 1
        )
        self.assertNotIn("finished", values)
        self.assertEqual(values["prompt_tokens"], 5.0)

    def test_removed_endpoint_is_not_reused_as_a_baseline(self) -> None:
        self.trainer._serving_step_values({"a": {"prompt_tokens": 10.0}}, 0)
        self.assertEqual(self.trainer._serving_step_values({}, 1), {})
        self.assertEqual(self.trainer._serving_step_values({"a": {"prompt_tokens": 100.0}}, 2), {})

    def test_histogram_means_are_weighted_by_observation_count(self) -> None:
        self.trainer._serving_step_values(
            {"a": {"ttft_sum": 10.0, "ttft_count": 20.0}, "b": {"ttft_sum": 2.0, "ttft_count": 2.0}}, 0
        )
        values = self.trainer._serving_step_values(
            {"a": {"ttft_sum": 16.0, "ttft_count": 30.0}, "b": {"ttft_sum": 6.0, "ttft_count": 4.0}}, 1
        )
        self.assertAlmostEqual(values["ttft_mean"], 10.0 / 12.0)
        self.assertNotIn("ttft_sum", values)
        self.assertNotIn("ttft_count", values)

    def test_histogram_reset_uses_matching_sum_and_count(self) -> None:
        # Count decreased but sum increased: both fields must use the new lifetime.
        self.trainer._serving_step_values({"a": {"ttft_sum": 10.0, "ttft_count": 20.0}}, 0)
        values = self.trainer._serving_step_values({"a": {"ttft_sum": 12.0, "ttft_count": 2.0}}, 1)
        self.assertEqual(values["ttft_mean"], 6.0)

    def test_no_observations_or_missing_sum_does_not_report_zero_latency(self) -> None:
        self.trainer._serving_step_values({"a": {"ttft_sum": 10.0, "ttft_count": 20.0, "e2e_count": 1.0}}, 0)
        values = self.trainer._serving_step_values({"a": {"ttft_sum": 10.0, "ttft_count": 20.0, "e2e_count": 2.0}}, 1)
        self.assertEqual(values, {})


class TestServingMetricLogging(unittest.TestCase):
    def test_logging_does_not_scrape_or_fabricate_missing_metrics(self) -> None:
        trainer = BaseRLTrainer.__new__(BaseRLTrainer)
        trainer.logger = MagicMock()
        trainer._total_train_steps = 1
        trainer._exp_tracker = MagicMock()
        trainer.rollout_controller = MagicMock()
        for serving_values in (None, {}, {"generation_tokens": 40.0, "window_steps": 2.0}):
            with self.subTest(serving_values=serving_values):
                trainer._log_step(
                    1, {}, ProduceBatchResult(rollout_states=[]), {}, {}, serving_step_values=serving_values
                )
                scalars = trainer._exp_tracker.add_scalars.call_args.kwargs["tag_scalar_dict"]
                self.assertEqual(
                    {key: value for key, value in scalars.items() if key.startswith("serving/")},
                    {f"serving/{key}": value for key, value in (serving_values or {}).items()},
                )
        trainer.rollout_controller.collect_server_metrics.remote.assert_not_called()


class TestTrainerCollectServingMetrics(unittest.IsolatedAsyncioTestCase):
    async def test_collects_without_blocking_other_tasks(self) -> None:
        trainer = BaseRLTrainer.__new__(BaseRLTrainer)
        producer_progress = asyncio.Event()

        async def scrape() -> dict[str, dict[str, float]]:
            await producer_progress.wait()
            return {"a": {"prompt_tokens": 3.0}}

        trainer.rollout_controller = SimpleNamespace(collect_server_metrics=SimpleNamespace(remote=scrape))
        task = asyncio.create_task(trainer._collect_serving_metrics())
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        producer_progress.set()
        with patch("xtuner.v1.train.rl_trainer.ray.get", side_effect=AssertionError("blocking ray.get")):
            self.assertEqual(await asyncio.wait_for(task, 1.0), {"a": {"prompt_tokens": 3.0}})

    async def test_controller_without_method_fails_open(self) -> None:
        trainer = BaseRLTrainer.__new__(BaseRLTrainer)
        trainer.rollout_controller = SimpleNamespace()
        self.assertIsNone(await trainer._collect_serving_metrics())

    async def test_remote_failure_is_not_an_empty_registry(self) -> None:
        trainer = BaseRLTrainer.__new__(BaseRLTrainer)
        trainer.rollout_controller = SimpleNamespace(
            collect_server_metrics=SimpleNamespace(remote=AsyncMock(side_effect=RuntimeError("actor down")))
        )
        self.assertIsNone(await trainer._collect_serving_metrics())

    async def test_timeout_is_bounded_and_cleans_up_local_waiter(self) -> None:
        trainer = BaseRLTrainer.__new__(BaseRLTrainer)
        cancelled = asyncio.Event()

        async def scrape() -> dict[str, dict[str, float]]:
            try:
                await asyncio.Event().wait()
                return {}
            finally:
                cancelled.set()

        trainer.rollout_controller = SimpleNamespace(collect_server_metrics=SimpleNamespace(remote=scrape))
        with patch("xtuner.v1.train.rl_trainer.SERVING_METRICS_GATHER_TIMEOUT_S", 0.01):
            self.assertIsNone(await trainer._collect_serving_metrics())
        self.assertTrue(cancelled.is_set())

    async def test_cancellation_is_not_swallowed(self) -> None:
        trainer = BaseRLTrainer.__new__(BaseRLTrainer)
        trainer.rollout_controller = SimpleNamespace(
            collect_server_metrics=SimpleNamespace(remote=AsyncMock(side_effect=asyncio.CancelledError))
        )
        with self.assertRaises(asyncio.CancelledError):
            await trainer._collect_serving_metrics()


def _bare_worker(server_url: str = "http://127.0.0.1:12345") -> RolloutWorker:
    worker = RolloutWorker.__new__(RolloutWorker)
    worker.server_url = server_url
    worker.logger = MagicMock()
    worker._last_scrape_warning_s = 0.0
    return worker


class TestWorkerScrapeMetrics(unittest.IsolatedAsyncioTestCase):
    async def test_successful_scrape_parses_response(self) -> None:
        worker = _bare_worker()
        response = httpx.Response(
            200, text=FIXTURE_PATH.read_text(), request=httpx.Request("GET", f"{worker.server_url}/metrics")
        )
        with patch("xtuner.v1.rl.rollout.worker.httpx.AsyncClient") as client_cls:
            client = client_cls.return_value.__aenter__.return_value
            client.get.return_value = response
            metrics = await worker.scrape_metrics()
        self.assertEqual(metrics["requests_running"], 2.0)
        client_cls.assert_called_once_with(timeout=2.0)
        client.get.assert_awaited_once_with("http://127.0.0.1:12345/metrics")
        client_cls.return_value.__aexit__.assert_awaited_once()

    async def test_network_failure_returns_empty_and_warns_once(self) -> None:
        # 行为：请求失败 fail-open；限频内重复失败只告警一次
        worker = _bare_worker()
        with (
            patch("xtuner.v1.rl.rollout.worker.httpx.AsyncClient") as client_cls,
            patch("xtuner.v1.rl.rollout.worker.time.monotonic", return_value=1000.0),
        ):
            client = client_cls.return_value.__aenter__.return_value
            client.get.side_effect = httpx.ConnectError("down")
            self.assertEqual(await worker.scrape_metrics(), {})
            self.assertEqual(await worker.scrape_metrics(), {})
        self.assertEqual(worker.logger.warning.call_count, 1)

    async def test_timeout_and_http_error_fail_open(self) -> None:
        response = httpx.Response(503, request=httpx.Request("GET", "http://test/metrics"))
        for error in (
            httpx.ReadTimeout("slow server"),
            httpx.HTTPStatusError("unavailable", request=response.request, response=response),
        ):
            with self.subTest(error=error), patch("xtuner.v1.rl.rollout.worker.httpx.AsyncClient") as client_cls:
                client_cls.return_value.__aenter__.return_value.get.side_effect = error
                self.assertEqual(await _bare_worker().scrape_metrics(), {})

    async def test_slow_scrape_keeps_pause_and_continue_schedulable(self) -> None:
        worker = _bare_worker()
        worker.receive_abort_request = asyncio.Event()
        scrape_started = asyncio.Event()
        release_scrape = asyncio.Event()
        abort_received = asyncio.Event()

        async def handle(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/metrics":
                scrape_started.set()
                await release_scrape.wait()
                return httpx.Response(200, text=FIXTURE_PATH.read_text())
            self.assertEqual(request.url.path, "/abort_request")
            self.assertTrue(worker.receive_abort_request.is_set())
            abort_received.set()
            return httpx.Response(200)

        # Exercise the real HTTP await and pause implementation on the same
        # event loop; no Ray cluster or inference server is needed.
        client_cls = httpx.AsyncClient
        with (
            patch(
                "xtuner.v1.rl.rollout.worker.httpx.AsyncClient",
                side_effect=lambda **kwargs: client_cls(transport=httpx.MockTransport(handle), **kwargs),
            ),
            patch("xtuner.v1.rl.rollout.worker.requests.get", side_effect=AssertionError("blocking metrics I/O")),
        ):
            task = asyncio.create_task(worker.scrape_metrics())
            try:
                await asyncio.wait_for(scrape_started.wait(), 1.0)
                self.assertTrue(await asyncio.wait_for(worker.pause_generation(), 1.0))
                self.assertTrue(abort_received.is_set())
                self.assertFalse(task.done())
                worker.continue_generation()
                self.assertFalse(worker.receive_abort_request.is_set())
                release_scrape.set()
                self.assertEqual((await asyncio.wait_for(task, 1.0))["requests_running"], 2.0)
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def test_cancellation_closes_client_and_is_not_swallowed(self) -> None:
        with patch("xtuner.v1.rl.rollout.worker.httpx.AsyncClient") as client_cls:
            client_cls.return_value.__aenter__.return_value.get.side_effect = asyncio.CancelledError
            with self.assertRaises(asyncio.CancelledError):
                await _bare_worker().scrape_metrics()
            client_cls.return_value.__aexit__.assert_awaited_once()


class _FakeEndpoint:
    def __init__(self, url: str, metrics: dict[str, float] | Exception, *, active: bool = True) -> None:
        self.url = url
        self.is_request_entrypoint = True
        self.is_active = lambda: active
        remote = AsyncMock(side_effect=metrics) if isinstance(metrics, Exception) else AsyncMock(return_value=metrics)
        self.actor = SimpleNamespace(scrape_metrics=SimpleNamespace(remote=remote))


class TestControllerCollectServerMetrics(unittest.IsolatedAsyncioTestCase):
    def _controller(self, entrypoints: list[_FakeEndpoint]) -> RolloutController:
        controller = RolloutController.__new__(RolloutController)
        controller.registry = SimpleNamespace(all_workers=lambda: entrypoints)
        controller.logger = MagicMock()
        return controller

    async def test_preserves_endpoint_identity(self) -> None:
        controller = self._controller(
            [
                _FakeEndpoint("a", {"requests_running": 2.0, "prompt_tokens": 100.0}),
                _FakeEndpoint("b", {"requests_running": 6.0, "prompt_tokens": 50.0}),
            ]
        )
        with patch("xtuner.v1.rl.rollout.controller.ray.get", side_effect=AssertionError("blocking ray.get")):
            metrics = await controller.collect_server_metrics()
        self.assertEqual(metrics["a"]["prompt_tokens"], 100.0)
        self.assertEqual(metrics["b"]["prompt_tokens"], 50.0)

    async def test_no_entrypoints_returns_empty(self) -> None:
        self.assertEqual(await self._controller([]).collect_server_metrics(), {})

    async def test_failed_and_inactive_endpoints_do_not_discard_healthy_results(self) -> None:
        inactive = _FakeEndpoint("inactive", {}, active=False)
        non_entrypoint = _FakeEndpoint("non-entrypoint", {})
        non_entrypoint.is_request_entrypoint = False
        controller = self._controller(
            [
                _FakeEndpoint("a", {"prompt_tokens": 100.0}),
                _FakeEndpoint("b", RuntimeError("actor died")),
                inactive,
                non_entrypoint,
            ]
        )
        self.assertEqual(
            await controller.collect_server_metrics(),
            {
                "a": {"prompt_tokens": 100.0},
                "b": {},
                "inactive": {},
            },
        )
        inactive.actor.scrape_metrics.remote.assert_not_called()
        non_entrypoint.actor.scrape_metrics.remote.assert_not_called()

    async def test_slow_endpoint_times_out_without_discarding_healthy_results(self) -> None:
        slow = _FakeEndpoint("slow", {})
        cancelled = asyncio.Event()

        async def scrape() -> dict[str, float]:
            try:
                await asyncio.Event().wait()
                return {}
            finally:
                cancelled.set()

        slow.actor.scrape_metrics.remote = scrape
        controller = self._controller([slow, _FakeEndpoint("a", {"prompt_tokens": 100.0})])
        with patch("xtuner.v1.rl.rollout.controller.SERVING_METRICS_GATHER_TIMEOUT_S", 0.01):
            metrics = await controller.collect_server_metrics()
        self.assertEqual(metrics, {"slow": {}, "a": {"prompt_tokens": 100.0}})
        self.assertTrue(cancelled.is_set())


if __name__ == "__main__":
    unittest.main()
