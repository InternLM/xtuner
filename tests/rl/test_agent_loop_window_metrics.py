"""Agent-loop 窗口自定义指标通道的契约测试。

覆盖的 public 行为:
- ``merge_metric_dicts`` 按逐 key 求和合并多个指标字典。
- ``derive_window_means`` 从 ``<name>_sum``/``<name>_count`` 对派生 ``<name>_mean``。
- ``collect_metrics`` 遵循 drain 契约：返回并清零，重复调用不重复报告旧窗口。
- ``RouterAgentLoop.collect_metrics`` 扇出所有 worker 并求和合并。
- ``AgentLoopActor.collect_metrics`` 委托内部 loop。
- ``collect_agent_loop_metrics`` 对本地 loop / Ray 代理统一分发。
- ``_log_step`` 把 ``ProduceBatchResult.custom_scalars`` 派生均值后写出 ``agent/*`` 标量。
"""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from xtuner.v1.rl.agent_loop import (
    AgentLoop,
    AgentLoopActor,
    RouterAgentLoop,
    collect_agent_loop_metrics,
    merge_metric_dicts,
)
from xtuner.v1.rl.agent_loop_manager.produce_utils import ProduceBatchResult, derive_window_means


class _CountingLoop(AgentLoop):
    """最小 AgentLoop 子类，绕过重型 __init__，只实现 drain 语义。"""

    def __init__(self) -> None:  # noqa: D107 - test stub
        self._pending: dict[str, float] = {}

    async def generate_sample(self, rollout_state, **kwargs):  # noqa: ANN001, ANN202 - abstract stub
        raise NotImplementedError

    async def collect_metrics(self) -> dict[str, float]:
        drained, self._pending = self._pending, {}
        return drained


class _NoMetricsLoop(AgentLoop):
    """不覆写 collect_metrics 的具体子类，验证基类默认返回空。"""

    def __init__(self) -> None:  # noqa: D107 - test stub
        pass

    async def generate_sample(self, rollout_state, **kwargs):  # noqa: ANN001, ANN202 - abstract stub
        raise NotImplementedError


class _FakeWorker:
    """可哈希的 worker 替身，RouterAgentLoop 需要用它做 dict key。"""

    def __init__(self, values: dict[str, float]) -> None:
        self.collect_metrics = SimpleNamespace(remote=AsyncMock(return_value=values))


def _fake_worker(values: dict[str, float]) -> SimpleNamespace:
    remote = AsyncMock(return_value=values)
    return SimpleNamespace(collect_metrics=SimpleNamespace(remote=remote))


class _FakeTracker:
    def __init__(self) -> None:  # noqa: D107 - test stub
        self.scalars: dict[int, dict[str, float]] = {}

    def add_scalars(self, tag_scalar_dict: dict[str, float], global_step: int) -> None:
        self.scalars[global_step] = dict(tag_scalar_dict)


class TestMergeMetricDicts(unittest.TestCase):
    def test_merges_by_per_key_summation(self):
        merged = merge_metric_dicts(
            [
                {"episodes_finished": 2.0, "t_sum": 10.0, "t_count": 2.0},
                {"episodes_finished": 3.0, "t_sum": 5.0, "t_count": 1.0},
            ]
        )
        self.assertEqual(merged["episodes_finished"], 5.0)
        self.assertEqual(merged["t_sum"], 15.0)
        self.assertEqual(merged["t_count"], 3.0)

    def test_skips_empty_dictionaries(self):
        merged = merge_metric_dicts([{}, {"episodes_finished": 1.0}, {}])
        self.assertEqual(merged, {"episodes_finished": 1.0})


class TestDeriveWindowMeans(unittest.TestCase):
    def test_derives_mean_from_sum_count_pairs(self):
        derived = derive_window_means({"latency_sum": 9.0, "latency_count": 3.0, "episodes": 4.0})
        self.assertAlmostEqual(derived["latency_mean"], 3.0)
        self.assertEqual(derived["episodes"], 4.0)

    def test_skips_non_positive_count(self):
        derived = derive_window_means({"latency_sum": 9.0, "latency_count": 0.0})
        self.assertNotIn("latency_mean", derived)

    def test_leaves_orphan_sum_untouched(self):
        derived = derive_window_means({"latency_sum": 9.0})
        self.assertNotIn("latency_mean", derived)
        self.assertEqual(derived["latency_sum"], 9.0)


class TestCollectMetricsDrain(unittest.IsolatedAsyncioTestCase):
    async def test_second_call_reports_empty_window(self):
        loop = _CountingLoop()
        loop._pending["episodes_finished"] = 2.0
        self.assertEqual(await loop.collect_metrics(), {"episodes_finished": 2.0})
        self.assertEqual(await loop.collect_metrics(), {})

    async def test_default_agent_loop_reports_nothing(self):
        self.assertEqual(await _NoMetricsLoop().collect_metrics(), {})


class TestRouterCollectMetrics(unittest.IsolatedAsyncioTestCase):
    async def test_fans_out_and_sums_worker_windows(self):
        router = RouterAgentLoop(
            workers=[_FakeWorker({"episodes_finished": 2.0}), _FakeWorker({"episodes_finished": 3.0})],
            rollout_ctl=None,
        )
        self.assertEqual(await router.collect_metrics(), {"episodes_finished": 5.0})


class TestActorDelegation(unittest.IsolatedAsyncioTestCase):
    async def test_delegates_to_inner_loop(self):
        actor = object.__new__(AgentLoopActor)
        actor.agent_loop = _CountingLoop()
        actor.agent_loop._pending["episodes_finished"] = 1.0
        self.assertEqual(await AgentLoopActor.collect_metrics(actor), {"episodes_finished": 1.0})


class TestCollectAgentLoopMetricsDispatch(unittest.IsolatedAsyncioTestCase):
    async def test_local_loop_is_called_directly(self):
        loop = _CountingLoop()
        loop._pending["episodes_finished"] = 4.0
        self.assertEqual(await collect_agent_loop_metrics(loop), {"episodes_finished": 4.0})

    async def test_ray_proxy_is_called_via_remote(self):
        proxy = _fake_worker({"episodes_finished": 7.0})
        self.assertEqual(await collect_agent_loop_metrics(proxy), {"episodes_finished": 7.0})


class TestLogStepWritesAgentScalars(unittest.IsolatedAsyncioTestCase):
    async def test_custom_scalars_land_under_agent_prefix_with_derived_means(self):
        from xtuner.v1.train.rl_trainer import BaseRLTrainer

        tracker = _FakeTracker()
        trainer = object.__new__(BaseRLTrainer)
        trainer._exp_tracker = tracker
        trainer.logger = SimpleNamespace(info=lambda *_: None)
        trainer._display_all_workers_log = False
        trainer._total_train_steps = 1
        produce_result = ProduceBatchResult(
            rollout_states=[],
            custom_scalars={"episodes_finished": 4.0, "t_acquire_sum": 12.0, "t_acquire_count": 4.0},
        )

        trainer._log_step(1, {}, produce_result, None, {})

        scalars = tracker.scalars[1]
        self.assertEqual(scalars["agent/episodes_finished"], 4.0)
        self.assertAlmostEqual(scalars["agent/t_acquire_mean"], 3.0)
        self.assertEqual(scalars["agent/t_acquire_count"], 4.0)


if __name__ == "__main__":
    unittest.main()
