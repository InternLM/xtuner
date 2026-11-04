"""RolloutTraceStore 的行为测试。

TestRolloutTraceCleanup —— session 释放与 ObjectRef 清理行为
TestRolloutTraceStore —— store actor 的 session 生命周期与引用递归释放
TestExportPartialTrainingTrace —— export_partial_training_trace 的部分导出行为
- test_exports_committed_segments_in_order:
    同一 session 的多个 segment 按写入顺序拼接导出,无 MoE 时 routed_experts 为 None。
- test_divergent_session_is_rejected:
    session 出现分叉(重试后轨迹不同)时抛错,而不是把两次尝试拼接在一起。
- test_empty_session_returns_empty_trace:
    不存在的 session 返回空 trace 而不是报错。
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import ray

from xtuner.v1.rl.rollout import trace_store as trace_store_module
from xtuner.v1.rl.rollout.trace_store import (
    RolloutTraceStore,
    TokenizedSegment,
    _free_ray_refs,
    get_existing_store,
    release_and_discard_rollout_groups,
    release_existing_sessions,
)


class TestRolloutTraceCleanup(unittest.TestCase):
    def test_release_and_discard_detaches_only_trace_owned_refs(self):
        trace_owned_ref = object()
        rollout_owned_ref = object()
        trace_owned = SimpleNamespace(session_id="trace-owned", routed_experts=trace_owned_ref)
        rollout_owned = SimpleNamespace(session_id="rollout-owned", routed_experts=rollout_owned_ref)
        routed_experts_seen_by_discard = {}

        def record_discard(item):
            routed_experts_seen_by_discard[item.session_id] = item.routed_experts

        with (
            patch(
                "xtuner.v1.rl.rollout.trace_store.release_existing_sessions",
                new=AsyncMock(return_value={"trace-owned"}),
            ) as release_sessions,
            patch(
                "xtuner.v1.rl.rollout.trace_store.discard_rollout_state",
                side_effect=record_discard,
            ) as discard,
        ):
            asyncio.run(release_and_discard_rollout_groups([[trace_owned, rollout_owned]]))

        release_sessions.assert_awaited_once_with(["trace-owned", "rollout-owned"])
        self.assertIsNone(routed_experts_seen_by_discard["trace-owned"])
        self.assertIs(routed_experts_seen_by_discard["rollout-owned"], rollout_owned_ref)
        self.assertEqual(discard.call_count, 2)

    def test_get_existing_store_returns_none_when_ray_is_uninitialized(self):
        cached_store = object()
        with (
            patch.object(trace_store_module, "_handle_cache", cached_store),
            patch.object(trace_store_module.ray, "is_initialized", return_value=False),
        ):
            self.assertIsNone(get_existing_store())
            self.assertIsNone(trace_store_module._handle_cache)


class TestRolloutTraceStore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.started_ray = False
        try:
            if not ray.is_initialized():
                ray.init(address="local", num_cpus=1, include_dashboard=False, ignore_reinit_error=True)
                cls.started_ray = True
        except Exception as exc:
            raise unittest.SkipTest(f"Ray init failed for trace-store tests: {exc}") from exc

    @classmethod
    def tearDownClass(cls):
        if cls.started_ray and ray.is_initialized():
            ray.shutdown()

    def test_release_sessions_deduplicates_and_skips_missing_ids(self):
        store = RolloutTraceStore.remote()
        try:
            ray.get(store.insert.remote("a", "prompt-a", {"value": 1}))
            ray.get(store.insert.remote("b", "prompt-b", {"value": 2}))

            released = ray.get(store.release_sessions.remote(["a", "missing", "a"]))

            self.assertEqual(released, ["a"])
            self.assertEqual(ray.get(store.list_sessions.remote()), ["b"])
        finally:
            ray.kill(store)

    def test_release_existing_sessions_stably_deduplicates_before_rpc(self):
        release_remote = AsyncMock(return_value=["one"])
        store = SimpleNamespace(release_sessions=SimpleNamespace(remote=release_remote))
        with patch(
            "xtuner.v1.rl.rollout.trace_store.get_existing_store",
            return_value=store,
        ):
            released = asyncio.run(release_existing_sessions(["one", "one", "missing"]))

        self.assertEqual(released, {"one"})
        release_remote.assert_awaited_once_with(["one", "missing"])

    def test_release_existing_sessions_handles_empty_input_and_missing_store(self):
        with patch("xtuner.v1.rl.rollout.trace_store.get_existing_store") as get_store:
            self.assertEqual(asyncio.run(release_existing_sessions([])), set())
            get_store.assert_not_called()

        with patch(
            "xtuner.v1.rl.rollout.trace_store.get_existing_store",
            return_value=None,
        ):
            self.assertEqual(asyncio.run(release_existing_sessions(["missing"])), set())

    def test_free_ray_refs_recurses_into_nested_containers(self):
        object_ref = ray.put({"payload": [1, 2, 3]})
        with patch.object(ray.internal, "free") as free:
            _free_ray_refs({"outer": [({"inner": object_ref},)]})

        free.assert_called_once_with([object_ref], local_only=False)


class TestExportPartialTrainingTrace(unittest.TestCase):
    """export_partial_training_trace 对 session trie 的部分导出行为。"""

    @classmethod
    def setUpClass(cls):
        cls.started_ray = False
        try:
            if not ray.is_initialized():
                ray.init(address="local", num_cpus=1, include_dashboard=False, ignore_reinit_error=True)
                cls.started_ray = True
        except Exception as exc:
            raise unittest.SkipTest(f"Ray init failed for trace-store tests: {exc}") from exc

    @classmethod
    def tearDownClass(cls):
        if cls.started_ray and ray.is_initialized():
            ray.shutdown()

    def test_exports_committed_segments_in_order(self):
        # 行为:同一 session 的多个 segment 按写入顺序拼接导出
        store = RolloutTraceStore.remote()
        try:
            ray.get(
                store.insert.remote(
                    "s",
                    "hello ",
                    TokenizedSegment(text="hello ", token_ids=[1, 2], labels=[-100, 5], logprobs=[0.0, 0.1]),
                )
            )
            ray.get(
                store.insert.remote(
                    "s",
                    "hello world",
                    TokenizedSegment(text="world", token_ids=[3, 4], labels=[6, 7], logprobs=[0.2, 0.3]),
                )
            )

            trace = ray.get(store.export_partial_training_trace.remote("s"))

            self.assertEqual(trace["input_ids"], [1, 2, 3, 4])
            self.assertEqual(trace["labels"], [-100, 5, 6, 7])
            self.assertEqual(trace["logprobs"], [0.0, 0.1, 0.2, 0.3])
            self.assertIsNone(trace["routed_experts"])
        finally:
            ray.kill(store)

    def test_divergent_session_is_rejected(self):
        # 行为:同一 session 出现分叉(重试后轨迹不同)时拒绝导出,而不是拼接两次尝试
        store = RolloutTraceStore.remote()
        try:
            ray.get(
                store.insert.remote(
                    "s",
                    "hello ",
                    TokenizedSegment(text="hello ", token_ids=[1, 2], labels=[-100, 5], logprobs=[0.0, 0.1]),
                )
            )
            ray.get(
                store.insert.remote(
                    "s",
                    "hello world",
                    TokenizedSegment(text="world", token_ids=[3, 4], labels=[6, 7], logprobs=[0.2, 0.3]),
                )
            )
            # 重试产生第二条不同的 assistant 轨迹,在 "hello " 节点下形成兄弟分支。
            ray.get(
                store.insert.remote(
                    "s",
                    "hello there",
                    TokenizedSegment(text="there", token_ids=[8, 9], labels=[10, 11], logprobs=[0.4, 0.5]),
                )
            )

            with self.assertRaises(ray.exceptions.RayTaskError) as ctx:
                ray.get(store.export_partial_training_trace.remote("s"))

            self.assertIn("divergent branches", str(ctx.exception))
        finally:
            ray.kill(store)

    def test_empty_session_returns_empty_trace(self):
        # 行为:不存在的 session 返回空 trace 而不是报错,也不会创建空 session
        store = RolloutTraceStore.remote()
        try:
            trace = ray.get(store.export_partial_training_trace.remote("missing"))

            self.assertEqual(trace["input_ids"], [])
            self.assertEqual(trace["labels"], [])
            self.assertIsNone(trace["routed_experts"])
            self.assertEqual(ray.get(store.list_sessions.remote()), [])
        finally:
            ray.kill(store)
