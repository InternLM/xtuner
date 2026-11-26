"""RolloutTraceStore 的行为测试。

TestRolloutTraceCleanup —— session 释放与 ObjectRef 清理行为
TestRolloutTraceStore —— store actor 的 session 生命周期与引用递归释放
TestExportPartialTrainingTrace —— export_partial_training_trace 的部分导出行为
- test_exports_committed_segments_in_order:
    同一 session 的多个 segment 按写入顺序拼接导出,无 MoE 时 routed_experts 为 None。
- test_divergent_session_exports_latest_chain:
    session 出现分叉(过期重放复用同一 session id)时导出最新写入的分支链
    (同一时钟内并列时退回 token 数最多的分支),而不是拒绝或选拼接/最旧链。
- test_overwritten_prefix_refreshes_chain_timestamp:
    重试覆盖写同一 key 时刷新节点时间戳,导出本次 attempt 的链而不是更旧更长的旧分支。
- test_tool_call_count_summed_on_export / ..._legacy_node_reports_unknown:
    tool_call_count 随 assistant 段写入、沿导出链求和;旧节点缺字段时返回 None。
- test_response_ids_are_contiguous_suffix_after_session_prompt:
    response_ids 从会话首个 prompt 段之后连续切出,tool-result 段保留(不按 label 过滤);
    同时返回首段 prompt_ids,满足 input_ids == prompt_ids + response_ids。
- test_empty_session_returns_empty_trace:
    不存在的 session 返回空 trace 而不是报错。
"""

import asyncio
import time
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
                    TokenizedSegment(text="hello ", token_ids=[1, 2], labels=[-100, -100], logprobs=[0.0, 0.1]),
                )
            )
            ray.get(
                store.insert.remote(
                    "s",
                    "hello world",
                    TokenizedSegment(text="world", token_ids=[3, 4], labels=[6, 7], logprobs=[0.2, 0.3],
                                     tool_call_count=2),
                )
            )

            trace = ray.get(store.export_partial_training_trace.remote("s"))

            self.assertEqual(trace["input_ids"], [1, 2, 3, 4])
            self.assertEqual(trace["labels"], [-100, -100, 6, 7])
            self.assertEqual(trace["logprobs"], [0.0, 0.1, 0.2, 0.3])
            self.assertIsNone(trace["routed_experts"])
            self.assertEqual(trace["tool_call_count"], 2)
            # 首个节点是会话 prompt 段,其余是连续响应后缀
            self.assertEqual(trace["prompt_ids"], [1, 2])
            self.assertEqual(trace["response_ids"], [3, 4])
        finally:
            ray.kill(store)

    def test_divergent_session_exports_latest_chain(self):
        # 行为:同一 session 出现分叉(过期重放复用同一 session id)时导出最新写入的
        # 分支链 —— 即使旧分支更长。导出链必须与最后一次尝试一致,否则训练 token
        # 与 rollout 元数据(reward/artifacts)描述的不是同一次尝试。
        store = RolloutTraceStore.remote()
        try:
            ray.get(
                store.insert.remote(
                    "s",
                    "hello ",
                    TokenizedSegment(text="hello ", token_ids=[1, 2], labels=[-100, -100], logprobs=[0.0, 0.1]),
                )
            )
            # 第一次尝试:长分支,先写入。
            ray.get(
                store.insert.remote(
                    "s",
                    "hello there friend",
                    TokenizedSegment(
                        text="there friend", token_ids=[8, 9, 10, 11], labels=[12, 13, 14, 15],
                        logprobs=[0.4, 0.5, 0.6, 0.7],
                    ),
                )
            )
            # 第二次尝试:更短,但更新 —— 在 "hello " 节点下形成兄弟分支。
            ray.get(
                store.insert.remote(
                    "s",
                    "hello world",
                    TokenizedSegment(text="world", token_ids=[3, 4], labels=[6, 7], logprobs=[0.2, 0.3]),
                )
            )
            # 分开写入,保证两次尝试的 written_at 不落在同一时刻。
            time.sleep(0.01)

            trace = ray.get(store.export_partial_training_trace.remote("s"))

            # 最新链是 "hello " + "world",而不是更长的 "there friend" 分支。
            self.assertEqual(trace["input_ids"], [1, 2, 3, 4])
            self.assertEqual(trace["labels"], [-100, -100, 6, 7])
            self.assertIsNone(trace["routed_experts"])
            self.assertEqual(trace["prompt_ids"], [1, 2])
            self.assertEqual(trace["response_ids"], [3, 4])
        finally:
            ray.kill(store)

    def test_overwritten_prefix_refreshes_chain_timestamp(self):
        # 回归:重试覆盖写同一 key 时必须刷新节点时间戳。否则“选最新链”会挑到
        # 更旧但更长的旧分支,导出上一次 attempt 的 turn,而 reward/artifacts
        # 描述的是本次 attempt。
        store = RolloutTraceStore.remote()
        try:
            ray.get(
                store.insert.remote(
                    "s",
                    "P",
                    TokenizedSegment(text="P", token_ids=[1, 2], labels=[-100, -100], logprobs=[0.0, 0.0]),
                )
            )
            # attempt 1:长链 P -> a1 -> a2 -> a3,先写入。
            ray.get(
                store.insert.remote(
                    "s",
                    "P a1",
                    TokenizedSegment(text="a1", token_ids=[3], labels=[3], logprobs=[0.1], tool_call_count=1),
                )
            )
            ray.get(
                store.insert.remote(
                    "s",
                    "P a1 a2",
                    TokenizedSegment(text="a2", token_ids=[4], labels=[4], logprobs=[0.2], tool_call_count=1),
                )
            )
            ray.get(
                store.insert.remote(
                    "s",
                    "P a1 a2 a3",
                    TokenizedSegment(text="a3", token_ids=[5], labels=[5], logprobs=[0.3], tool_call_count=1),
                )
            )
            time.sleep(0.01)
            # attempt 2:与上次逐字相同的前缀覆盖写,随后超时(只写完 a1)。
            ray.get(
                store.insert.remote(
                    "s",
                    "P a1",
                    TokenizedSegment(text="a1", token_ids=[3], labels=[3], logprobs=[0.1], tool_call_count=1),
                )
            )

            trace = ray.get(store.export_partial_training_trace.remote("s"))

            # 覆盖写刷新 written_at 后,P -> a1 才是“最后写入”;旧分支更长也不选。
            self.assertEqual(trace["input_ids"], [1, 2, 3])
            self.assertEqual(trace["labels"], [-100, -100, 3])
            self.assertEqual(trace["prompt_ids"], [1, 2])
            self.assertEqual(trace["response_ids"], [3])
            self.assertEqual(trace["tool_call_count"], 1)
        finally:
            ray.kill(store)

    def test_tool_call_count_summed_on_export(self):
        # 行为:assistant 段携带 tool_call_count,导出时沿链求和;prompt 增量段不参与
        # (prompt 段全掩码,与真实链路一致)
        store = RolloutTraceStore.remote()
        try:
            ray.get(
                store.insert.remote(
                    "s",
                    "hello ",
                    TokenizedSegment(text="hello ", token_ids=[1, 2], labels=[-100, -100], logprobs=[0.0, 0.1]),
                )
            )
            ray.get(
                store.insert.remote(
                    "s",
                    "hello world",
                    TokenizedSegment(text="world", token_ids=[3, 4], labels=[6, 7], logprobs=[0.2, 0.3],
                                     tool_call_count=3),
                )
            )

            trace = ray.get(store.export_partial_training_trace.remote("s"))

            self.assertEqual(trace["tool_call_count"], 3)
            self.assertEqual(trace["prompt_ids"], [1, 2])
            self.assertEqual(trace["response_ids"], [3, 4])
        finally:
            ray.kill(store)

    def test_tool_call_count_legacy_node_reports_unknown(self):
        # 行为:链上存在缺 tool_call_count 字段的旧 assistant 段(有可训练 label)时,
        # 导出返回 None 而不是静默少计
        store = RolloutTraceStore.remote()
        try:
            ray.get(
                store.insert.remote(
                    "s",
                    "hello ",
                    TokenizedSegment(text="hello ", token_ids=[1, 2], labels=[-100, -100], logprobs=[0.0, 0.1]),
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

            self.assertIsNone(trace["tool_call_count"])
            self.assertEqual(trace["prompt_ids"], [1, 2])
            self.assertEqual(trace["response_ids"], [3, 4])
        finally:
            ray.kill(store)

    def test_empty_session_returns_empty_trace(self):
        # 行为:不存在的 session 返回空 trace 而不是报错,也不会创建空 session
        store = RolloutTraceStore.remote()
        try:
            trace = ray.get(store.export_partial_training_trace.remote("missing"))

            self.assertEqual(trace["input_ids"], [])
            self.assertEqual(trace["labels"], [])
            self.assertEqual(trace["prompt_ids"], [])
            self.assertEqual(trace["response_ids"], [])
            self.assertIsNone(trace["routed_experts"])
            self.assertEqual(ray.get(store.list_sessions.remote()), [])
        finally:
            ray.kill(store)

    def test_response_ids_are_contiguous_suffix_after_session_prompt(self):
        # 回归:response_ids 必须是会话初始 prompt 段之后的连续后缀,tool-result 段
        # (labels 全 -100)也要保留。按监督 label 过滤会得到带洞子序列,使下游
        # token-staleness mask 的 len(labels)-len(response_ids) offset 错位。
        store = RolloutTraceStore.remote()
        try:
            ray.get(
                store.insert.remote(
                    "s",
                    "hi ",
                    TokenizedSegment(text="hi ", token_ids=[1, 2], labels=[-100, -100], logprobs=[0.0, 0.0]),
                )
            )
            ray.get(
                store.insert.remote(
                    "s",
                    "hi a1",
                    TokenizedSegment(text="a1", token_ids=[3], labels=[3], logprobs=[0.1], tool_call_count=1),
                )
            )
            ray.get(
                store.insert.remote(
                    "s",
                    "hi a1 tool ",
                    TokenizedSegment(text=" tool ", token_ids=[4, 5], labels=[-100, -100], logprobs=[0.0, 0.0]),
                )
            )
            ray.get(
                store.insert.remote(
                    "s",
                    "hi a1 tool a2",
                    TokenizedSegment(text="a2", token_ids=[6], labels=[6], logprobs=[0.2], tool_call_count=0),
                )
            )

            trace = ray.get(store.export_partial_training_trace.remote("s"))

            self.assertEqual(trace["input_ids"], [1, 2, 3, 4, 5, 6])
            self.assertEqual(trace["labels"], [-100, -100, 3, -100, -100, 6])
            # prompt 段 [1, 2] 被排除;tool-result 段 [4, 5] 保留在连续后缀中。
            self.assertEqual(trace["prompt_ids"], [1, 2])
            self.assertEqual(trace["response_ids"], [3, 4, 5, 6])
            self.assertEqual(trace["input_ids"], trace["prompt_ids"] + trace["response_ids"])
            self.assertEqual(trace["tool_call_count"], 1)
        finally:
            ray.kill(store)
