import asyncio
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import ray

from xtuner.v1.rl.rollout import trace_store as trace_store_module
from xtuner.v1.rl.rollout.trace_store import (
    RolloutTraceStore,
    TreeNode,
    Trie,
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


class TestTraceTrie(unittest.TestCase):
    def test_keys_handles_a_trace_deeper_than_the_recursion_limit(self) -> None:
        trie, nodes = self._deep_trie()
        keys = trie.keys()
        self.assertEqual(keys, ["x" * (i + 1) for i in range(len(nodes))])

    def test_release_handles_a_trace_deeper_than_the_recursion_limit(self) -> None:
        trie, nodes = self._deep_trie()
        freed = []
        with patch.object(trace_store_module, "_free_ray_refs", side_effect=freed.append):
            trie.release()
            trie.release()
        self.assertEqual(freed, list(reversed(range(len(nodes)))))
        self.assertFalse(trie.root.children)
        self.assertTrue(all(node.value is None and not node.children and node.parent is None for node in nodes))
        trie.insert("new", "new-value")
        self.assertEqual(trie.keys(), ["new"])

    def test_branch_order_and_partial_release_are_preserved(self) -> None:
        trie = Trie()
        for key in ("", "a", "ab", "ac", "z"):
            trie.insert(key, key)
        self.assertEqual(trie.keys(), ["", "a", "ab", "ac", "z"])
        freed = []
        with patch.object(trace_store_module, "_free_ray_refs", side_effect=freed.append):
            trie.release("a")
            self.assertEqual(trie.keys(), ["", "ab", "ac", "z"])
            trie.release("missing")
            trie.release("ab")
            self.assertEqual(trie.keys(), ["", "ac", "z"])
            trie.release()
        self.assertEqual(freed, ["a", "ab", "ac", "z", ""])
        self.assertEqual(trie.keys(), [])

    def test_empty_tree_can_be_released_repeatedly(self) -> None:
        trie = Trie()
        with patch.object(trace_store_module, "_free_ray_refs") as free:
            trie.release()
            trie.release()
        free.assert_not_called()
        self.assertEqual(trie.keys(), [])

    @staticmethod
    def _deep_trie() -> tuple[Trie, list[TreeNode]]:
        trie = Trie()
        node = trie.root
        nodes = []
        # Incremental conversation prefixes form this chain. Construct it
        # directly to avoid quadratic insert time in the test fixture.
        for i in range(max(3000, sys.getrecursionlimit() + 100)):
            child = TreeNode(value=i, parent=node)
            node.children["x"] = child
            node = child
            nodes.append(node)
        return trie, nodes
