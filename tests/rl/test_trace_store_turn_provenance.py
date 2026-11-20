"""Cross-weight-update rollout trace export 与共享引用所有权测试。

Good Tests:
- 使用真实 RolloutTraceStore actor、真实 Trie 和真实 Ray ObjectRef 验证导出契约。
- 通过 export_training_trace 的返回值和释放后的 ray.get 可读性验证行为。

Bad Tests:
- 不直接比较 Trie 内部节点。
- 不断言单个集合 helper 的实现。
"""

import unittest

import ray

from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.rl.rollout import trace_store as trace_store_module
from xtuner.v1.rl.rollout.trace_store import RolloutTraceStore, TokenizedSegment, Trie


def _segment(
    tokens: list[int],
    *,
    labels: list[int] | None = None,
    request_id: str | None = None,
    model_step: int | None = None,
    expert_key=None,
    shared: bool = False,
) -> TokenizedSegment:
    return TokenizedSegment(
        text="",
        token_ids=tokens,
        labels=list(tokens) if labels is None else labels,
        logprobs=[0.0] * len(tokens),
        request_id=request_id,
        model_step=model_step,
        expert_key=expert_key,
        routed_experts_shared=shared,
    )


class TestTurnProvenanceExport(unittest.TestCase):
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

    def _build_session(self, store, session_id: str, *, expert=None) -> str:
        ray.get(
            store.insert.remote(
                session_id,
                "P",
                _segment([1, 2], labels=[-100, -100], request_id="r0", model_step=0, expert_key=expert, shared=True),
            )
        )
        ray.get(
            store.insert.remote(
                session_id, "PA", _segment([3, 4], request_id="r0", model_step=0, expert_key=expert, shared=True)
            )
        )
        ray.get(
            store.insert.remote(
                session_id, "PAT", _segment([5], request_id="r1", model_step=1, expert_key=expert, shared=True)
            )
        )
        ray.get(
            store.insert.remote(
                session_id, "PATT", _segment([6, 7], request_id="r1", model_step=1, expert_key=expert, shared=True)
            )
        )
        ray.get(
            store.insert.remote(
                session_id, "PB", _segment([8], request_id="rx", model_step=2, expert_key=expert, shared=True)
            )
        )
        return "PATT"

    def test_strict_export_returns_prompt_and_full_suffix_versions(self):
        store = RolloutTraceStore.remote()
        try:
            prompt_text = self._build_session(store, "s0", expert=ray.put([0]))
            trace = ray.get(
                store.export_training_trace.remote(
                    "s0", prompt_text, expected_request_ids=["r0", "r1"], require_routed_experts=True
                )
            )
        finally:
            ray.kill(store)

        self.assertEqual(trace["input_ids"], [1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(trace["labels"], [-100, -100, 3, 4, 5, 6, 7])
        self.assertEqual(trace["prompt_ids"], [1, 2])
        self.assertEqual(trace["response_ids"], [3, 4, 5, 6, 7])
        self.assertEqual(trace["response_model_steps"], [0, 0, 1, 1, 1])
        self.assertEqual(trace["request_ids"], ["r0", "r1"])
        self.assertTrue(trace["routed_experts_shared"])

    def test_transcript_ids_may_name_another_committed_branch(self):
        store = RolloutTraceStore.remote()
        try:
            prompt_text = self._build_session(store, "s0")
            trace = ray.get(
                store.export_training_trace.remote("s0", prompt_text, expected_request_ids=["r0", "rx", "r1"])
            )
        finally:
            ray.kill(store)
        self.assertEqual(trace["request_ids"], ["r0", "r1"])

    def test_uncommitted_or_mismatched_transcript_ids_are_rejected(self):
        store = RolloutTraceStore.remote()
        try:
            prompt_text = self._build_session(store, "s0")
            for expected_ids in (["r1"], ["r0"], ["r0", "r2"]):
                with self.subTest(expected_ids=expected_ids):
                    with self.assertRaisesRegex(ValueError, "do not match this trace path|uncommitted"):
                        ray.get(
                            store.export_training_trace.remote("s0", prompt_text, expected_request_ids=expected_ids)
                        )
            # The path is valid but the transcript also references an id that was never committed.
            with self.assertRaisesRegex(ValueError, "uncommitted requests"):
                ray.get(
                    store.export_training_trace.remote("s0", prompt_text, expected_request_ids=["r0", "missing", "r1"])
                )
        finally:
            ray.kill(store)

    def test_missing_token_versions_or_prompt_only_head_are_rejected(self):
        store = RolloutTraceStore.remote()
        try:
            ray.get(
                store.insert.remote("no-version", "P", _segment([1], labels=[-100], request_id="r0", model_step=0))
            )
            ray.get(store.insert.remote("no-version", "PA", _segment([2], request_id="r0")))
            with self.assertRaisesRegex(ValueError, "missing token versions"):
                ray.get(store.export_training_trace.remote("no-version", "PA", expected_request_ids=["r0"]))

            ray.get(store.insert.remote("bad-head", "P", _segment([1], labels=[1], request_id="r0", model_step=0)))
            ray.get(store.insert.remote("bad-head", "PA", _segment([2], request_id="r0", model_step=0)))
            with self.assertRaisesRegex(ValueError, "prompt-only|not a prompt segment"):
                ray.get(store.export_training_trace.remote("bad-head", "PA", expected_request_ids=["r0"]))
        finally:
            ray.kill(store)

    def test_another_request_cannot_overwrite_a_committed_node(self):
        store = RolloutTraceStore.remote()
        try:
            ray.get(store.insert.remote("s0", "P", _segment([1], labels=[-100], request_id="r0")))
            with self.assertRaisesRegex(ValueError, "Conflicting trace write"):
                ray.get(store.insert.remote("s0", "P", _segment([9], labels=[9], request_id="r1")))
        finally:
            ray.kill(store)

    def test_legacy_export_keeps_its_contract(self):
        store = RolloutTraceStore.remote()
        try:
            ray.get(store.insert.remote("s0", "P", _segment([1, 2], labels=[-100, -100])))
            trace = ray.get(store.export_training_trace.remote("s0", "P"))
        finally:
            ray.kill(store)
        self.assertEqual(trace["input_ids"], [1, 2])
        self.assertEqual(trace["prompt_ids"], [1, 2])
        self.assertEqual(trace["response_ids"], [])
        self.assertEqual(trace["tool_call_count"], 0)
        self.assertFalse(trace["routed_experts_shared"])


class TestSharedRoutedExpertOwnership(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.started_ray = False
        try:
            if not ray.is_initialized():
                ray.init(address="local", num_cpus=1, include_dashboard=False, ignore_reinit_error=True)
                cls.started_ray = True
        except Exception as exc:
            raise unittest.SkipTest(f"Ray init failed for shared-ref tests: {exc}") from exc

    @classmethod
    def tearDownClass(cls):
        if cls.started_ray and ray.is_initialized():
            ray.shutdown()

    def test_releasing_a_shared_trace_keeps_expert_refs_readable(self):
        ref = ray.put({"experts": [1, 2]})
        trie = Trie()
        trie.insert("P", _segment([1], labels=[-100], request_id="r0", expert_key=ref, shared=True))

        trie.release()

        self.assertEqual(ray.get(ref), {"experts": [1, 2]})

    def test_legacy_trace_release_keeps_the_explicit_free_path(self):
        segment = _segment([1], labels=[-100], expert_key="legacy-ref")
        trie = Trie()
        trie.insert("P", segment)
        freed = []

        original = trace_store_module._free_ray_refs
        trace_store_module._free_ray_refs = freed.append
        try:
            trie.release()
        finally:
            trace_store_module._free_ray_refs = original

        self.assertEqual(freed, [segment])

    def test_shared_rollout_state_cleanup_detaches_without_freeing(self):
        from xtuner.v1.data_proto.rl_data import free_rollout_state_refs
        from xtuner.v1.rl.utils import ray_utils

        expert_ref = ray.put("expert")
        pixel_ref = ray.put("pixel")
        state = RolloutState(
            message=[],
            routed_experts=[expert_ref],
            routed_experts_shared=True,
            mm_info={"pixel_values": pixel_ref},
        )
        freed = []
        original = ray_utils.free_object_refs
        ray_utils.free_object_refs = freed.append
        try:
            free_rollout_state_refs(state)
        finally:
            ray_utils.free_object_refs = original

        self.assertIsNone(state.routed_experts)
        self.assertFalse(state.routed_experts_shared)
        self.assertEqual(freed, [[pixel_ref]])
        self.assertEqual(ray.get(expert_ref), "expert")

    def test_reset_response_frees_only_owned_experts(self):
        from xtuner.v1.rl.utils import ray_utils

        expert_ref = ray.put("expert")
        state = RolloutState(
            message=[],
            routed_experts=expert_ref,
            routed_experts_shared=True,
            response_ids=[1],
            status=Status.COMPLETED,
        )
        freed = []
        original = ray_utils.free_object_refs
        ray_utils.free_object_refs = freed.append
        try:
            state.reset_response()
        finally:
            ray_utils.free_object_refs = original

        self.assertEqual(freed, [])
        self.assertIsNone(state.routed_experts)
        self.assertFalse(state.routed_experts_shared)


if __name__ == "__main__":
    unittest.main()
