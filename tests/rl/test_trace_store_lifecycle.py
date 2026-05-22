import os
import unittest

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

from xtuner.v1.data_proto.rl_data import Status
from xtuner.v1.rl.rollout.trace_store import RolloutTraceStore, TokenizedSegment, TraceState


class RolloutTraceStoreLifecycleTest(unittest.TestCase):
    def setUp(self):
        store_cls = RolloutTraceStore.__ray_metadata__.modified_class
        self.store = store_cls()

    def _insert_segment(self, session_id: str, key: str = "prompt"):
        segment = TokenizedSegment(text=key, token_ids=[1], labels=[1], logprobs=[0.0])
        self.store.insert(session_id, key, segment)

    def _state(self, session_id: str):
        state = self.store.get_state(session_id)
        return None if state is None else state["state"]

    def _move_to_train_running(self, session_id: str, key: str = "prompt"):
        self._insert_segment(session_id, key)
        self.store.mark_rollout_status(session_id, Status.COMPLETED)
        self.store.export_training_trace(session_id, key)

    def test_completed_marks_rollout_finished(self):
        self._insert_segment("completed")

        state = self.store.mark_rollout_status("completed", Status.COMPLETED)

        self.assertEqual(state, TraceState.ROLLOUT_FINISHED.value)
        self.assertEqual(self._state("completed"), TraceState.ROLLOUT_FINISHED.value)

    def test_release_like_rollout_status_releases_session(self):
        cases = (
            (Status.FILTERED, {}),
            (Status.FAILED, {}),
            (Status.EXPIRED, {}),
            (Status.ABORTED, {"enable_partial_rollout": False}),
        )
        for status, kwargs in cases:
            session_id = f"release-{status.value}"
            self._insert_segment(session_id)

            state = self.store.mark_rollout_status(session_id, status, **kwargs)

            self.assertEqual(state, TraceState.RELEASED.value)
            self.assertIsNone(self.store.get_state(session_id))

    def test_aborted_with_partial_rollout_keeps_session_running(self):
        self._insert_segment("partial-abort")

        state = self.store.mark_rollout_status(
            "partial-abort",
            Status.ABORTED,
            enable_partial_rollout=True,
        )

        self.assertEqual(state, TraceState.ROLLOUT_RUNNING.value)
        self.assertEqual(self._state("partial-abort"), TraceState.ROLLOUT_RUNNING.value)
        matched_key, nodes = self.store.search("partial-abort", "prompt", True)
        self.assertEqual(matched_key, "prompt")
        self.assertEqual(len(nodes), 1)

    def test_mark_commit_failed_and_rollout_discarded_release_sessions(self):
        self._insert_segment("commit-failed")
        self._insert_segment("discarded")
        self.store.mark_rollout_status("discarded", Status.COMPLETED)

        commit_failed_state = self.store.mark_commit_failed("commit-failed")
        discarded_state = self.store.mark_rollout_discarded("discarded")

        self.assertEqual(commit_failed_state, TraceState.RELEASED.value)
        self.assertIsNone(self.store.get_state("commit-failed"))
        self.assertEqual(discarded_state, TraceState.RELEASED.value)
        self.assertIsNone(self.store.get_state("discarded"))

    def test_mark_train_finished_and_abandoned_release_train_running_sessions(self):
        self._move_to_train_running("train-finished")
        self._move_to_train_running("train-abandoned")

        finished_state = self.store.mark_train_finished("train-finished")
        abandoned_state = self.store.mark_train_abandoned("train-abandoned")

        self.assertEqual(finished_state, TraceState.RELEASED.value)
        self.assertIsNone(self.store.get_state("train-finished"))
        self.assertEqual(abandoned_state, TraceState.RELEASED.value)
        self.assertIsNone(self.store.get_state("train-abandoned"))

    def test_train_lifecycle_missing_and_invalid_state_behavior(self):
        self.assertEqual(self.store.mark_train_finished("missing-finished"), TraceState.RELEASED.value)
        self.assertEqual(self.store.mark_train_abandoned("missing-abandoned"), TraceState.RELEASED.value)

        self._insert_segment("rollout-running")
        with self.assertRaisesRegex(RuntimeError, "mark_train_finished"):
            self.store.mark_train_finished("rollout-running")
        with self.assertRaisesRegex(RuntimeError, "mark_train_abandoned"):
            self.store.mark_train_abandoned("rollout-running")


if __name__ == "__main__":
    unittest.main()
