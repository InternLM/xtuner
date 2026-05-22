import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from xtuner.v1.data_proto.rl_data import Status
from xtuner.v1.rl.agent_loop_manager import ProduceContext, ProduceProgress
from xtuner.v1.rl.agent_loop_manager.agent_loop_manager import AgentLoopManager, _TaskRunner
from xtuner.v1.rl.replay_buffer import AsyncReplayBufferConfig, RefreshStalenessResult
from xtuner.v1.rl.rollout.trace_store import RolloutTraceStore, TokenizedSegment, TraceState


class _TraceRolloutState:
    def __init__(
        self,
        uid: int | str,
        *,
        status: Status = Status.COMPLETED,
        reward_score: float | None = None,
        session_uid: int | str | None = None,
    ):
        self.uid = uid
        self.id = uid
        self.session_uid = session_uid
        self.status = status
        self.seq_staleness = 0
        self.response_ids = []
        self.extra_fields = {}
        self.reward = {"score": reward_score} if reward_score is not None else None


class _TraceRefreshReplayBuffer:
    def __init__(self, expired_session_ids_by_task: dict[str, list[int | str]]):
        self._expired_session_ids_by_task = expired_session_ids_by_task

    async def refresh_staleness(
        self,
        *,
        task_stale_thresholds: dict[str, int],
        current_train_step: int,
        statuses: list[Status] | None = None,
    ):
        return {
            task_name: RefreshStalenessResult(
                expired_count=len(self._expired_session_ids_by_task.get(task_name, [])),
                expired_session_ids=self._expired_session_ids_by_task.get(task_name, []),
            )
            for task_name in task_stale_thresholds
        }


class _TraceProduceStrategy:
    stale_threshold = 5
    enable_partial_rollout = False


class TestRolloutTraceStoreRolloutStatus(unittest.TestCase):
    def setUp(self):
        store_cls = RolloutTraceStore.__ray_metadata__.modified_class
        self.store = store_cls()

    def _insert_segment(self, session_id: str):
        self.store.insert(session_id, "prompt", TokenizedSegment(text="prompt", token_ids=[1]))

    def test_mark_rollout_statuses_marks_completed_and_releases_filtered(self):
        self._insert_segment("completed")
        self._insert_segment("filtered")

        results = self.store.mark_rollout_statuses(
            [
                ("completed", Status.COMPLETED),
                ("filtered", Status.FILTERED),
            ]
        )

        self.assertEqual(results["completed"], TraceState.ROLLOUT_FINISHED.value)
        self.assertEqual(results["filtered"], TraceState.RELEASED.value)
        self.assertEqual(
            self.store.get_state("completed")["state"],
            TraceState.ROLLOUT_FINISHED.value,
        )
        self.assertIsNone(self.store.get_state("filtered"))

    def test_mark_rollout_statuses_discards_expired_finished_session(self):
        self._insert_segment("expired")
        self.store.mark_rollout_status("expired", Status.COMPLETED)

        results = self.store.mark_rollout_statuses([("expired", Status.EXPIRED)])

        self.assertEqual(results["expired"], TraceState.RELEASED.value)
        self.assertIsNone(self.store.get_state("expired"))


class TestTraceStoreProducerReporting(unittest.IsolatedAsyncioTestCase):
    async def test_put_generated_group_reports_final_status_to_trace_store(self):
        task_name = "test_trace_status"
        progress = ProduceProgress.build([task_name])
        replay_buffer = AsyncReplayBufferConfig().build()
        ctx = ProduceContext(
            agent_loop=MagicMock(),
            sampler=MagicMock(),
            replay_buffer=replay_buffer,
            task_batch_size=1,
            task_name=task_name,
            train_step=0,
            update_event=asyncio.Event(),
            model_step=0,
            progress=progress,
            is_valid_sample_fn=lambda samples: False,
        )
        store = MagicMock()
        store.mark_rollout_statuses.remote = AsyncMock(return_value={})

        completed_group = [
            _TraceRolloutState(
                1,
                status=Status.COMPLETED,
                reward_score=1.0,
                session_uid="trace-session",
            )
        ]
        with patch("xtuner.v1.rl.agent_loop_manager.producer.get_store", return_value=store):
            self.assertFalse(await ctx.put_generated_group(completed_group))

        store.mark_rollout_statuses.remote.assert_awaited_once_with(
            [("trace-session", Status.FILTERED)],
            enable_partial_rollout=False,
        )


class TestTraceStoreManagerReporting(unittest.IsolatedAsyncioTestCase):
    async def test_refresh_for_all_tasks_reports_expired_sessions_to_trace_store(self):
        replay_buffer = _TraceRefreshReplayBuffer({"task_a": ["sid-a", "sid-b"]})
        manager = AgentLoopManager(
            task_runners=[
                _TaskRunner(
                    task_name="task_a",
                    agent_loop=MagicMock(),
                    produce_strategy=_TraceProduceStrategy(),
                    sampler=MagicMock(),
                    weight=1.0,
                    order=0,
                ),
            ],
            replay_buffer=replay_buffer,
        )
        store = MagicMock()
        store.mark_rollout_statuses.remote = AsyncMock(return_value={})

        with patch("xtuner.v1.rl.agent_loop_manager.agent_loop_manager.get_store", return_value=store):
            await manager._refresh_for_all_tasks(9, [Status.COMPLETED, Status.ABORTED])

        store.mark_rollout_statuses.remote.assert_awaited_once_with(
            [("sid-a", Status.EXPIRED), ("sid-b", Status.EXPIRED)],
            enable_partial_rollout=False,
        )


if __name__ == "__main__":
    unittest.main()
