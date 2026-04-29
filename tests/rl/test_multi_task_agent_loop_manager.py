import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from xtuner.v1.data_proto.rl_data import Status
from xtuner.v1.rl.agent_loop_manager.agent_loop_manager import (
    AgentLoopManager,
    AgentLoopManagerConfig,
    AgentLoopManagerStatus,
    TaskSpecConfig,
    _TaskRunner,
)
from xtuner.v1.rl.agent_loop_manager.producer import GROUP_GENERATE_TIME_KEY, ProduceBatchStatus
from xtuner.v1.rl.utils import calculate_seq_staleness


class _FakeSampler:
    def __init__(self, size: int = 1):
        self._size = size
        self.saved_paths: list[Path] = []
        self.resumed_paths: list[Path] = []

    def __len__(self) -> int:
        return self._size

    def save(self, checkpoint_path):
        self.saved_paths.append(Path(checkpoint_path))
        return None

    def resume(self, checkpoint_path):
        self.resumed_paths.append(Path(checkpoint_path))
        return None


class _FakeProduceStrategy:
    def __init__(
        self,
        status: ProduceBatchStatus = ProduceBatchStatus.NORMAL,
        cleanup_pause_time_s: float = 0.0,
        stale_threshold: int = 1,
    ):
        self.status = status
        self.cleanup_pause_time_s = cleanup_pause_time_s
        self.stale_threshold = stale_threshold
        self.called_batch_sizes: list[int] = []
        self.called_train_steps: list[int] = []
        self.called_model_steps: list[int] = []
        self.called_update_events: list[object | None] = []
        self.called_update_event_states: list[bool | None] = []
        self.called_progresses: list[object] = []
        self.cleanup_model_steps: list[int] = []
        self.cleanup_progresses: list[object | None] = []
        self.cleanup_call_count = 0

    async def produce_batch(
        self,
        agent_loop,
        sampler,
        replay_buffer,
        batch_size: int,
        task_name: str,
        train_step: int = 0,
        update_event=None,
        *,
        model_step: int,
        progress,
    ) -> ProduceBatchStatus:
        self.called_batch_sizes.append(batch_size)
        self.called_train_steps.append(train_step)
        self.called_model_steps.append(model_step)
        self.called_update_events.append(update_event)
        self.called_update_event_states.append(None if update_event is None else update_event.is_set())
        self.called_progresses.append(progress)
        return self.status

    async def pause_produce(self, agent_loop, replay_buffer, task_name: str, *, progress) -> float:
        self.cleanup_call_count += 1
        self.cleanup_progresses.append(progress)
        return self.cleanup_pause_time_s


class _FakeStatusProduceStrategy:
    def __init__(self, status: ProduceBatchStatus, pause_time_s: float):
        self.status = status
        self.pause_time_s = pause_time_s
        self.cleanup_call_count = 0
        self.called_train_steps: list[int] = []
        self.called_model_steps: list[int] = []
        self.called_update_events: list[object | None] = []
        self.called_update_event_states: list[bool | None] = []
        self.called_progresses: list[object] = []
        self.cleanup_model_steps: list[int] = []
        self.cleanup_progresses: list[object | None] = []

    async def produce_batch(
        self,
        agent_loop,
        sampler,
        replay_buffer,
        batch_size: int,
        task_name: str,
        train_step: int = 0,
        update_event=None,
        *,
        model_step: int,
        progress,
    ) -> ProduceBatchStatus:
        self.called_train_steps.append(train_step)
        self.called_model_steps.append(model_step)
        self.called_update_events.append(update_event)
        self.called_update_event_states.append(None if update_event is None else update_event.is_set())
        self.called_progresses.append(progress)
        return self.status

    async def pause_produce(self, agent_loop, replay_buffer, task_name: str, *, progress) -> float:
        self.cleanup_call_count += 1
        self.cleanup_progresses.append(progress)
        return self.pause_time_s


class _FakeRolloutState:
    def __init__(self, uid: str, group_generate_time_s: float):
        self.uid = uid
        self.extra_fields = {GROUP_GENERATE_TIME_KEY: group_generate_time_s}


class _FakeStalenessRolloutState(_FakeRolloutState):
    def __init__(self, uid: str, group_generate_time_s: float, response_model_steps: list[int], seq_staleness: int = 0):
        super().__init__(uid, group_generate_time_s)
        self.response_model_steps = response_model_steps
        self.seq_staleness = seq_staleness


class _SequencedProduceStrategy(_FakeProduceStrategy):
    def __init__(self, statuses: list[ProduceBatchStatus], cleanup_pause_time_s: float = 0.0):
        super().__init__(status=ProduceBatchStatus.NORMAL, cleanup_pause_time_s=cleanup_pause_time_s)
        self._statuses = list(statuses)

    async def produce_batch(
        self,
        agent_loop,
        sampler,
        replay_buffer,
        batch_size: int,
        task_name: str,
        train_step: int = 0,
        update_event=None,
        *,
        model_step: int,
        progress,
    ) -> ProduceBatchStatus:
        self.called_batch_sizes.append(batch_size)
        self.called_train_steps.append(train_step)
        self.called_model_steps.append(model_step)
        self.called_update_events.append(update_event)
        self.called_update_event_states.append(None if update_event is None else update_event.is_set())
        self.called_progresses.append(progress)
        return self._statuses.pop(0) if self._statuses else ProduceBatchStatus.NORMAL


class _FakeReplayBuffer:
    def __init__(self, rollout_states_by_task: dict[str, list[list[str]]], leftover_counts: dict[tuple[str, Status], int]):
        self._rollout_states_by_task = rollout_states_by_task
        self._leftover_counts = leftover_counts
        self.saved_paths: list[Path] = []
        self.resumed_paths: list[Path] = []
        self.refresh_staleness_calls: list[tuple[str, int, int, tuple[Status, ...]]] = []

    async def get(self, batch_size: int, task_name: str, group_status: Status):
        assert group_status == Status.COMPLETED
        groups = self._rollout_states_by_task.get(task_name, [])
        selected = groups[:batch_size]
        self._rollout_states_by_task[task_name] = groups[batch_size:]
        return selected

    async def count(self, task_name: str, group_status: Status):
        return self._leftover_counts.get((task_name, group_status), 0)

    async def refresh_staleness(
        self,
        task_name: str,
        current_train_step: int,
        stale_threshold: int,
        statuses: list[Status] | None = None,
    ):
        self.refresh_staleness_calls.append((task_name, current_train_step, stale_threshold, tuple(statuses or ())))
        for group in self._rollout_states_by_task.get(task_name, []):
            for state in group:
                response_model_steps = getattr(state, "response_model_steps", None) or []
                if response_model_steps and hasattr(state, "seq_staleness"):
                    state.seq_staleness = calculate_seq_staleness(
                        min(response_model_steps), current_train_step
                    )
        return 0

    async def save(self, checkpoint_path: Path | str):
        self.saved_paths.append(Path(checkpoint_path))

    async def resume(self, checkpoint_path: Path | str):
        self.resumed_paths.append(Path(checkpoint_path))


class _SequencedCompletedReplayBuffer(_FakeReplayBuffer):
    def __init__(self, completed_counts: list[int], rollout_states_by_task: dict[str, list[list[str]]]):
        super().__init__(rollout_states_by_task=rollout_states_by_task, leftover_counts={})
        self._completed_counts = list(completed_counts)
        self.get_calls: list[tuple[int, str, Status]] = []
        self.completed_count_call_count = 0

    async def get(self, batch_size: int, task_name: str, group_status: Status):
        self.get_calls.append((batch_size, task_name, group_status))
        return await super().get(batch_size, task_name, group_status)

    async def count(self, task_name: str, group_status: Status):
        if group_status == Status.COMPLETED:
            self.completed_count_call_count += 1
            if self._completed_counts:
                return self._completed_counts.pop(0)
        return await super().count(task_name, group_status)


def _fake_agent_loop():
    rollout_ctl = MagicMock()
    rollout_ctl.continue_generation.remote = AsyncMock()
    rollout_ctl.pause_generation.remote = AsyncMock()
    rollout_ctl.get_rollout_metadata.remote = AsyncMock(return_value={"server_url_dict": {}})
    agent_loop = MagicMock()
    agent_loop.rollout_ctl = rollout_ctl
    return agent_loop


class TestMultiTaskAgentLoopManager(unittest.IsolatedAsyncioTestCase):
    async def _wait_until(self, predicate, timeout_s: float = 1.0):
        deadline = asyncio.get_running_loop().time() + timeout_s
        while asyncio.get_running_loop().time() < deadline:
            if predicate():
                return
            await asyncio.sleep(0.01)
        self.fail("Timed out waiting for condition.")

    def test_manager_config_accepts_single_task_spec(self):
        task = TaskSpecConfig.model_construct(
            task_name="single_task",
            agent_loop_config=MagicMock(),
            produce_strategy_config=MagicMock(),
            sampler_config=MagicMock(),
            weight=1.0,
        )

        manager_config = AgentLoopManagerConfig(tasks=task)

        self.assertEqual(manager_config.tasks.task_name, "single_task")

    async def test_produce_batch_allocates_by_weight_and_returns_task_sorted_results(self):
        strategy_a = _FakeProduceStrategy()
        strategy_b = _FakeProduceStrategy()
        strategy_c = _FakeProduceStrategy()
        replay_buffer = _FakeReplayBuffer(
            rollout_states_by_task={
                "task_a": [["a-0"], ["a-1"]],
                "task_b": [["b-0"], ["b-1"], ["b-2"]],
                "task_c": [],
            },
            leftover_counts={
                ("task_a", Status.COMPLETED): 1,
                ("task_b", Status.ABORTED): 2,
            },
        )

        multi_task_manager = AgentLoopManager(
            task_runners=[
                _TaskRunner(
                    task_name="task_b",
                    agent_loop=_fake_agent_loop(),
                    produce_strategy=strategy_b,
                    sampler=_FakeSampler(),
                    weight=1.0,
                    order=0,
                ),
                _TaskRunner(
                    task_name="task_a",
                    agent_loop=_fake_agent_loop(),
                    produce_strategy=strategy_a,
                    sampler=_FakeSampler(),
                    weight=2.0,
                    order=1,
                ),
                _TaskRunner(
                    task_name="task_c",
                    agent_loop=_fake_agent_loop(),
                    produce_strategy=strategy_c,
                    sampler=_FakeSampler(),
                    weight=0.0,
                    order=2,
                ),
            ],
            replay_buffer=replay_buffer,
        )
        multi_task_manager._status = AgentLoopManagerStatus.UPDATE_ABORT
        multi_task_manager._update_event.set()

        result = await multi_task_manager.produce_batch(batch_size=7, train_step=3, model_step=2)

        self.assertEqual(result.task_batch_sizes, {"task_a": 5, "task_b": 2, "task_c": 0})
        # sync produce_batch 在本轮入口恢复 NORMAL，收尾 pause 后保留 UPDATE_ABORT 到下一轮入口再清理。
        self.assertEqual(multi_task_manager._status, AgentLoopManagerStatus.UPDATE_ABORT)
        self.assertTrue(multi_task_manager._update_event.is_set())
        self.assertEqual(multi_task_manager._model_step, 2)
        self.assertEqual(strategy_a.called_batch_sizes, [5])
        self.assertEqual(strategy_b.called_batch_sizes, [2])
        self.assertEqual(strategy_c.called_batch_sizes, [])
        self.assertEqual(strategy_a.called_train_steps, [3])
        self.assertEqual(strategy_b.called_train_steps, [3])
        self.assertEqual(strategy_a.called_model_steps, [2])
        self.assertEqual(strategy_b.called_model_steps, [2])
        self.assertEqual(len(strategy_a.called_update_events), 1)
        self.assertFalse(strategy_a.called_update_event_states[0])
        self.assertEqual(len(strategy_b.called_update_events), 1)
        self.assertFalse(strategy_b.called_update_event_states[0])
        self.assertEqual(result.rollout_states, [["a-0"], ["a-1"], ["b-0"], ["b-1"]])
        self.assertEqual(result.leftover_completed, 1)
        self.assertEqual(result.leftover_aborted, 2)
        self.assertEqual(result.leftover_expired, 0)
        self.assertIn("task_a", result.task_results)
        self.assertIn("task_b", result.task_results)
        self.assertIn("task_c", result.task_results)

    def test_save_and_resume_roundtrip_restores_paused_manager_state(self):
        sampler = _FakeSampler()
        replay_buffer = _FakeReplayBuffer({}, {})
        manager = AgentLoopManager(
            task_runners=[
                _TaskRunner(
                    task_name="task_a",
                    agent_loop=_fake_agent_loop(),
                    produce_strategy=_FakeProduceStrategy(stale_threshold=5),
                    sampler=sampler,
                    weight=1.0,
                    order=0,
                )
            ],
            replay_buffer=replay_buffer,
        )
        manager._status = AgentLoopManagerStatus.EXPIRED_BATCH
        manager._model_step = 2
        manager._pause_time_s = 1.5

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir)
            manager.save(checkpoint_path, model_step=7)

            state_path = checkpoint_path / manager._MANAGER_STATE_PATH
            with state_path.open("r") as f:
                state = json.load(f)

            self.assertEqual(state["status"], "EXPIRED_BATCH")
            self.assertEqual(state["model_step"], 7)
            self.assertNotIn("model_step_override", state)
            self.assertEqual(state["next_consumer_step"], 1)
            self.assertEqual(state["producer_future_step"], 1)
            self.assertEqual(state["consumed_samples"], {"task_a": 0})
            self.assertEqual(state["target_samples"], {"task_a": 0})
            self.assertEqual(state["target_upto_future_step"], 0)

            restored_step = manager.resume(checkpoint_path)

        self.assertEqual(restored_step, 7)
        self.assertEqual(manager._status, AgentLoopManagerStatus.UPDATE_ABORT)
        self.assertTrue(manager._update_event.is_set())
        self.assertFalse(manager._finish_event.is_set())
        self.assertEqual(manager._pause_time_s, 0.0)
        self.assertEqual(manager._model_step, 7)
        self.assertEqual(sampler.saved_paths, [Path(tmp_dir) / manager._TASK_CHECKPOINT_DIR / "task_a"])
        self.assertEqual(sampler.resumed_paths, [Path(tmp_dir) / manager._TASK_CHECKPOINT_DIR / "task_a"])
        self.assertEqual(replay_buffer.saved_paths, [Path(tmp_dir)])
        self.assertEqual(replay_buffer.resumed_paths, [Path(tmp_dir)])

    def test_save_rejects_pending_async_tasks(self):
        strategy = _FakeProduceStrategy()
        strategy._pending_tasks = {object()}
        manager = AgentLoopManager(
            task_runners=[
                _TaskRunner(
                    task_name="task_a",
                    agent_loop=_fake_agent_loop(),
                    produce_strategy=strategy,
                    sampler=_FakeSampler(),
                    weight=1.0,
                    order=0,
                )
            ],
            replay_buffer=_FakeReplayBuffer({}, {}),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaisesRegex(RuntimeError, "pending rollout tasks"):
                manager.save(tmp_dir, model_step=0)

    async def test_custom_get_task_batch_sizes_can_disable_tasks(self):
        strategy_a = _FakeProduceStrategy()
        strategy_b = _FakeProduceStrategy()
        replay_buffer = _FakeReplayBuffer(
            rollout_states_by_task={
                "task_a": [["a-0"]],
                "task_b": [["b-0"], ["b-1"]],
            },
            leftover_counts={},
        )

        class _CustomBatchManager(AgentLoopManager):
            def get_task_batch_sizes(self, global_batch_size: int, train_step: int) -> dict[str, int]:
                self.observed_train_step = train_step
                return {"task_a": 0, "task_b": global_batch_size}

        multi_task_manager = _CustomBatchManager(
            task_runners=[
                _TaskRunner(
                    task_name="task_a",
                    agent_loop=_fake_agent_loop(),
                    produce_strategy=strategy_a,
                    sampler=_FakeSampler(),
                    weight=1.0,
                    order=0,
                ),
                _TaskRunner(
                    task_name="task_b",
                    agent_loop=_fake_agent_loop(),
                    produce_strategy=strategy_b,
                    sampler=_FakeSampler(),
                    weight=1.0,
                    order=1,
                ),
            ],
            replay_buffer=replay_buffer,
        )

        result = await multi_task_manager.produce_batch(batch_size=2, train_step=9, model_step=8)

        self.assertEqual(multi_task_manager.observed_train_step, 9)
        self.assertEqual(result.task_batch_sizes, {"task_a": 0, "task_b": 2})
        self.assertEqual(strategy_a.called_batch_sizes, [])
        self.assertEqual(strategy_b.called_batch_sizes, [2])
        self.assertEqual(result.rollout_states, [["b-0"], ["b-1"]])

    async def test_status_returning_strategy_uses_cleanup_and_reconstructs_group_timing_stats(self):
        strategy = _FakeStatusProduceStrategy(status=ProduceBatchStatus.NORMAL, pause_time_s=1.25)
        replay_buffer = _FakeReplayBuffer(
            rollout_states_by_task={
                "task_a": [
                    [_FakeRolloutState("a-0", 0.5)],
                    [_FakeRolloutState("a-1", 1.0)],
                ],
            },
            leftover_counts={},
        )

        manager = AgentLoopManager(
            task_runners=[
                _TaskRunner(
                    task_name="task_a",
                    agent_loop=_fake_agent_loop(),
                    produce_strategy=strategy,
                    sampler=_FakeSampler(),
                    weight=1.0,
                    order=0,
                ),
            ],
            replay_buffer=replay_buffer,
        )

        result = await manager.produce_batch(batch_size=2, train_step=7, model_step=6)

        self.assertEqual(strategy.cleanup_call_count, 1)
        self.assertEqual(len(strategy.cleanup_progresses), 1)
        self.assertIsNotNone(strategy.cleanup_progresses[0])
        self.assertEqual(strategy.cleanup_progresses[0].consumed_samples["task_a"], 2)
        self.assertEqual(manager._model_step, 6)
        self.assertEqual(strategy.called_train_steps, [7])
        self.assertEqual(strategy.called_model_steps, [6])
        self.assertEqual(len(strategy.called_update_events), 1)
        self.assertFalse(strategy.called_update_event_states[0])
        self.assertEqual(manager._status, AgentLoopManagerStatus.UPDATE_ABORT)
        self.assertTrue(manager._update_event.is_set())
        self.assertEqual(result.group_gen_count, 2)
        self.assertAlmostEqual(result.group_gen_mean_s, 0.75)
        self.assertAlmostEqual(result.group_gen_p50_s, 1.0)
        self.assertAlmostEqual(result.group_gen_p99_s, 1.0)
        self.assertAlmostEqual(result.group_gen_pause_time_s, 1.25)

    async def test_produce_batch_requires_non_empty_rollout_states(self):
        manager = AgentLoopManager(
            task_runners=[
                _TaskRunner(
                    task_name="task_a",
                    agent_loop=_fake_agent_loop(),
                    produce_strategy=_FakeStatusProduceStrategy(status=ProduceBatchStatus.NORMAL, pause_time_s=0.0),
                    sampler=_FakeSampler(),
                    weight=1.0,
                    order=0,
                ),
            ],
            replay_buffer=_FakeReplayBuffer(rollout_states_by_task={}, leftover_counts={}),
        )

        with self.assertRaisesRegex(AssertionError, "must return non-empty rollout_states"):
            await manager.produce_batch(batch_size=1, train_step=3, model_step=2)

    async def test_pause_produce_from_async_produce_loop_sets_status_and_pause_time(self):
        strategy = _FakeProduceStrategy(cleanup_pause_time_s=2.5)
        manager = AgentLoopManager(
            task_runners=[
                _TaskRunner(
                    task_name="task_a",
                    agent_loop=_fake_agent_loop(),
                    produce_strategy=strategy,
                    sampler=_FakeSampler(),
                    weight=1.0,
                    order=0,
                ),
            ],
            replay_buffer=_FakeReplayBuffer({}, {}),
        )

        pause_time_s = await manager.pause_produce(use_global_progress=True)

        self.assertEqual(pause_time_s, 2.5)
        self.assertEqual(strategy.cleanup_call_count, 1)
        self.assertEqual(len(strategy.cleanup_progresses), 1)
        self.assertIs(strategy.cleanup_progresses[0], manager._produce_progress)
        self.assertTrue(manager._update_event.is_set())
        self.assertEqual(manager._status, AgentLoopManagerStatus.UPDATE_ABORT)
        self.assertEqual(manager._pause_time_s, 2.5)

    async def test_pause_produce_validates_progress_selection_before_state_change(self):
        strategy = _FakeProduceStrategy(cleanup_pause_time_s=2.5)
        manager = AgentLoopManager(
            task_runners=[
                _TaskRunner(
                    task_name="task_a",
                    agent_loop=_fake_agent_loop(),
                    produce_strategy=strategy,
                    sampler=_FakeSampler(),
                    weight=1.0,
                    order=0,
                ),
            ],
            replay_buffer=_FakeReplayBuffer({}, {}),
        )

        with self.assertRaisesRegex(ValueError, "progress must not be provided"):
            await manager.pause_produce(use_global_progress=True, progress=object())
        self.assertFalse(manager._update_event.is_set())
        self.assertEqual(manager._status, AgentLoopManagerStatus.NORMAL)

        with self.assertRaisesRegex(ValueError, "progress must be provided"):
            await manager.pause_produce(use_global_progress=False)
        self.assertFalse(manager._update_event.is_set())
        self.assertEqual(manager._status, AgentLoopManagerStatus.NORMAL)
        self.assertEqual(strategy.cleanup_call_count, 0)

    async def test_get_batch_returns_expired_batch_when_manager_is_expired(self):
        manager = AgentLoopManager(
            task_runners=[
                _TaskRunner(
                    task_name="task_a",
                    agent_loop=_fake_agent_loop(),
                    produce_strategy=_FakeProduceStrategy(),
                    sampler=_FakeSampler(),
                    weight=1.0,
                    order=0,
                ),
            ],
            replay_buffer=_FakeReplayBuffer({}, {}),
        )
        manager._status = AgentLoopManagerStatus.EXPIRED_BATCH

        result = await manager.get_batch(batch_size=2, train_step=11)

        self.assertEqual(result.status, ProduceBatchStatus.EXPIRED_BATCH)
        self.assertEqual(result.rollout_states, [])

    async def test_get_batch_refreshes_staleness_at_entry(self):
        replay_buffer = _FakeReplayBuffer(
            rollout_states_by_task={
                "task_a": [[_FakeStalenessRolloutState("a-0", 0.2, response_model_steps=[4], seq_staleness=0)]],
            },
            leftover_counts={("task_a", Status.COMPLETED): 1},
        )
        manager = AgentLoopManager(
            task_runners=[
                _TaskRunner(
                    task_name="task_a",
                    agent_loop=_fake_agent_loop(),
                    produce_strategy=_FakeProduceStrategy(stale_threshold=5),
                    sampler=_FakeSampler(),
                    weight=1.0,
                    order=0,
                ),
            ],
            replay_buffer=replay_buffer,
        )

        result = await manager.get_batch(batch_size=1, train_step=9)

        self.assertEqual(
            replay_buffer.refresh_staleness_calls,
            [
                ("task_a", 9, 5, (Status.COMPLETED, Status.ABORTED)),
                ("task_a", 10, 5, (Status.COMPLETED, Status.ABORTED)),
            ],
        )
        self.assertEqual(result.rollout_states[0][0].seq_staleness, 4)
        self.assertEqual(manager._produce_progress.next_consumer_step, 10)
        self.assertEqual(manager._produce_progress.consumed_samples["task_a"], 1)

    async def test_get_batch_waits_until_requested_batch_size_is_ready(self):
        replay_buffer = _SequencedCompletedReplayBuffer(
            completed_counts=[0, 1, 2],
            rollout_states_by_task={
                "task_a": [
                    [_FakeStalenessRolloutState("a-0", 0.2, response_model_steps=[4], seq_staleness=0)],
                    [_FakeStalenessRolloutState("a-1", 0.3, response_model_steps=[4], seq_staleness=0)],
                ],
            },
        )
        manager = AgentLoopManager(
            task_runners=[
                _TaskRunner(
                    task_name="task_a",
                    agent_loop=_fake_agent_loop(),
                    produce_strategy=_FakeProduceStrategy(),
                    sampler=_FakeSampler(),
                    weight=1.0,
                    order=0,
                ),
            ],
            replay_buffer=replay_buffer,
        )
        manager._STATUS_POLL_INTERVAL_S = 0.01

        result = await asyncio.wait_for(manager.get_batch(batch_size=2, train_step=9), timeout=1.0)

        self.assertEqual([group[0].uid for group in result.rollout_states], ["a-0", "a-1"])
        self.assertEqual(replay_buffer.get_calls, [(2, "task_a", Status.COMPLETED)])
        self.assertGreaterEqual(replay_buffer.completed_count_call_count, 3)
        self.assertEqual(manager._produce_progress.consumed_samples["task_a"], 2)
        self.assertEqual(manager._produce_progress.next_consumer_step, 10)

    async def test_produce_batch_to_buffer_aggregates_status_with_update_abort_priority(self):
        manager = AgentLoopManager(
            task_runners=[
                _TaskRunner(
                    task_name="task_a",
                    agent_loop=_fake_agent_loop(),
                    produce_strategy=_FakeProduceStrategy(status=ProduceBatchStatus.NORMAL),
                    sampler=_FakeSampler(),
                    weight=1.0,
                    order=0,
                ),
                _TaskRunner(
                    task_name="task_b",
                    agent_loop=_fake_agent_loop(),
                    produce_strategy=_FakeProduceStrategy(status=ProduceBatchStatus.EXPIRED_BATCH),
                    sampler=_FakeSampler(),
                    weight=1.0,
                    order=1,
                ),
                _TaskRunner(
                    task_name="task_c",
                    agent_loop=_fake_agent_loop(),
                    produce_strategy=_FakeProduceStrategy(status=ProduceBatchStatus.UPDATE_ABORT),
                    sampler=_FakeSampler(),
                    weight=1.0,
                    order=2,
                ),
            ],
            replay_buffer=_FakeReplayBuffer({}, {}),
        )

        manager._model_step = 5
        manager._produce_progress.producer_future_step = 5
        status = await manager._produce_batch_to_buffer(batch_size=3, progress=manager._produce_progress)

        self.assertEqual(status, ProduceBatchStatus.UPDATE_ABORT)

    async def test_produce_loop_waits_for_continue_produce_and_stops_on_finish(self):
        strategy = _SequencedProduceStrategy(
            statuses=[ProduceBatchStatus.NORMAL, ProduceBatchStatus.EXPIRED_BATCH, ProduceBatchStatus.NORMAL],
        )
        manager = AgentLoopManager(
            task_runners=[
                _TaskRunner(
                    task_name="task_a",
                    agent_loop=_fake_agent_loop(),
                    produce_strategy=strategy,
                    sampler=_FakeSampler(),
                    weight=1.0,
                    order=0,
                ),
            ],
            replay_buffer=_FakeReplayBuffer({}, {}),
        )
        manager._STATUS_POLL_INTERVAL_S = 0.01

        manager._produce_progress.producer_future_step = 3
        loop_task = asyncio.create_task(manager.produce_loop(batch_size=1))
        await self._wait_until(lambda: manager._status == AgentLoopManagerStatus.EXPIRED_BATCH)
        self.assertEqual(manager._status, AgentLoopManagerStatus.EXPIRED_BATCH)
        self.assertEqual(strategy.called_train_steps[:2], [3, 4])

        manager.continue_produce(model_step=9)
        await self._wait_until(lambda: len(strategy.called_train_steps) >= 3)
        self.assertEqual(manager._status, AgentLoopManagerStatus.NORMAL)
        self.assertEqual(strategy.called_train_steps[:3], [3, 4, 4])
        self.assertEqual(strategy.called_model_steps[2], 9)

        manager._status = AgentLoopManagerStatus.FINISH
        manager._finish_event.set()
        await asyncio.wait_for(loop_task, timeout=1.0)
