import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from xtuner.v1.rl.agent_loop.agent_loop_manager import (
    AgentLoopManager,
    AgentLoopManagerStatus,
    AgentLoopManagerConfig,
    TaskSpecConfig,
    _TaskRunner,
)
from xtuner.v1.rl.agent_loop.producer import GROUP_GENERATE_TIME_KEY, ProduceBatchStatus
from xtuner.v1.data_proto import Status


class _FakeSampler:
    def __init__(self, size: int = 1):
        self._size = size

    def __len__(self) -> int:
        return self._size

    def save(self, checkpoint_path):
        return None

    def resume(self, checkpoint_path):
        return None


class _FakeProduceStrategy:
    def __init__(self, status: ProduceBatchStatus = ProduceBatchStatus.NORMAL, cleanup_pause_time_s: float = 0.0):
        self.status = status
        self.cleanup_pause_time_s = cleanup_pause_time_s
        self.called_batch_sizes: list[int] = []
        self.called_rollout_steps: list[int] = []
        self.called_model_rollout_steps: list[int | None] = []
        self.called_update_events: list[object | None] = []
        self.cleanup_call_count = 0

    async def produce_batch(
        self,
        agent_loop,
        sampler,
        replay_buffer,
        batch_size: int,
        task_name: str,
        rollout_step: int = 0,
        model_rollout_step: int | None = None,
        update_event=None,
    ) -> ProduceBatchStatus:
        self.called_batch_sizes.append(batch_size)
        self.called_rollout_steps.append(rollout_step)
        self.called_model_rollout_steps.append(model_rollout_step)
        self.called_update_events.append(update_event)
        return self.status

    async def cleanup_pending_tasks(self, agent_loop, replay_buffer, task_name: str) -> float:
        self.cleanup_call_count += 1
        return self.cleanup_pause_time_s


class _FakeStatusProduceStrategy:
    def __init__(self, status: ProduceBatchStatus, pause_time_s: float):
        self.status = status
        self.pause_time_s = pause_time_s
        self.cleanup_call_count = 0
        self.called_rollout_steps: list[int] = []
        self.called_model_rollout_steps: list[int | None] = []
        self.called_update_events: list[object | None] = []

    async def produce_batch(
        self,
        agent_loop,
        sampler,
        replay_buffer,
        batch_size: int,
        task_name: str,
        rollout_step: int = 0,
        model_rollout_step: int | None = None,
        update_event=None,
    ) -> ProduceBatchStatus:
        self.called_rollout_steps.append(rollout_step)
        self.called_model_rollout_steps.append(model_rollout_step)
        self.called_update_events.append(update_event)
        return self.status

    async def cleanup_pending_tasks(self, agent_loop, replay_buffer, task_name: str) -> float:
        self.cleanup_call_count += 1
        return self.pause_time_s


class _FakeRolloutState:
    def __init__(self, uid: str, group_generate_time_s: float):
        self.uid = uid
        self.extra_fields = {GROUP_GENERATE_TIME_KEY: group_generate_time_s}


class _FakeStalenessRolloutState(_FakeRolloutState):
    def __init__(self, uid: str, group_generate_time_s: float, response_rollout_steps: list[int], seq_staleness: int = 0):
        super().__init__(uid, group_generate_time_s)
        self.response_rollout_steps = response_rollout_steps
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
        rollout_step: int = 0,
        model_rollout_step: int | None = None,
        update_event=None,
    ) -> ProduceBatchStatus:
        self.called_batch_sizes.append(batch_size)
        self.called_rollout_steps.append(rollout_step)
        self.called_model_rollout_steps.append(model_rollout_step)
        self.called_update_events.append(update_event)
        return self._statuses.pop(0) if self._statuses else ProduceBatchStatus.NORMAL


class _FakeReplayBuffer:
    def __init__(self, rollout_states_by_task: dict[str, list[list[str]]], leftover_counts: dict[tuple[str, Status], int]):
        self._rollout_states_by_task = rollout_states_by_task
        self._leftover_counts = leftover_counts

    async def get(self, batch_size: int, task_name: str, group_status: Status):
        assert group_status == Status.COMPLETED
        return self._rollout_states_by_task.get(task_name, [])

    async def count(self, task_name: str, group_status: Status):
        return self._leftover_counts.get((task_name, group_status), 0)

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

        result = await multi_task_manager.produce_batch(batch_size=7, rollout_step=3)

        self.assertEqual(result.task_batch_sizes, {"task_a": 5, "task_b": 2, "task_c": 0})
        self.assertEqual(strategy_a.called_batch_sizes, [5])
        self.assertEqual(strategy_b.called_batch_sizes, [2])
        self.assertEqual(strategy_c.called_batch_sizes, [])
        self.assertEqual(strategy_a.called_rollout_steps, [3])
        self.assertEqual(strategy_b.called_rollout_steps, [3])
        self.assertEqual(strategy_a.called_model_rollout_steps, [3])
        self.assertEqual(strategy_b.called_model_rollout_steps, [3])
        self.assertEqual(len(strategy_a.called_update_events), 1)
        self.assertFalse(strategy_a.called_update_events[0].is_set())
        self.assertEqual(len(strategy_b.called_update_events), 1)
        self.assertFalse(strategy_b.called_update_events[0].is_set())
        self.assertEqual(result.rollout_states, [["a-0"], ["a-1"], ["b-0"], ["b-1"], ["b-2"]])
        self.assertEqual(result.leftover_completed, 1)
        self.assertEqual(result.leftover_aborted, 2)
        self.assertEqual(result.leftover_expired, 0)
        self.assertIn("task_a", result.task_results)
        self.assertIn("task_b", result.task_results)
        self.assertIn("task_c", result.task_results)

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
            def get_task_batch_sizes(self, global_batch_size: int, rollout_step: int) -> dict[str, int]:
                self.observed_rollout_step = rollout_step
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

        result = await multi_task_manager.produce_batch(batch_size=2, rollout_step=9)

        self.assertEqual(multi_task_manager.observed_rollout_step, 9)
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

        result = await manager.produce_batch(batch_size=2, rollout_step=7)

        self.assertEqual(strategy.cleanup_call_count, 1)
        self.assertEqual(strategy.called_rollout_steps, [7])
        self.assertEqual(strategy.called_model_rollout_steps, [7])
        self.assertEqual(len(strategy.called_update_events), 1)
        self.assertFalse(strategy.called_update_events[0].is_set())
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
            await manager.produce_batch(batch_size=1, rollout_step=3)

    async def test_cleanup_pending_tasks_for_pause_product_for_update_sets_status_and_pause_time(self):
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

        pause_time_s = await manager.cleanup_pending_tasks(pause_product_for_update=True)

        self.assertEqual(pause_time_s, 2.5)
        self.assertEqual(strategy.cleanup_call_count, 1)
        self.assertTrue(manager._update_event.is_set())
        self.assertEqual(manager._status, AgentLoopManagerStatus.UPDATE_ABORT)
        self.assertEqual(manager._pause_time_s, 2.5)

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

        result = await manager.get_batch(batch_size=2, rollout_step=11)

        self.assertEqual(result.status, ProduceBatchStatus.EXPIRED_BATCH)
        self.assertEqual(result.rollout_states, [])

    async def test_get_batch_refreshes_seq_staleness_from_buffer(self):
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
            replay_buffer=_FakeReplayBuffer(
                rollout_states_by_task={
                    "task_a": [[_FakeStalenessRolloutState("a-0", 0.2, response_rollout_steps=[4], seq_staleness=0)]],
                },
                leftover_counts={},
            ),
        )

        result = await manager.get_batch(batch_size=1, rollout_step=9)

        self.assertEqual(result.rollout_states[0][0].seq_staleness, 5)

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

        status = await manager._produce_batch_to_buffer(batch_size=3, rollout_step=5, model_rollout_step=5)

        self.assertEqual(status, ProduceBatchStatus.UPDATE_ABORT)

    async def test_produce_loop_waits_for_continue_product_and_stops_on_finish(self):
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

        loop_task = asyncio.create_task(manager.produce_loop(batch_size=1, start_rollout_step=3))
        await self._wait_until(lambda: manager._status == AgentLoopManagerStatus.EXPIRED_BATCH)
        self.assertEqual(manager._status, AgentLoopManagerStatus.EXPIRED_BATCH)
        self.assertEqual(strategy.called_rollout_steps[:2], [3, 4])

        manager.continue_product(model_rollout_step=9)
        await self._wait_until(lambda: len(strategy.called_rollout_steps) >= 3)
        self.assertEqual(manager._status, AgentLoopManagerStatus.NORMAL)
        self.assertEqual(strategy.called_rollout_steps[:3], [3, 4, 4])
        self.assertEqual(strategy.called_model_rollout_steps[2], 9)

        manager._status = AgentLoopManagerStatus.FINISH
        manager._finish_event.set()
        await asyncio.wait_for(loop_task, timeout=1.0)
