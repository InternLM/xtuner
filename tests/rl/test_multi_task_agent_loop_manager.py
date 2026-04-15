import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from xtuner.v1.data_proto import Status
from xtuner.v1.rl.agent_loop import AgentLoopManager, AgentLoopManagerConfig
from xtuner.v1.rl.agent_loop.manager_base import _TaskRunner
from xtuner.v1.rl.agent_loop.producer import ProducerTimings


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
    def __init__(
        self,
        generate_times_s: list[float],
        *,
        over_sample_threshold: float = 0.0,
        enable_partial_rollout: bool = False,
    ):
        self.generate_times_s = generate_times_s
        self.over_sample_threshold = over_sample_threshold
        self.enable_partial_rollout = enable_partial_rollout
        self.calls: list[dict[str, object]] = []

    async def produce_batch(
        self,
        agent_loop,
        sampler,
        replay_buffer,
        batch_size: int,
        task_name: str,
        rollout_step: int = 0,
    ) -> ProducerTimings:
        self.calls.append(
            {
                "agent_loop": agent_loop,
                "sampler": sampler,
                "replay_buffer": replay_buffer,
                "batch_size": batch_size,
                "task_name": task_name,
                "rollout_step": rollout_step,
                "over_sample_threshold": self.over_sample_threshold,
                "enable_partial_rollout": self.enable_partial_rollout,
            }
        )
        return ProducerTimings(generate_times_s=self.generate_times_s)


class _FakeReplayBuffer:
    def __init__(self, rollout_states_by_task: dict[str, list[list[str]]], leftover_counts: dict[tuple[str, Status], int]):
        self._rollout_states_by_task = {
            task_name: list(rollout_states) for task_name, rollout_states in rollout_states_by_task.items()
        }
        self._leftover_counts = dict(leftover_counts)

    async def get(self, batch_size: int, task_name: str, group_status: Status):
        assert group_status == Status.COMPLETED
        rollout_states = self._rollout_states_by_task.get(task_name, [])
        selected = rollout_states[:batch_size]
        self._rollout_states_by_task[task_name] = rollout_states[batch_size:]
        return selected

    async def count(self, task_name: str, group_status: Status):
        if group_status == Status.COMPLETED:
            return len(self._rollout_states_by_task.get(task_name, []))
        return self._leftover_counts.get((task_name, group_status), 0)


def _fake_agent_loop(rollout_ctl=None):
    return SimpleNamespace(rollout_ctl=rollout_ctl or SimpleNamespace())


class TestManagerCompatibility(unittest.TestCase):
    def test_agent_loop_manager_config_is_public(self):
        self.assertIsNotNone(AgentLoopManagerConfig)


class TestSingleTaskAgentLoopManager(unittest.IsolatedAsyncioTestCase):
    async def test_fullasync_produce_batch_attaches_single_task_metadata(self):
        rollout_ctl = SimpleNamespace()
        strategy = _FakeProduceStrategy(
            generate_times_s=[1.0, 3.0],
            over_sample_threshold=0.5,
            enable_partial_rollout=True,
        )
        replay_buffer = _FakeReplayBuffer(
            rollout_states_by_task={"task_a": [["a-0"], ["a-1"], ["a-2"]]},
            leftover_counts={
                ("task_a", Status.ABORTED): 2,
                ("task_a", Status.EXPIRED): 3,
            },
        )
        manager = AgentLoopManager(
            task_runners=[_TaskRunner(
                task_name="task_a",
                agent_loop=_fake_agent_loop(rollout_ctl),
                produce_strategy=strategy,
                sampler=_FakeSampler(),
                weight=1.0,
                order=0,
            )],
            replay_buffer=replay_buffer,
            logger=MagicMock(),
        )

        with patch(
            "xtuner.v1.rl.agent_loop.agent_loop_manager.continue_generation",
            new=AsyncMock(),
        ) as continue_mock:
            await manager.fullasync_produce_batch(batch_size=2, rollout_step=7)
            result = await manager.get_completed_batch(batch_size=2, rollout_step=7)

        self.assertEqual(len(strategy.calls), 1)
        self.assertEqual(strategy.calls[0]["batch_size"], 2)
        self.assertEqual(strategy.calls[0]["task_name"], "task_a")
        self.assertEqual(strategy.calls[0]["rollout_step"], 7)
        self.assertEqual(strategy.calls[0]["over_sample_threshold"], 0.5)
        self.assertTrue(strategy.calls[0]["enable_partial_rollout"])
        continue_mock.assert_awaited_once_with(rollout_ctl)
        self.assertEqual(result.rollout_states, [["a-0"], ["a-1"]])
        self.assertEqual(result.task_batch_sizes, {"task_a": 2})
        self.assertEqual(result.required_task_batch_sizes, {"task_a": 2})
        self.assertEqual(result.leftover_completed, 1)
        self.assertEqual(result.leftover_aborted, 2)
        self.assertEqual(result.leftover_expired, 3)
        self.assertEqual(result.group_gen_count, 2)
        self.assertAlmostEqual(result.group_gen_mean_s, 2.0)
        self.assertIsNotNone(result.task_results)
        self.assertIn("task_a", result.task_results)
        self.assertIsNone(result.task_results["task_a"].task_results)
        self.assertEqual(result.task_results["task_a"].rollout_states, [["a-0"], ["a-1"]])


class TestMultiTaskAgentLoopManager(unittest.IsolatedAsyncioTestCase):
    async def test_fullasync_produce_batch_allocates_by_weight_and_aggregates_results(self):
        rollout_ctl = SimpleNamespace()
        strategy_b = _FakeProduceStrategy(generate_times_s=[1.0, 1.0, 1.0])
        strategy_a = _FakeProduceStrategy(
            generate_times_s=[2.0, 2.0],
            over_sample_threshold=0.25,
            enable_partial_rollout=True,
        )
        strategy_c = _FakeProduceStrategy(generate_times_s=[])
        replay_buffer = _FakeReplayBuffer(
            rollout_states_by_task={
                "task_b": [["b-0"], ["b-1"], ["b-2"]],
                "task_a": [["a-0"], ["a-1"], ["a-2"], ["a-3"], ["a-4"]],
                "task_c": [],
            },
            leftover_counts={
                ("task_b", Status.ABORTED): 2,
            },
        )
        manager = AgentLoopManager(
            task_runners=[
                _TaskRunner(
                    task_name="task_b",
                    agent_loop=_fake_agent_loop(rollout_ctl),
                    produce_strategy=strategy_b,
                    sampler=_FakeSampler(),
                    weight=1.0,
                    order=0,
                ),
                _TaskRunner(
                    task_name="task_a",
                    agent_loop=_fake_agent_loop(rollout_ctl),
                    produce_strategy=strategy_a,
                    sampler=_FakeSampler(),
                    weight=2.0,
                    order=1,
                ),
                _TaskRunner(
                    task_name="task_c",
                    agent_loop=_fake_agent_loop(rollout_ctl),
                    produce_strategy=strategy_c,
                    sampler=_FakeSampler(),
                    weight=0.0,
                    order=2,
                ),
            ],
            replay_buffer=replay_buffer,
            logger=MagicMock(),
        )

        manager.set_sync_interval_context(trigger_parameter_sync_step=1, total_train_steps=3)
        with patch(
            "xtuner.v1.rl.agent_loop.agent_loop_manager.continue_generation",
            new=AsyncMock(),
        ) as continue_mock:
            await manager.fullasync_produce_batch(batch_size=7, rollout_step=3)
            result = await manager.get_completed_batch(batch_size=7, rollout_step=3)

        self.assertEqual(strategy_a.calls[0]["batch_size"], 5)
        self.assertEqual(strategy_b.calls[0]["batch_size"], 2)
        self.assertEqual(strategy_a.calls[0]["rollout_step"], 3)
        self.assertEqual(strategy_b.calls[0]["rollout_step"], 3)
        self.assertEqual(strategy_a.calls[0]["over_sample_threshold"], 0.25)
        self.assertTrue(strategy_a.calls[0]["enable_partial_rollout"])
        self.assertEqual(strategy_b.calls[0]["over_sample_threshold"], 0.0)
        self.assertFalse(strategy_b.calls[0]["enable_partial_rollout"])
        self.assertEqual(strategy_c.calls, [])
        continue_mock.assert_awaited_once_with(rollout_ctl)
        self.assertEqual(result.task_batch_sizes, {"task_b": 2, "task_a": 5, "task_c": 0})
        self.assertEqual(result.required_task_batch_sizes, {"task_b": 2, "task_a": 5, "task_c": 0})
        self.assertEqual(
            result.rollout_states,
            [["b-0"], ["b-1"], ["a-0"], ["a-1"], ["a-2"], ["a-3"], ["a-4"]],
        )
        self.assertEqual(result.leftover_completed, 1)
        self.assertEqual(result.leftover_aborted, 2)
        self.assertEqual(result.leftover_expired, 0)
        self.assertEqual(result.group_gen_count, 5)
        self.assertAlmostEqual(result.group_gen_mean_s, 1.4)
        self.assertEqual(result.group_gen_p50_s, 1.0)
        self.assertEqual(result.group_gen_p99_s, 2.0)
        self.assertEqual(result.group_gen_p99_p50_ratio, 2.0)
        self.assertIsNotNone(result.task_results)
        self.assertIn("task_a", result.task_results)
        self.assertIn("task_b", result.task_results)
        self.assertIn("task_c", result.task_results)
        self.assertEqual(result.task_results["task_c"].rollout_states, [])

    async def test_custom_get_task_batch_sizes_can_disable_tasks(self):
        rollout_ctl = SimpleNamespace()
        strategy_a = _FakeProduceStrategy(generate_times_s=[2.0])
        strategy_b = _FakeProduceStrategy(generate_times_s=[1.0, 1.0])
        replay_buffer = _FakeReplayBuffer(
            rollout_states_by_task={
                "task_a": [["a-0"]],
                "task_b": [["b-0"], ["b-1"]],
            },
            leftover_counts={},
        )

        class _CustomBatchManager(AgentLoopManager):
            def get_task_batch_sizes(self, train_batch_size: int, rollout_step: int) -> dict[str, int]:
                self.observed_rollout_step = rollout_step
                return {"task_a": 0, "task_b": train_batch_size}

        manager = _CustomBatchManager(
            task_runners=[
                _TaskRunner(
                    task_name="task_a",
                    agent_loop=_fake_agent_loop(rollout_ctl),
                    produce_strategy=strategy_a,
                    sampler=_FakeSampler(),
                    weight=1.0,
                    order=0,
                ),
                _TaskRunner(
                    task_name="task_b",
                    agent_loop=_fake_agent_loop(rollout_ctl),
                    produce_strategy=strategy_b,
                    sampler=_FakeSampler(),
                    weight=1.0,
                    order=1,
                ),
            ],
            replay_buffer=replay_buffer,
            logger=MagicMock(),
        )

        manager.set_sync_interval_context(trigger_parameter_sync_step=1, total_train_steps=9)
        with patch(
            "xtuner.v1.rl.agent_loop.agent_loop_manager.continue_generation",
            new=AsyncMock(),
        ) as continue_mock:
            await manager.fullasync_produce_batch(batch_size=2, rollout_step=9)
            result = await manager.get_completed_batch(batch_size=2, rollout_step=9)

        self.assertEqual(manager.observed_rollout_step, 9)
        self.assertEqual(strategy_a.calls, [])
        self.assertEqual(strategy_b.calls[0]["batch_size"], 2)
        continue_mock.assert_awaited_once_with(rollout_ctl)
        self.assertEqual(result.task_batch_sizes, {"task_a": 0, "task_b": 2})
        self.assertEqual(result.rollout_states, [["b-0"], ["b-1"]])
