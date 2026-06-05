"""AgentLoopManager 共卡 produce_batch 的 public 行为测试。

本文件从旧的 test_multi_task_agent_loop_manager.py 迁入共卡路径测试：
- produce_batch 按 task 权重分配 batch，并按 task 名稳定返回训练数据。
- 自定义 get_task_batch_sizes 可以禁用某些 task。
- produce_batch 会汇总 producer 收尾耗时和 group 生成耗时。
- 共卡 produce_batch 必须返回非空训练 batch。
- 多 task 中任一 task 返回 UPDATE_WEIGHT_AND_ABORT 时，public 结果状态会体现该中断。

非共卡 get_batch / produce_loop / pause_continue 编排暂不迁入 PR-fast。
"""

import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from xtuner.v1.data_proto.rl_data import Status
from xtuner.v1.rl.agent_loop_manager.agent_loop_manager import (
    AgentLoopManager,
    AgentLoopManagerConfig,
    TaskSpecConfig,
    _TaskRunner,
)
from xtuner.v1.rl.agent_loop_manager.producer import GROUP_GENERATE_TIME_KEY, ProduceBatchStatus


class _FakeSampler:
    def __init__(self, size: int = 1):
        self._size = size

    def __len__(self) -> int:
        return self._size


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

    async def produce_batch(self, ctx) -> ProduceBatchStatus:
        self.called_batch_sizes.append(ctx.task_batch_size)
        self.called_train_steps.append(ctx.train_step)
        self.called_model_steps.append(ctx.model_step)
        self.called_update_events.append(ctx.update_event)
        self.called_update_event_states.append(None if ctx.update_event is None else ctx.update_event.is_set())
        self.called_progresses.append(ctx.progress)
        return self.status

    async def pause_produce(self, ctx) -> float:
        self.cleanup_call_count += 1
        self.cleanup_model_steps.append(ctx.model_step)
        self.cleanup_progresses.append(ctx.progress)
        return self.cleanup_pause_time_s

    def is_model_expired(self, train_step: int, model_step: int) -> bool:
        return False


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

    async def produce_batch(self, ctx) -> ProduceBatchStatus:
        self.called_train_steps.append(ctx.train_step)
        self.called_model_steps.append(ctx.model_step)
        self.called_update_events.append(ctx.update_event)
        self.called_update_event_states.append(None if ctx.update_event is None else ctx.update_event.is_set())
        self.called_progresses.append(ctx.progress)
        return self.status

    async def pause_produce(self, ctx) -> float:
        self.cleanup_call_count += 1
        self.cleanup_model_steps.append(ctx.model_step)
        self.cleanup_progresses.append(ctx.progress)
        return self.pause_time_s

    def is_model_expired(self, train_step: int, model_step: int) -> bool:
        return False


class _FakeRolloutState:
    def __init__(self, uid: str, group_generate_time_s: float):
        self.uid = uid
        self.extra_fields = {GROUP_GENERATE_TIME_KEY: group_generate_time_s}


class _FakeReplayBuffer:
    def __init__(self, rollout_states_by_task: dict[str, list[list[Any]]], leftover_counts: dict[tuple[str, Status], int]):
        self._rollout_states_by_task = rollout_states_by_task
        self._leftover_counts = leftover_counts
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
        *,
        task_stale_thresholds: dict[str, int],
        current_train_step: int,
        statuses: list[Status] | None = None,
    ):
        expired_counts = {}
        for task_name, stale_threshold in task_stale_thresholds.items():
            self.refresh_staleness_calls.append(
                (task_name, current_train_step, stale_threshold, tuple(statuses or ()))
            )
            expired_counts[task_name] = 0
        return expired_counts

    async def is_ready(self, task_batch_sizes: dict[str, int], *, group_status: Status = Status.COMPLETED):
        for task_name, batch_size in task_batch_sizes.items():
            if await self.count(task_name, group_status) < batch_size:
                return False
        return True

    async def take_batch(self, task_batch_sizes: dict[str, int], *, group_status: Status = Status.COMPLETED):
        batch_by_task = {}
        consumed_counts = {}
        for task_name, batch_size in task_batch_sizes.items():
            batch = await self.get(batch_size, task_name, group_status)
            batch_by_task[task_name] = batch
            consumed_counts[task_name] = len(batch)
        return batch_by_task, consumed_counts

    async def count_statuses(self, task_names: list[str], statuses: list[Status]):
        return {
            task_name: {status: self._leftover_counts.get((task_name, status), 0) for status in statuses}
            for task_name in task_names
        }


def _fake_agent_loop():
    rollout_ctl = MagicMock()
    rollout_ctl.continue_generation.remote = AsyncMock()
    rollout_ctl.pause_generation.remote = AsyncMock()
    rollout_ctl.get_rollout_metadata.remote = AsyncMock(return_value={"server_url_dict": {}})
    agent_loop = MagicMock()
    agent_loop.rollout_ctl = rollout_ctl
    return agent_loop


class TestMultiTaskAgentLoopManager(unittest.IsolatedAsyncioTestCase):
    def test_manager_config_accepts_single_task_spec(self):
        # 单 task 配置可以直接传入，兼容最小 AgentLoopManager 配置。
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
        # 共卡 produce_batch 按 task 权重分配 batch，并按 task 名稳定返回训练数据和 leftover 统计。
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

        result = await multi_task_manager.produce_batch(batch_size=7, train_step=3, model_step=2)

        self.assertEqual(result.task_batch_sizes, {"task_a": 5, "task_b": 2, "task_c": 0})
        self.assertEqual(result.rollout_states, [["a-0"], ["a-1"], ["b-0"], ["b-1"]])
        self.assertEqual(result.leftover_init, 0)
        self.assertEqual(result.leftover_completed, 1)
        self.assertEqual(result.leftover_aborted, 2)
        self.assertEqual(result.leftover_expired, 0)
        self.assertEqual(result.leftover_failed, 0)
        self.assertEqual(result.leftover_filtered, 0)
        self.assertIn("task_a", result.task_results)
        self.assertIn("task_b", result.task_results)
        self.assertIn("task_c", result.task_results)

    async def test_custom_get_task_batch_sizes_can_disable_tasks(self):
        # 自定义 task batch size 可以禁用某个 task，训练 batch 只从启用 task 取数。
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
        # 共卡 produce_batch 会把 producer 收尾耗时和 rollout group 生成耗时汇总到结果中。
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

        self.assertEqual(result.group_gen_count, 2)
        self.assertAlmostEqual(result.group_gen_mean_s, 0.75)
        self.assertAlmostEqual(result.group_gen_p50_s, 1.0)
        self.assertAlmostEqual(result.group_gen_p99_s, 1.0)
        self.assertAlmostEqual(result.group_gen_pause_time_s, 1.25)

    async def test_produce_batch_requires_non_empty_rollout_states(self):
        # 共卡 produce_batch 不能返回空训练 batch，空/expired 语义只能走非共卡 get_batch。
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

    async def test_produce_batch_returns_update_abort_when_any_task_requests_abort(self):
        # 多 task 共卡生产时，任一 task 返回 UPDATE_WEIGHT_AND_ABORT 会体现在 public 结果状态中。
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
                    produce_strategy=_FakeProduceStrategy(status=ProduceBatchStatus.UPDATE_WEIGHT_AND_ABORT),
                    sampler=_FakeSampler(),
                    weight=1.0,
                    order=2,
                ),
            ],
            replay_buffer=_FakeReplayBuffer(
                rollout_states_by_task={
                    "task_a": [["a-0"]],
                    "task_b": [["b-0"]],
                    "task_c": [["c-0"]],
                },
                leftover_counts={},
            ),
        )

        result = await manager.produce_batch(batch_size=3, train_step=6, model_step=5)

        self.assertEqual(result.status, ProduceBatchStatus.UPDATE_WEIGHT_AND_ABORT)
        self.assertEqual(result.rollout_states, [["a-0"], ["b-0"], ["c-0"]])
