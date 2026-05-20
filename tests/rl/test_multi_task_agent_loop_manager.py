"""AgentLoopManager 的 public 行为测试。

Good Tests:
- 通过 AgentLoopManager 的公开入口验证 manager 对 rollout 生产和 replay buffer 消费的编排行为。
- 断言 ProduceBatchResult、checkpoint/resume 结果、pause/continue 对外可见顺序、get_batch expired 语义。
- 仅在验证 pause/continue 顺序等外部协议时，少量使用 fake rollout controller 观察调用边界。

Bad Tests:
- 不直接调用 `_produce_batch_to_buffer` 等私有方法。
- 不把 `_status`、`_update_event`、`_model_step`、`_produce_progress` 的内部推进当成主要断言目标。
- 不重复测试 ProduceProgress 或 _PendingTasks 的深模块契约；它们有独立测试文件。

本文件主要覆盖的 public 行为:
- 共卡 produce_batch 按 task 权重分配并返回非空训练 batch。
- 非共卡 get_batch 等待 ready batch，并处理 Expired Produce Batch 的空/非空/fail-fast 语义。
- pause_produce / continue_produce 控制 rollout controller 和后台 produce_loop 恢复。
- save/resume/shutdown 对 producer 生命周期和 checkpoint 的外部行为。
"""

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
        self.pending_task_count_value = 0

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
        # fake strategy 的过期状态由用例显式返回 status 控制。
        return False

    def pending_task_count(self) -> int:
        return self.pending_task_count_value


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
        self.pending_task_count_value = 0

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
        # fake strategy 的过期状态由用例显式返回 status 控制。
        return False

    def pending_task_count(self) -> int:
        return self.pending_task_count_value


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

    async def produce_batch(self, ctx) -> ProduceBatchStatus:
        self.called_batch_sizes.append(ctx.task_batch_size)
        self.called_train_steps.append(ctx.train_step)
        self.called_model_steps.append(ctx.model_step)
        self.called_update_events.append(ctx.update_event)
        self.called_update_event_states.append(None if ctx.update_event is None else ctx.update_event.is_set())
        self.called_progresses.append(ctx.progress)
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
            for group in self._rollout_states_by_task.get(task_name, []):
                for state in group:
                    response_model_steps = getattr(state, "response_model_steps", None) or []
                    if response_model_steps and hasattr(state, "seq_staleness"):
                        state.seq_staleness = calculate_seq_staleness(
                            min(response_model_steps), current_train_step
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
            task_name: {
                status: self._leftover_counts.get((task_name, status), 0)
                for status in statuses
            }
            for task_name in task_names
        }

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
    rollout_ctl.cleanup_after_pause.remote = AsyncMock()
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
        # 验证单 task 配置可以直接传入，兼容最小 AgentLoopManager 配置。
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
        # 验证共卡 produce_batch 按 task 权重分配 batch，并按 task 名稳定返回训练数据和 leftover 统计。
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

    def test_save_and_resume_roundtrip_restores_paused_manager_state(self):
        # 验证 checkpoint/resume 会保留模型版本和 producer 暂停态，并恢复 sampler / replay buffer。
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
        self.assertEqual(manager._status, AgentLoopManagerStatus.UPDATE_WEIGHT_AND_ABORT)
        self.assertTrue(manager._update_event.is_set())
        self.assertFalse(manager._finish_event.is_set())
        self.assertEqual(manager._pause_time_s, 0.0)
        self.assertEqual(manager._model_step, 7)
        self.assertEqual(sampler.saved_paths, [Path(tmp_dir) / manager._TASK_CHECKPOINT_DIR / "task_a"])
        self.assertEqual(sampler.resumed_paths, [Path(tmp_dir) / manager._TASK_CHECKPOINT_DIR / "task_a"])
        self.assertEqual(replay_buffer.saved_paths, [Path(tmp_dir)])
        self.assertEqual(replay_buffer.resumed_paths, [Path(tmp_dir)])

    def test_save_rejects_pending_async_tasks(self):
        # 验证保存前必须先清空 pending rollout task，避免 checkpoint 缺失后台生产结果。
        strategy = _FakeProduceStrategy()
        strategy.pending_task_count_value = 1
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
        # 验证自定义 task batch size 可以禁用某个 task，训练 batch 只从启用 task 取数。
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
        # 验证共卡 produce_batch 会把 producer 收尾耗时和 rollout group 生成耗时汇总到结果中。
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
        # 验证共卡 produce_batch 不能返回空训练 batch，空/expired 语义只能走非共卡 get_batch。
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
        # 验证 pause_produce 会先进入暂停态再暂停 rollout controller，并返回 producer drain 耗时。
        strategy = _FakeProduceStrategy(cleanup_pause_time_s=2.5)
        agent_loop = _fake_agent_loop()
        manager = AgentLoopManager(
            task_runners=[
                _TaskRunner(
                    task_name="task_a",
                    agent_loop=agent_loop,
                    produce_strategy=strategy,
                    sampler=_FakeSampler(),
                    weight=1.0,
                    order=0,
                ),
            ],
            replay_buffer=_FakeReplayBuffer({}, {}),
        )

        async def _assert_update_state_before_controller_pause():
            self.assertTrue(manager._update_event.is_set())
            self.assertEqual(manager._status, AgentLoopManagerStatus.UPDATE_WEIGHT_AND_ABORT)

        agent_loop.rollout_ctl.pause_generation.remote.side_effect = _assert_update_state_before_controller_pause

        pause_time_s = await manager.pause_produce(use_global_progress=True)

        self.assertEqual(pause_time_s, 2.5)
        agent_loop.rollout_ctl.pause_generation.remote.assert_awaited_once_with()
        self.assertEqual(strategy.cleanup_call_count, 1)
        self.assertEqual(len(strategy.cleanup_progresses), 1)
        self.assertEqual(strategy.cleanup_model_steps, [0])
        self.assertIs(strategy.cleanup_progresses[0], manager._produce_progress)
        self.assertTrue(manager._update_event.is_set())
        self.assertEqual(manager._status, AgentLoopManagerStatus.UPDATE_WEIGHT_AND_ABORT)
        self.assertEqual(manager._pause_time_s, 2.5)

    async def test_pause_produce_validates_progress_selection_before_state_change(self):
        # 验证 pause_produce 参数非法时 fail fast，且不会提前改变 producer 控制状态。
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

    async def test_continue_produce_resumes_rollout_controller_before_normal_status(self):
        # 验证 continue_produce 必须先恢复 rollout controller，再让后台 produce_loop 继续生产。
        agent_loop = _fake_agent_loop()
        manager = AgentLoopManager(
            task_runners=[
                _TaskRunner(
                    task_name="task_a",
                    agent_loop=agent_loop,
                    produce_strategy=_FakeProduceStrategy(),
                    sampler=_FakeSampler(),
                    weight=1.0,
                    order=0,
                ),
            ],
            replay_buffer=_FakeReplayBuffer({}, {}),
        )
        manager._status = AgentLoopManagerStatus.UPDATE_WEIGHT_AND_ABORT
        manager._update_event.set()

        async def _assert_paused_before_controller_resume():
            self.assertEqual(manager._status, AgentLoopManagerStatus.UPDATE_WEIGHT_AND_ABORT)
            self.assertTrue(manager._update_event.is_set())

        agent_loop.rollout_ctl.continue_generation.remote.side_effect = _assert_paused_before_controller_resume

        await manager.continue_produce(model_step=6)

        agent_loop.rollout_ctl.continue_generation.remote.assert_awaited_once_with()
        self.assertEqual(manager._model_step, 6)
        self.assertEqual(manager._status, AgentLoopManagerStatus.NORMAL)
        self.assertFalse(manager._update_event.is_set())

    async def test_get_batch_returns_empty_expired_when_model_step_is_newer_than_rollout(self):
        # 验证已有更新 Model Step 时，Expired Produce Batch 可以返回空 batch 让 trainer 直接同步。
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
        manager._model_step = 9
        manager._pause_time_s = 1.5

        result = await manager.get_batch(batch_size=2, train_step=11)

        self.assertEqual(result.status, ProduceBatchStatus.EXPIRED_BATCH)
        self.assertEqual(result.rollout_states, [])
        self.assertEqual(result.group_gen_pause_time_s, 1.5)
        self.assertEqual(manager._pause_time_s, 0.0)

    async def test_get_batch_returns_ready_batch_as_expired_when_no_newer_model_exists(self):
        # 验证没有更新 Model Step 时，Expired Produce Batch 必须返回已 ready 的训练 batch。
        replay_buffer = _FakeReplayBuffer(
            rollout_states_by_task={
                "task_a": [
                    [_FakeRolloutState("a-0", 0.2)],
                    [_FakeRolloutState("a-1", 0.3)],
                ],
            },
            leftover_counts={("task_a", Status.COMPLETED): 2},
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
        manager._status = AgentLoopManagerStatus.EXPIRED_BATCH
        manager._model_step = 8

        result = await manager.get_batch(batch_size=2, train_step=9)

        self.assertEqual(result.status, ProduceBatchStatus.EXPIRED_BATCH)
        self.assertEqual([group[0].uid for group in result.rollout_states], ["a-0", "a-1"])
        self.assertEqual(manager._produce_progress.next_consumer_step, 10)
        self.assertEqual(manager._produce_progress.consumed_samples["task_a"], 2)

    async def test_get_batch_fails_fast_when_expired_without_ready_batch_or_newer_model(self):
        # 验证 expired 且没有更新 Model Step / ready batch 时 fail fast，并打印调度不变量。
        replay_buffer = _FakeReplayBuffer(
            rollout_states_by_task={},
            leftover_counts={
                ("task_a", Status.INIT): 3,
                ("task_a", Status.COMPLETED): 1,
                ("task_a", Status.ABORTED): 2,
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
        manager._status = AgentLoopManagerStatus.EXPIRED_BATCH
        manager._model_step = 8
        manager._produce_progress.producer_future_step = 9
        manager._produce_progress.target_upto_future_step = 8
        manager._produce_progress.target_samples["task_a"] = 2
        manager._produce_progress.consumed_samples["task_a"] = 1

        with self.assertRaisesRegex(
            RuntimeError, "train_step=9.*current_model_step=8.*rollout_model_step=8"
        ) as caught:
            await manager.get_batch(batch_size=2, train_step=9)
        message = str(caught.exception)

        self.assertIn("manager_status=EXPIRED_BATCH", message)
        self.assertIn("producer_future_step=9", message)
        self.assertIn("next_consumer_step=9", message)
        self.assertIn("target_upto_future_step=8", message)
        self.assertIn("target_samples={'task_a': 2}", message)
        self.assertIn("consumed_samples={'task_a': 1}", message)
        self.assertIn("task_batch_sizes={'task_a': 2}", message)
        self.assertIn("leftover_status_counts=", message)

    async def test_get_batch_refreshes_staleness_at_entry(self):
        # 验证 get_batch 入口和成功消费后都会刷新 completed / aborted 的 staleness。
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

    async def test_get_batch_returns_raw_reward_stats_from_progress(self):
        # 验证 get_batch 会带出 producer 累积的 raw reward 统计，并在读取后清零。
        replay_buffer = _FakeReplayBuffer(
            rollout_states_by_task={"task_a": [[_FakeRolloutState("a-0", 0.2)]]},
            leftover_counts={("task_a", Status.COMPLETED): 1},
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
        manager._produce_progress.add_raw_rewards("task_a", 1.25, 2)

        result = await manager.get_batch(batch_size=1, train_step=9)

        self.assertEqual(result.raw_rewards_sum, 1.25)
        self.assertEqual(result.raw_rewards_count, 2)
        self.assertEqual(manager._produce_progress.consume_raw_rewards("task_a"), (0.0, 0))

    async def test_get_batch_waits_until_requested_batch_size_is_ready(self):
        # 验证非共卡 get_batch 会等待 replay buffer 凑齐当前 Train Batch Size 后再取数。
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

    async def test_produce_batch_returns_update_abort_when_any_task_requests_abort(self):
        # 验证多 task 共卡生产时，任一 task 返回 UPDATE_WEIGHT_AND_ABORT 会体现在 public 结果状态中。
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

    async def test_produce_loop_waits_for_continue_produce_and_stops_on_finish(self):
        # 验证后台 produce_loop 遇到 Expired Produce Batch 后等待 trainer 显式 continue_produce 恢复。
        strategy = _SequencedProduceStrategy(
            statuses=[ProduceBatchStatus.NORMAL, ProduceBatchStatus.EXPIRED_BATCH, ProduceBatchStatus.NORMAL],
        )
        agent_loop = _fake_agent_loop()
        manager = AgentLoopManager(
            task_runners=[
                _TaskRunner(
                    task_name="task_a",
                    agent_loop=agent_loop,
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
        agent_loop.rollout_ctl.continue_generation.remote.assert_not_awaited()

        await manager.continue_produce(model_step=9)
        await self._wait_until(lambda: len(strategy.called_train_steps) >= 3)
        self.assertEqual(manager._status, AgentLoopManagerStatus.NORMAL)
        self.assertEqual(strategy.called_train_steps[:3], [3, 4, 4])
        self.assertEqual(strategy.called_model_steps[2], 9)
        agent_loop.rollout_ctl.continue_generation.remote.assert_awaited_once_with()

        manager.shutdown()
        await asyncio.wait_for(loop_task, timeout=1.0)

    async def test_shutdown_stops_background_produce_loop(self):
        # 验证 shutdown 是后台 producer 的公开退出入口，produce_loop 可以被它正常收口。
        strategy = _SequencedProduceStrategy(statuses=[ProduceBatchStatus.NORMAL])
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

        loop_task = asyncio.create_task(manager.produce_loop(batch_size=1))
        await self._wait_until(lambda: len(strategy.called_train_steps) >= 1)
        manager.shutdown()

        await asyncio.wait_for(loop_task, timeout=1.0)
