import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from xtuner.v1.data_proto.rl_data import Status
from xtuner.v1.rl.agent_loop_manager import (
    AsyncProduceStrategyConfig,
    ProduceBatchStatus,
    ProduceContext,
    ProduceProgress,
    SamplerConfig,
    SyncProduceStrategyConfig,
)
from xtuner.v1.rl.agent_loop_manager.producer import _PendingTasks
from xtuner.v1.rl.replay_buffer import AsyncReplayBufferConfig


class MockRolloutState:
    def __init__(self, id, seq_staleness=1, status=Status.COMPLETED):
        self.id = id
        self.uid = id
        self.status = status
        self.seq_staleness = seq_staleness
        self.response_ids = []
        self.extra_fields = {}


class TestProducer(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # 1. 模拟 DataloaderConfig 和 Dataloader
        self.mock_dataloader_cfg = MagicMock()
        self.mock_dataloader = MagicMock()
        # 模拟 next(dataloader_iter) 返回 [RolloutState]
        self.mock_dataloader.__iter__.return_value = iter([[MockRolloutState(i)] for i in range(100)])
        self.mock_dataloader_cfg.build.return_value = self.mock_dataloader

        # 2. 模拟 Tokenizer
        self.mock_tokenizer = MagicMock()

        # 3. 准备 ReplayBuffer
        replay_buffer_cfg = AsyncReplayBufferConfig()
        self.replay_buffer = replay_buffer_cfg.build()

    def _build_sampler(self):
        sampler_cfg = SamplerConfig.model_construct(dataloader_cfg=self.mock_dataloader_cfg)
        return sampler_cfg.build(self.mock_tokenizer, self.replay_buffer)

    def _build_progress(
        self,
        task_name: str,
        target: int,
        train_step: int = 0,
        consumed: int = 0,
    ) -> ProduceProgress:
        return ProduceProgress(
            next_consumer_step=train_step,
            producer_future_step=train_step,
            consumed_samples={task_name: consumed},
            target_samples={task_name: target},
            target_upto_future_step=train_step,
        )

    def _build_agent_loop(self, sleep_by_id: dict[int, float] | None = None):
        mock_agent_loop = MagicMock()
        mock_agent_loop.rollout_ctl.continue_generation.remote = AsyncMock(return_value=None)
        mock_agent_loop.rollout_ctl.pause_generation.remote = AsyncMock(return_value=None)
        mock_agent_loop.rollout_ctl.get_rollout_metadata.remote = AsyncMock(return_value={"server_url_dict": {}})

        sleep_by_id = sleep_by_id or {}

        async def mock_gen(rs, **kwargs):
            await asyncio.sleep(sleep_by_id.get(rs[0].id, 0.0))
            for r in rs:
                r.seq_staleness = kwargs.get("model_step", kwargs.get("train_step", 0))
                r.status = Status.COMPLETED
            return rs

        mock_agent_loop.generate_group = mock_gen
        return mock_agent_loop

    def _build_context(
        self,
        strategy,
        task_name: str,
        agent_loop,
        sampler,
        *,
        batch_size: int,
        train_step: int = 0,
        model_step: int = 0,
        progress: ProduceProgress | None = None,
        update_event: asyncio.Event | None = None,
    ) -> ProduceContext:
        # 测试只走新的 ProduceContext 入口，不再覆盖旧散装参数兼容逻辑。
        if progress is None:
            progress = self._build_progress(task_name, target=batch_size, train_step=train_step)
        if update_event is None:
            update_event = asyncio.Event()
        return ProduceContext(
            agent_loop=agent_loop,
            sampler=sampler,
            replay_buffer=self.replay_buffer,
            task_batch_size=batch_size,
            task_name=task_name,
            train_step=train_step,
            update_event=update_event,
            model_step=model_step,
            progress=progress,
            is_valid_sample_fn=strategy.is_valid_sample_fn,
            stale_threshold=getattr(strategy, "stale_threshold", None),
        )

    def test_produce_progress_methods_keep_absolute_window(self):
        progress = ProduceProgress.build(["task_a", "task_b"])

        def allocate(batch_size: int, step: int) -> dict[str, int]:
            self.assertEqual(batch_size, 4)
            return {"task_a": step, "task_b": batch_size - step}

        current_sizes = progress.ensure_target_upto(
            batch_size=4,
            future_step=2,
            allocate_batch_sizes=allocate,
        )

        self.assertEqual(current_sizes, {"task_a": 2, "task_b": 2})
        self.assertEqual(progress.target_samples, {"task_a": 3, "task_b": 5})
        self.assertEqual(progress.target_upto_future_step, 2)

        progress.begin_consume(2)
        progress.mark_consumed({"task_a": 1, "task_b": 2})
        progress.finish_consume(2)
        progress.advance_future_step()
        self.assertEqual(progress.next_consumer_step, 3)
        self.assertEqual(progress.producer_future_step, 2)
        self.assertEqual(progress.consumed_samples, {"task_a": 1, "task_b": 2})

        local_progress = ProduceProgress.build_local(["task_a", "task_b"], {"task_a": 1, "task_b": 3}, 7)
        self.assertEqual(local_progress.target_samples, {"task_a": 1, "task_b": 3})
        self.assertEqual(progress.target_samples, {"task_a": 3, "task_b": 5})

        consumed_ref = progress.consumed_samples
        target_ref = progress.target_samples
        progress.load_state_dict(
            {
                "next_consumer_step": 8,
                "producer_future_step": 9,
                "consumed_samples": {"task_a": 4, "task_b": 5},
                "target_samples": {"task_a": 6, "task_b": 7},
                "target_upto_future_step": 10,
            }
        )
        self.assertIs(progress.consumed_samples, consumed_ref)
        self.assertIs(progress.target_samples, target_ref)
        self.assertEqual(progress.state_dict()["target_samples"], {"task_a": 6, "task_b": 7})

    async def test_pending_tasks_claim_ready_only_once(self):
        pending_tasks = _PendingTasks()

        async def spawn_one():
            async def done():
                return "done"

            return asyncio.create_task(done())

        scheduled = await pending_tasks.schedule_one(
            max_pending=1,
            should_abort=lambda: False,
            spawn_one=spawn_one,
        )
        self.assertTrue(scheduled)
        self.assertEqual(pending_tasks.count(), 1)

        await asyncio.sleep(0)
        claimed = await pending_tasks.claim_ready()
        self.assertEqual(len(claimed), 1)
        self.assertEqual(await pending_tasks.claim_ready(), set())
        self.assertEqual(pending_tasks.count(), 0)

    async def test_pending_tasks_schedule_respects_abort_and_limit(self):
        pending_tasks = _PendingTasks()
        spawn_count = 0

        async def spawn_one():
            nonlocal spawn_count
            spawn_count += 1

            async def wait_forever():
                await asyncio.Event().wait()

            return asyncio.create_task(wait_forever())

        self.assertFalse(
            await pending_tasks.schedule_one(max_pending=0, should_abort=lambda: False, spawn_one=spawn_one)
        )
        self.assertFalse(
            await pending_tasks.schedule_one(max_pending=1, should_abort=lambda: True, spawn_one=spawn_one)
        )
        self.assertTrue(
            await pending_tasks.schedule_one(max_pending=1, should_abort=lambda: False, spawn_one=spawn_one)
        )
        self.assertFalse(
            await pending_tasks.schedule_one(max_pending=1, should_abort=lambda: False, spawn_one=spawn_one)
        )
        self.assertEqual(spawn_count, 1)

        self.assertEqual(await pending_tasks.cancel_all(), 1)
        self.assertEqual(pending_tasks.count(), 0)

    async def test_pending_tasks_claim_all_clears_before_wait_claims(self):
        pending_tasks = _PendingTasks()

        async def spawn_one():
            async def wait_forever():
                await asyncio.Event().wait()

            return asyncio.create_task(wait_forever())

        self.assertTrue(
            await pending_tasks.schedule_one(max_pending=1, should_abort=lambda: False, spawn_one=spawn_one)
        )
        claimed = await pending_tasks.claim_all()
        self.assertEqual(len(claimed), 1)
        self.assertEqual(await pending_tasks.wait_and_claim(timeout_s=0), set())
        self.assertEqual(pending_tasks.count(), 0)
        for task in claimed:
            task.cancel()
        await asyncio.gather(*claimed, return_exceptions=True)

    async def test_sampler_with_replay_buffer(self):
        task_name = "test_task"
        sampler = self._build_sampler()

        # 场景 A: ReplayBuffer 为空，从 Dataloader 拿
        data = await sampler.sample(task_name)
        self.assertEqual(data[0].id, 0)

        # 场景 B: ReplayBuffer 有多个候选状态，按列表顺序优先拿
        aborted_item = MockRolloutState(999, status=Status.ABORTED)
        expired_item = MockRolloutState(1000, status=Status.EXPIRED)
        await self.replay_buffer.put([aborted_item], task_name)
        await self.replay_buffer.put([expired_item], task_name)

        data = await sampler.sample(task_name, group_status=[Status.EXPIRED, Status.ABORTED])
        self.assertEqual(data[0].id, 1000)

        data = await sampler.sample(task_name, group_status=[Status.EXPIRED, Status.ABORTED])
        self.assertEqual(data[0].id, 999)

        # 场景 C: ReplayBuffer 对应状态都为空，回退到 Dataloader
        data = await sampler.sample(task_name, group_status=[Status.EXPIRED, Status.ABORTED])
        self.assertEqual(data[0].id, 1)

    async def test_put_generated_group_only_validates_completed_group(self):
        task_name = "test_valid_completed_only"
        valid_checked_statuses = []

        def is_valid_sample_fn(samples):
            valid_checked_statuses.append([sample.status for sample in samples])
            return False

        strategy = SyncProduceStrategyConfig(is_valid_sample_fn=is_valid_sample_fn).build()
        ctx = self._build_context(
            strategy,
            task_name,
            self._build_agent_loop(),
            self._build_sampler(),
            batch_size=1,
        )

        completed_group = [MockRolloutState(1, status=Status.COMPLETED)]
        self.assertFalse(await ctx.put_generated_group(completed_group))
        self.assertEqual(completed_group[0].status, Status.FILTERED)

        aborted_group = [MockRolloutState(2, status=Status.ABORTED)]
        self.assertFalse(await ctx.put_generated_group(aborted_group))
        self.assertEqual(aborted_group[0].status, Status.ABORTED)

        self.assertEqual(valid_checked_statuses, [[Status.COMPLETED]])
        self.assertEqual(await self.replay_buffer.count(task_name, Status.FILTERED), 1)
        self.assertEqual(await self.replay_buffer.count(task_name, Status.ABORTED), 1)

    async def test_sync_produce_strategy(self):
        task_name = "test_task"
        mock_agent_loop = self._build_agent_loop({0: 0.0, 1: 0.01})
        produce_strategy_cfg = SyncProduceStrategyConfig()
        sampler = self._build_sampler()
        strategy = produce_strategy_cfg.build()

        # 执行：生产 batch_size 为 2 的数据
        ctx = self._build_context(
            strategy,
            task_name,
            mock_agent_loop,
            sampler,
            batch_size=2,
            train_step=4,
            model_step=3,
            progress=self._build_progress(task_name, target=2, train_step=4),
        )
        status = await strategy.produce_batch(ctx)
        self.assertEqual(status, ProduceBatchStatus.NORMAL)

        # 验证：ReplayBuffer 中应该有 2 条 COMPLETED 数据
        final_data = await self.replay_buffer.get(10, task_name, Status.COMPLETED)
        print(final_data[0][0].id, final_data[0][0].status)
        print(final_data[1][0].id, final_data[1][0].status)
        self.assertEqual(len(final_data), 2)
        self.assertEqual(final_data[0][0].id, 0)
        self.assertEqual(final_data[1][0].id, 1)

    async def test_async_produce_strategy(self):
        # 这个async_produce_strategy的测试主要验证超发逻辑 + staleness 优先get的逻辑
        # 异步的其他功能如 partial_rollout, tail_batch不在这里进行验证
        mock_agent_loop = MagicMock()
        mock_agent_loop.pause = AsyncMock()
        mock_agent_loop.rollout_ctl.continue_generation.remote = AsyncMock(return_value=None)
        task_name = "test_task"
        call_count = 0
        async def mock_gen(rs, **kwargs):
            nonlocal call_count
            call_count += 1
            for r in rs:
                if r.id == 999:
                    r.seq_staleness = 5
                else:
                    r.seq_staleness = call_count
                r.status = Status.COMPLETED
                print(r.id, r.seq_staleness, r.status)
            return rs

        mock_agent_loop.generate_group = mock_gen

        sampler_cfg = SamplerConfig.model_construct(dataloader_cfg=self.mock_dataloader_cfg)
        produce_strategy_cfg = AsyncProduceStrategyConfig(over_sample_threshold= 1)
        sampler = sampler_cfg.build(self.mock_tokenizer, self.replay_buffer)
        strategy = produce_strategy_cfg.build()
        # 预处理
        aborted_item = MockRolloutState(999, status=Status.ABORTED)
        await self.replay_buffer.put([aborted_item], task_name)
        # 执行
        ctx = self._build_context(
            strategy,
            task_name,
            mock_agent_loop,
            sampler,
            batch_size=2,
            model_step=0,
            progress=self._build_progress(task_name, target=2),
        )
        status = await strategy.produce_batch(ctx)
        self.assertEqual(status, ProduceBatchStatus.NORMAL)

        # 验证：ReplayBuffer 中应该有 4 条 COMPLETED 数据。
        final_data = await self.replay_buffer.get(10, task_name, Status.COMPLETED)
        self.assertEqual(len(final_data), 4)
        self.assertEqual(sorted(group[0].id for group in final_data), [0, 1, 2, 999])

    async def test_async_produce_strategy_accepts_context_entrypoint(self):
        task_name = "test_context_entry"
        mock_agent_loop = self._build_agent_loop()
        sampler = self._build_sampler()
        strategy = AsyncProduceStrategyConfig(over_sample_threshold=0.0).build()
        progress = self._build_progress(task_name, target=1, train_step=1)
        ctx = self._build_context(
            strategy,
            task_name,
            mock_agent_loop,
            sampler,
            batch_size=1,
            train_step=1,
            progress=progress,
        )

        status = await strategy.produce_batch(ctx)

        self.assertEqual(status, ProduceBatchStatus.NORMAL)
        self.assertEqual(await self.replay_buffer.count(task_name, Status.COMPLETED), 1)

    async def test_async_produce_strategy_uses_live_consumed_progress(self):
        task_name = "test_live_consumed"
        call_count = 0

        async def mock_gen(rs, **kwargs):
            nonlocal call_count
            call_count += 1
            for r in rs:
                r.status = Status.COMPLETED
            return rs

        mock_agent_loop = self._build_agent_loop()
        mock_agent_loop.generate_group = mock_gen
        sampler = self._build_sampler()
        # 该用例验证版本记录顺序，放宽 stale 策略避免在生产入口提前返回。
        strategy = AsyncProduceStrategyConfig(over_sample_threshold=0.0, max_staleness=3).build()
        progress = ProduceProgress(
            next_consumer_step=1,
            producer_future_step=2,
            consumed_samples={task_name: 1},
            target_samples={task_name: 2},
            target_upto_future_step=2,
        )

        ctx = self._build_context(
            strategy,
            task_name,
            mock_agent_loop,
            sampler,
            batch_size=1,
            train_step=2,
            model_step=1,
            progress=progress,
        )
        status = await strategy.produce_batch(ctx)

        self.assertEqual(status, ProduceBatchStatus.NORMAL)
        self.assertEqual(call_count, 1)
        self.assertEqual(await self.replay_buffer.count(task_name, Status.COMPLETED), 1)

    async def test_async_produce_strategy_uses_fixed_batch_oversample_budget(self):
        task_name = "test_fixed_oversample"
        sampler = MagicMock()
        sample_ids = iter(range(100, 200))

        async def sample(task_name, group_status=None):
            self.assertEqual(group_status, [Status.ABORTED])
            return [MockRolloutState(next(sample_ids), status=Status.ABORTED)]

        sampler.sample = AsyncMock(side_effect=sample)
        mock_agent_loop = self._build_agent_loop()
        strategy = AsyncProduceStrategyConfig(over_sample_threshold=1.0).build()
        progress = self._build_progress(task_name, target=10, consumed=9)

        ctx = self._build_context(
            strategy,
            task_name,
            mock_agent_loop,
            sampler,
            batch_size=4,
            model_step=0,
            progress=progress,
        )
        status = await strategy.produce_batch(ctx)

        self.assertEqual(status, ProduceBatchStatus.NORMAL)
        # 当前只缺 1 个样本，但 over-sample 预算固定为 over * batch_size = 4，
        # 因此本轮最多调度到 target + 4，对应初始发射 5 个任务。
        self.assertEqual(sampler.sample.await_count, 5)
        self.assertEqual(await self.replay_buffer.count(task_name, Status.COMPLETED), 5)

    async def test_async_produce_strategy_tail_batch_is_static_and_no_oversample(self):
        task_name = "test_tail_static"
        for sample_id in (900, 901):
            await self.replay_buffer.put([MockRolloutState(sample_id, status=Status.EXPIRED)], task_name)

        sampler = self._build_sampler()
        original_sample = sampler.sample
        sampled_statuses: list[list[Status] | None] = []

        async def instrumented_sample(task_name, group_status=None):
            sampled_statuses.append(group_status)
            return await original_sample(task_name=task_name, group_status=group_status)

        sampler.sample = instrumented_sample
        mock_agent_loop = self._build_agent_loop()
        strategy = AsyncProduceStrategyConfig(
            over_sample_threshold=1.0,
            tail_batch_trigger_size=1,
        ).build()

        ctx = self._build_context(
            strategy,
            task_name,
            mock_agent_loop,
            sampler,
            batch_size=2,
            model_step=0,
            progress=self._build_progress(task_name, target=2),
        )
        status = await strategy.produce_batch(ctx)

        self.assertEqual(status, ProduceBatchStatus.NORMAL)
        # tail-batch 模式在本轮优先走 EXPIRED pool，并且不使用 over-sample 额外发射。
        self.assertEqual(sampled_statuses, [[Status.EXPIRED, Status.ABORTED], [Status.EXPIRED, Status.ABORTED]])
        completed = await self.replay_buffer.get(10, task_name, Status.COMPLETED)
        self.assertEqual(sorted(group[0].id for group in completed), [900, 901])

    async def test_async_produce_strategy_fails_fast_on_invalid_progress(self):
        task_name = "test_invalid_progress"
        strategy = AsyncProduceStrategyConfig(over_sample_threshold=0.0).build()
        mock_agent_loop = self._build_agent_loop()
        sampler = MagicMock()
        sampler.sample = AsyncMock(side_effect=AssertionError("sampler.sample should not be called"))

        missing_consumed = ProduceProgress(
            next_consumer_step=1,
            producer_future_step=1,
            consumed_samples={},
            target_samples={task_name: 1},
            target_upto_future_step=1,
        )
        with self.assertRaisesRegex(KeyError, "consumed_samples"):
            ctx = self._build_context(
                strategy,
                task_name,
                mock_agent_loop,
                sampler,
                batch_size=1,
                train_step=1,
                model_step=0,
                progress=missing_consumed,
            )
            await strategy.produce_batch(ctx)

        missing_target = ProduceProgress(
            next_consumer_step=1,
            producer_future_step=1,
            consumed_samples={task_name: 0},
            target_samples={},
            target_upto_future_step=1,
        )
        with self.assertRaisesRegex(KeyError, "target_samples"):
            ctx = self._build_context(
                strategy,
                task_name,
                mock_agent_loop,
                sampler,
                batch_size=1,
                train_step=1,
                model_step=0,
                progress=missing_target,
            )
            await strategy.produce_batch(ctx)

    async def test_async_produce_strategy_records_sample_version_before_staleness_refresh(self):
        task_name = "test_sample_version"

        async def mock_gen(rs, **kwargs):
            self.assertNotIn("model_step", kwargs)
            for r in rs:
                r.response_ids = [10, 11]
                r.status = Status.COMPLETED
            return rs

        mock_agent_loop = self._build_agent_loop()
        mock_agent_loop.generate_group = mock_gen
        sampler = self._build_sampler()
        # 该用例验证版本记录顺序，放宽 stale 策略避免在生产入口提前返回。
        strategy = AsyncProduceStrategyConfig(over_sample_threshold=0.0, max_staleness=3).build()

        ctx = self._build_context(
            strategy,
            task_name,
            mock_agent_loop,
            sampler,
            batch_size=1,
            train_step=5,
            model_step=3,
            progress=self._build_progress(task_name, target=1, train_step=5),
        )
        status = await strategy.produce_batch(ctx)

        self.assertEqual(status, ProduceBatchStatus.NORMAL)
        completed = await self.replay_buffer.get(1, task_name, Status.COMPLETED)
        self.assertEqual(completed[0][0].response_model_steps, [3, 3])
        self.assertEqual(completed[0][0].seq_staleness, 1)

    async def test_async_produce_strategy_preserves_partial_rollout_old_versions(self):
        task_name = "test_partial_rollout_versions"
        partial_item = MockRolloutState(700, status=Status.ABORTED)
        partial_item.response_ids = [10]
        partial_item.response_model_steps = [1]
        await self.replay_buffer.put([partial_item], task_name)

        async def mock_gen(rs, **kwargs):
            self.assertNotIn("model_step", kwargs)
            # partial rollout 的历史 token 已有版本，新 token 应按本次调度时的模型版本补齐。
            rs[0].response_ids = [10, 11, 12]
            rs[0].status = Status.COMPLETED
            return rs

        mock_agent_loop = self._build_agent_loop()
        mock_agent_loop.generate_group = mock_gen
        sampler = self._build_sampler()
        # 该用例验证 partial rollout 版本拼接，放宽 stale 策略保留旧分段。
        strategy = AsyncProduceStrategyConfig(
            over_sample_threshold=0.0,
            enable_partial_rollout=True,
            max_staleness=3,
        ).build()

        ctx = self._build_context(
            strategy,
            task_name,
            mock_agent_loop,
            sampler,
            batch_size=1,
            train_step=5,
            model_step=3,
            progress=self._build_progress(task_name, target=1, train_step=5),
        )
        status = await strategy.produce_batch(ctx)

        self.assertEqual(status, ProduceBatchStatus.NORMAL)
        completed = await self.replay_buffer.get(1, task_name, Status.COMPLETED)
        self.assertEqual(completed[0][0].response_model_steps, [1, 3, 3])
        self.assertEqual(completed[0][0].seq_staleness, 3)

    async def test_async_produce_strategy_reclaims_cross_call_pending_and_records_timing(self):
        task_name = "test_task"
        mock_agent_loop = self._build_agent_loop({0: 0.01, 1: 0.05, 2: 0.05})
        produce_strategy_cfg = AsyncProduceStrategyConfig(over_sample_threshold=2.0, enable_partial_rollout=True)
        sampler = self._build_sampler()
        strategy = produce_strategy_cfg.build()
        progress = self._build_progress(task_name, target=1)

        ctx = self._build_context(
            strategy,
            task_name,
            mock_agent_loop,
            sampler,
            batch_size=1,
            model_step=0,
            progress=progress,
        )
        status = await strategy.produce_batch(ctx)
        self.assertEqual(status, ProduceBatchStatus.NORMAL)
        self.assertGreater(strategy.pending_task_count(), 0)

        await asyncio.sleep(0.08)

        ctx = self._build_context(
            strategy,
            task_name,
            mock_agent_loop,
            sampler,
            batch_size=1,
            model_step=0,
            progress=progress,
        )
        status = await strategy.produce_batch(ctx)
        self.assertEqual(status, ProduceBatchStatus.NORMAL)
        self.assertEqual(strategy.pending_task_count(), 0)

        final_data = await self.replay_buffer.get(10, task_name, Status.COMPLETED)
        self.assertEqual(len(final_data), 3)
        self.assertEqual(sorted(group[0].id for group in final_data), [0, 1, 2])
        for group in final_data:
            self.assertIn("group_generate_time_s", group[0].extra_fields)
            self.assertGreater(group[0].extra_fields["group_generate_time_s"], 0.0)

    async def test_async_produce_strategy_pause_produce_is_explicit(self):
        task_name = "test_cleanup"
        mock_agent_loop = self._build_agent_loop({0: 0.01, 1: 0.2, 2: 0.2})
        produce_strategy_cfg = AsyncProduceStrategyConfig(over_sample_threshold=2.0, enable_partial_rollout=True)
        sampler = self._build_sampler()
        strategy = produce_strategy_cfg.build()
        progress = self._build_progress(task_name, target=1)

        ctx = self._build_context(
            strategy,
            task_name,
            mock_agent_loop,
            sampler,
            batch_size=1,
            model_step=0,
            progress=progress,
        )
        await strategy.produce_batch(ctx)
        self.assertGreater(strategy.pending_task_count(), 0)

        pause_time_s = await strategy.pause_produce(ctx)

        self.assertGreaterEqual(pause_time_s, 0.0)
        self.assertEqual(strategy.pending_task_count(), 0)
        completed = await self.replay_buffer.count(task_name, Status.COMPLETED)
        aborted = await self.replay_buffer.count(task_name, Status.ABORTED)
        expired = await self.replay_buffer.count(task_name, Status.EXPIRED)
        self.assertEqual(completed + aborted + expired, 3)

    async def test_async_produce_strategy_pause_produce_cancels_all_on_timeout(self):
        task_name = "test_cleanup_timeout"
        mock_agent_loop = self._build_agent_loop({0: 0.01, 1: 60.0, 2: 60.0})
        produce_strategy_cfg = AsyncProduceStrategyConfig(over_sample_threshold=2.0, enable_partial_rollout=True)
        sampler = self._build_sampler()
        strategy = produce_strategy_cfg.build()
        strategy.cleanup_task_time = 0
        progress = self._build_progress(task_name, target=1)

        ctx = self._build_context(
            strategy,
            task_name,
            mock_agent_loop,
            sampler,
            batch_size=1,
            model_step=0,
            progress=progress,
        )
        await strategy.produce_batch(ctx)
        self.assertGreater(strategy.pending_task_count(), 0)

        await strategy.pause_produce(ctx)

        self.assertEqual(strategy.pending_task_count(), 0)
        completed = await self.replay_buffer.count(task_name, Status.COMPLETED)
        aborted = await self.replay_buffer.count(task_name, Status.ABORTED)
        expired = await self.replay_buffer.count(task_name, Status.EXPIRED)
        self.assertEqual(completed + aborted + expired, 1)

    async def test_async_produce_strategy_returns_update_abort_without_sampling(self):
        task_name = "test_update_abort"
        strategy = AsyncProduceStrategyConfig(over_sample_threshold=1.0).build()
        mock_agent_loop = self._build_agent_loop()
        sampler = MagicMock()
        sampler.sample = AsyncMock(side_effect=AssertionError("sampler.sample should not be called"))
        update_event = asyncio.Event()
        update_event.set()

        ctx = self._build_context(
            strategy,
            task_name,
            mock_agent_loop,
            sampler,
            batch_size=1,
            train_step=1,
            model_step=1,
            update_event=update_event,
            progress=self._build_progress(task_name, target=1, train_step=1),
        )
        status = await strategy.produce_batch(ctx)

        self.assertEqual(status, ProduceBatchStatus.UPDATE_ABORT)
        self.assertEqual(await self.replay_buffer.count(task_name, Status.COMPLETED), 0)

    async def test_async_produce_strategy_returns_update_abort_after_schedule_pause(self):
        task_name = "test_update_abort_after_schedule"
        strategy = AsyncProduceStrategyConfig(over_sample_threshold=0.0).build()
        mock_agent_loop = self._build_agent_loop({0: 0.05})
        sampler = MagicMock()
        update_event = asyncio.Event()
        progress = self._build_progress(task_name, target=1)

        async def sample(task_name, group_status=None):
            # 模拟 manager 在调度临界区中途触发 pause；当前样本会进入 pending，后续应停止继续调度。
            update_event.set()
            return [MockRolloutState(0, status=Status.ABORTED)]

        sampler.sample = AsyncMock(side_effect=sample)

        ctx = self._build_context(
            strategy,
            task_name,
            mock_agent_loop,
            sampler,
            batch_size=1,
            update_event=update_event,
            model_step=0,
            progress=progress,
        )
        status = await strategy.produce_batch(ctx)

        self.assertEqual(status, ProduceBatchStatus.UPDATE_ABORT)
        self.assertEqual(sampler.sample.await_count, 1)

        await strategy.pause_produce(ctx)
        self.assertEqual(strategy.pending_task_count(), 0)

    async def test_async_produce_strategy_returns_expired_batch_before_processing_leftovers(self):
        task_name = "test_expired_batch"
        strategy = AsyncProduceStrategyConfig(max_staleness=0).build()
        mock_agent_loop = self._build_agent_loop()
        sampler = MagicMock()
        sampler.sample = AsyncMock(side_effect=AssertionError("sampler.sample should not be called"))
        await self.replay_buffer.put([MockRolloutState(999, status=Status.COMPLETED)], task_name)

        ctx = self._build_context(
            strategy,
            task_name,
            mock_agent_loop,
            sampler,
            batch_size=1,
            train_step=3,
            model_step=1,
            progress=self._build_progress(task_name, target=1, train_step=3),
        )
        status = await strategy.produce_batch(ctx)

        self.assertEqual(status, ProduceBatchStatus.EXPIRED_BATCH)
        self.assertEqual(await self.replay_buffer.count(task_name, Status.COMPLETED), 1)
        self.assertEqual(await self.replay_buffer.count(task_name, Status.ABORTED), 0)

    async def test_refresh_staleness_refreshes_before_expire_check(self):
        task_name = "test_refresh_leftover"
        stale_item = MockRolloutState(1000, seq_staleness=0, status=Status.COMPLETED)
        stale_item.response_model_steps = [3]
        await self.replay_buffer.put([stale_item], task_name)

        expired_counts = await self.replay_buffer.refresh_staleness(
            task_stale_thresholds={task_name: 2},
            current_train_step=6,
        )
        expired_groups = await self.replay_buffer.get(10, task_name, Status.EXPIRED)

        self.assertEqual(expired_counts, {task_name: 1})
        self.assertEqual(len(expired_groups), 1)
        self.assertEqual(expired_groups[0][0].seq_staleness, 2)
