import unittest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from xtuner.v1.rl.agent_loop import (
    AsyncProduceStrategyConfig,
    ProduceBatchStatus,
    ProduceProgress,
    SamplerConfig,
    SyncProduceStrategyConfig,
)
from xtuner.v1.rl.replay_buffer import AsyncReplayBufferConfig
from xtuner.v1.data_proto.rl_data import Status


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

    async def test_sync_produce_strategy(self):
        task_name = "test_task"
        mock_agent_loop = self._build_agent_loop({0: 0.0, 1: 0.01})
        produce_strategy_cfg = SyncProduceStrategyConfig()
        sampler = self._build_sampler()
        strategy = produce_strategy_cfg.build()

        # 执行：生产 batch_size 为 2 的数据
        status = await strategy.produce_batch(
            mock_agent_loop,
            sampler,
            self.replay_buffer,
            batch_size=2,
            task_name=task_name,
            train_step=4,
            model_step=3,
            progress=self._build_progress(task_name, target=2, train_step=4),
        )
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
        status = await strategy.produce_batch(
            mock_agent_loop,
            sampler,
            self.replay_buffer,
            batch_size=2,
            task_name=task_name,
            model_step=0,
            progress=self._build_progress(task_name, target=2),
        )
        self.assertEqual(status, ProduceBatchStatus.NORMAL)

        # 验证：ReplayBuffer 中应该有 4 条 COMPLETED 数据。
        final_data = await self.replay_buffer.get(10, task_name, Status.COMPLETED)
        self.assertEqual(len(final_data), 4)
        self.assertEqual(sorted(group[0].id for group in final_data), [0, 1, 2, 999])

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
        strategy = AsyncProduceStrategyConfig(over_sample_threshold=0.0).build()
        progress = ProduceProgress(
            next_consumer_step=1,
            producer_future_step=2,
            consumed_samples={task_name: 1},
            target_samples={task_name: 2},
            target_upto_future_step=2,
        )

        status = await strategy.produce_batch(
            mock_agent_loop,
            sampler,
            self.replay_buffer,
            batch_size=1,
            task_name=task_name,
            train_step=2,
            model_step=1,
            target_cumulative=2,
            progress=progress,
        )

        self.assertEqual(status, ProduceBatchStatus.NORMAL)
        self.assertEqual(call_count, 1)
        self.assertEqual(await self.replay_buffer.count(task_name, Status.COMPLETED), 1)

    async def test_async_produce_strategy_uses_fixed_batch_oversample_budget(self):
        task_name = "test_fixed_oversample"
        sampler = MagicMock()
        sample_ids = iter(range(100, 200))

        async def sample(task_name, group_status=None):
            self.assertEqual(group_status, Status.ABORTED)
            return [MockRolloutState(next(sample_ids), status=Status.ABORTED)]

        sampler.sample = AsyncMock(side_effect=sample)
        mock_agent_loop = self._build_agent_loop()
        strategy = AsyncProduceStrategyConfig(over_sample_threshold=1.0).build()
        progress = self._build_progress(task_name, target=10, consumed=9)

        status = await strategy.produce_batch(
            mock_agent_loop,
            sampler,
            self.replay_buffer,
            batch_size=4,
            task_name=task_name,
            model_step=0,
            progress=progress,
        )

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

        status = await strategy.produce_batch(
            mock_agent_loop,
            sampler,
            self.replay_buffer,
            batch_size=2,
            task_name=task_name,
            model_step=0,
            progress=self._build_progress(task_name, target=2),
        )

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
            await strategy.produce_batch(
                mock_agent_loop,
                sampler,
                self.replay_buffer,
                batch_size=1,
                task_name=task_name,
                train_step=1,
                model_step=0,
                target_cumulative=1,
                progress=missing_consumed,
            )

        mismatched_target = ProduceProgress(
            next_consumer_step=1,
            producer_future_step=1,
            consumed_samples={task_name: 0},
            target_samples={task_name: 2},
            target_upto_future_step=1,
        )
        with self.assertRaisesRegex(ValueError, "target_cumulative"):
            await strategy.produce_batch(
                mock_agent_loop,
                sampler,
                self.replay_buffer,
                batch_size=1,
                task_name=task_name,
                train_step=1,
                model_step=0,
                target_cumulative=1,
                progress=mismatched_target,
            )

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
        strategy = AsyncProduceStrategyConfig(over_sample_threshold=0.0).build()

        status = await strategy.produce_batch(
            mock_agent_loop,
            sampler,
            self.replay_buffer,
            batch_size=1,
            task_name=task_name,
            train_step=5,
            model_step=3,
            progress=self._build_progress(task_name, target=1, train_step=5),
        )

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
        strategy = AsyncProduceStrategyConfig(over_sample_threshold=0.0, enable_partial_rollout=True).build()

        status = await strategy.produce_batch(
            mock_agent_loop,
            sampler,
            self.replay_buffer,
            batch_size=1,
            task_name=task_name,
            train_step=5,
            model_step=3,
            progress=self._build_progress(task_name, target=1, train_step=5),
        )

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

        status = await strategy.produce_batch(
            mock_agent_loop,
            sampler,
            self.replay_buffer,
            batch_size=1,
            task_name=task_name,
            model_step=0,
            progress=progress,
        )
        self.assertEqual(status, ProduceBatchStatus.NORMAL)
        self.assertGreater(len(strategy._pending_tasks), 0)

        await asyncio.sleep(0.08)

        status = await strategy.produce_batch(
            mock_agent_loop,
            sampler,
            self.replay_buffer,
            batch_size=1,
            task_name=task_name,
            model_step=0,
            progress=progress,
        )
        self.assertEqual(status, ProduceBatchStatus.NORMAL)
        self.assertEqual(len(strategy._pending_tasks), 0)

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

        await strategy.produce_batch(
            mock_agent_loop,
            sampler,
            self.replay_buffer,
            batch_size=1,
            task_name=task_name,
            model_step=0,
            progress=progress,
        )
        self.assertGreater(len(strategy._pending_tasks), 0)

        pause_time_s = await strategy.pause_produce(
            mock_agent_loop,
            self.replay_buffer,
            task_name,
            progress=progress,
        )

        self.assertGreaterEqual(pause_time_s, 0.0)
        self.assertEqual(len(strategy._pending_tasks), 0)
        completed = await self.replay_buffer.count(task_name, Status.COMPLETED)
        aborted = await self.replay_buffer.count(task_name, Status.ABORTED)
        expired = await self.replay_buffer.count(task_name, Status.EXPIRED)
        self.assertEqual(completed + aborted + expired, 3)

    async def test_async_produce_strategy_returns_update_abort_without_sampling(self):
        task_name = "test_update_abort"
        strategy = AsyncProduceStrategyConfig(over_sample_threshold=1.0).build()
        mock_agent_loop = self._build_agent_loop()
        sampler = MagicMock()
        sampler.sample = AsyncMock(side_effect=AssertionError("sampler.sample should not be called"))
        update_event = asyncio.Event()
        update_event.set()

        status = await strategy.produce_batch(
            mock_agent_loop,
            sampler,
            self.replay_buffer,
            batch_size=1,
            task_name=task_name,
            train_step=1,
            model_step=1,
            update_event=update_event,
            progress=self._build_progress(task_name, target=1, train_step=1),
        )

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

        status = await strategy.produce_batch(
            mock_agent_loop,
            sampler,
            self.replay_buffer,
            batch_size=1,
            task_name=task_name,
            update_event=update_event,
            model_step=0,
            progress=progress,
        )

        self.assertEqual(status, ProduceBatchStatus.UPDATE_ABORT)
        self.assertEqual(sampler.sample.await_count, 1)

        await strategy.pause_produce(
            mock_agent_loop,
            self.replay_buffer,
            task_name,
            progress=progress,
        )
        self.assertEqual(len(strategy._pending_tasks), 0)

    async def test_async_produce_strategy_returns_expired_batch_before_processing_leftovers(self):
        task_name = "test_expired_batch"
        strategy = AsyncProduceStrategyConfig(tail_batch_stale_threshold=1).build()
        mock_agent_loop = self._build_agent_loop()
        sampler = MagicMock()
        sampler.sample = AsyncMock(side_effect=AssertionError("sampler.sample should not be called"))
        await self.replay_buffer.put([MockRolloutState(999, status=Status.COMPLETED)], task_name)

        status = await strategy.produce_batch(
            mock_agent_loop,
            sampler,
            self.replay_buffer,
            batch_size=1,
            task_name=task_name,
            train_step=3,
            model_step=1,
            progress=self._build_progress(task_name, target=1, train_step=3),
        )

        self.assertEqual(status, ProduceBatchStatus.EXPIRED_BATCH)
        self.assertEqual(await self.replay_buffer.count(task_name, Status.COMPLETED), 1)
        self.assertEqual(await self.replay_buffer.count(task_name, Status.ABORTED), 0)

    async def test_refresh_completed_samples_refreshes_staleness_before_expire_check(self):
        task_name = "test_refresh_leftover"
        stale_item = MockRolloutState(1000, seq_staleness=0, status=Status.COMPLETED)
        stale_item.response_model_steps = [3]
        await self.replay_buffer.put([stale_item], task_name)

        expired_count = await self.replay_buffer.refresh_completed_staleness(
            task_name=task_name,
            current_train_step=6,
            tail_batch_stale_threshold=2,
        )
        expired_groups = await self.replay_buffer.get(10, task_name, Status.EXPIRED)

        self.assertEqual(expired_count, 1)
        self.assertEqual(len(expired_groups), 1)
        self.assertEqual(expired_groups[0][0].seq_staleness, 2)
