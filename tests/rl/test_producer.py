import unittest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from xtuner.v1.rl.agent_loop import (
    AsyncProduceStrategyConfig,
    ProduceBatchStatus,
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

    def _build_agent_loop(self, sleep_by_id: dict[int, float] | None = None):
        mock_agent_loop = MagicMock()
        mock_agent_loop.rollout_ctl.continue_generation.remote = AsyncMock(return_value=None)
        mock_agent_loop.rollout_ctl.pause_generation.remote = AsyncMock(return_value=None)
        mock_agent_loop.rollout_ctl.get_rollout_metadata.remote = AsyncMock(return_value={"server_url_dict": {}})

        sleep_by_id = sleep_by_id or {}

        async def mock_gen(rs, **kwargs):
            await asyncio.sleep(sleep_by_id.get(rs[0].id, 0.0))
            for r in rs:
                r.seq_staleness = kwargs.get("model_rollout_step", kwargs.get("rollout_step", 0))
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

        # 场景 B: ReplayBuffer 有 ABORTED 数据，优先拿
        aborted_item = MockRolloutState(999, status=Status.ABORTED)
        await self.replay_buffer.put([aborted_item], task_name)
        
        data = await sampler.sample(task_name, group_status=Status.ABORTED)
        self.assertEqual(data[0].id, 999)

    async def test_sync_produce_strategy(self):
        task_name = "test_task"
        mock_agent_loop = self._build_agent_loop({0: 0.0, 1: 0.01})
        produce_strategy_cfg = SyncProduceStrategyConfig()
        sampler = self._build_sampler()
        strategy = produce_strategy_cfg.build()

        # 执行：生产 batch_size 为 2 的数据
        await strategy.produce_batch(mock_agent_loop, sampler, self.replay_buffer, batch_size=2, task_name=task_name)

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
        await strategy.produce_batch(mock_agent_loop, sampler, self.replay_buffer, batch_size=2, task_name=task_name)

        # 验证：ReplayBuffer 中应该有 4 条 COMPLETED 数据，
        # NOTE(@duanyanhui): 目前还没实现暂停功能，所以4条都会推理完成,4条数据按照新鲜度顺序排列，999 是最旧的，0 是最新的
        final_data = await self.replay_buffer.get(10, task_name, Status.COMPLETED)
        self.assertEqual(len(final_data), 4)
        self.assertEqual(final_data[0][0].id, 999)
        self.assertEqual(final_data[1][0].id, 2)
        self.assertEqual(final_data[2][0].id, 1)
        self.assertEqual(final_data[3][0].id, 0)

    async def test_async_produce_strategy_reclaims_cross_call_pending_and_records_timing(self):
        task_name = "test_task"
        mock_agent_loop = self._build_agent_loop({0: 0.01, 1: 0.05, 2: 0.05})
        produce_strategy_cfg = AsyncProduceStrategyConfig(over_sample_threshold=2.0, enable_partial_rollout=True)
        sampler = self._build_sampler()
        strategy = produce_strategy_cfg.build()

        status = await strategy.produce_batch(
            mock_agent_loop, sampler, self.replay_buffer, batch_size=1, task_name=task_name
        )
        self.assertEqual(status, ProduceBatchStatus.NORMAL)
        self.assertGreater(len(strategy._pending_tasks), 0)

        await asyncio.sleep(0.08)

        status = await strategy.produce_batch(
            mock_agent_loop, sampler, self.replay_buffer, batch_size=1, task_name=task_name
        )
        self.assertEqual(status, ProduceBatchStatus.NORMAL)
        self.assertEqual(len(strategy._pending_tasks), 0)

        final_data = await self.replay_buffer.get(10, task_name, Status.COMPLETED)
        self.assertEqual(len(final_data), 3)
        self.assertEqual(sorted(group[0].id for group in final_data), [0, 1, 2])
        for group in final_data:
            self.assertIn("group_generate_time_s", group[0].extra_fields)
            self.assertGreater(group[0].extra_fields["group_generate_time_s"], 0.0)

    async def test_async_produce_strategy_cleanup_pending_tasks_is_explicit(self):
        task_name = "test_cleanup"
        mock_agent_loop = self._build_agent_loop({0: 0.01, 1: 0.2, 2: 0.2})
        produce_strategy_cfg = AsyncProduceStrategyConfig(over_sample_threshold=2.0, enable_partial_rollout=True)
        sampler = self._build_sampler()
        strategy = produce_strategy_cfg.build()

        await strategy.produce_batch(mock_agent_loop, sampler, self.replay_buffer, batch_size=1, task_name=task_name)
        self.assertGreater(len(strategy._pending_tasks), 0)

        pause_time_s = await strategy.cleanup_pending_tasks(mock_agent_loop, self.replay_buffer, task_name)

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
            rollout_step=1,
            model_rollout_step=1,
            update_event=update_event,
        )

        self.assertEqual(status, ProduceBatchStatus.UPDATE_ABORT)
        self.assertEqual(await self.replay_buffer.count(task_name, Status.COMPLETED), 0)

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
            rollout_step=2,
            model_rollout_step=1,
        )

        self.assertEqual(status, ProduceBatchStatus.EXPIRED_BATCH)
        self.assertEqual(await self.replay_buffer.count(task_name, Status.COMPLETED), 1)
        self.assertEqual(await self.replay_buffer.count(task_name, Status.ABORTED), 0)
