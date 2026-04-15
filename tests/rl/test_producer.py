import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from xtuner.v1.rl.agent_loop import SamplerConfig, SyncProduceStrategyConfig, AsyncProduceStrategyConfig
from xtuner.v1.rl.replay_buffer import AsyncReplayBufferConfig
from xtuner.v1.data_proto.rl_data import RolloutState, Status

class MockRolloutState:
    def __init__(self, id, seq_staleness=1, status=Status.COMPLETED):
        self.id = id
        self.uid = id
        self.status = status
        self.seq_staleness = seq_staleness
        self.response_ids = []

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

    async def test_sampler_with_replay_buffer(self):
        task_name = "test_task"
        sampler_cfg = SamplerConfig.model_construct(dataloader_cfg=self.mock_dataloader_cfg)
        sampler = sampler_cfg.build(self.mock_tokenizer, self.replay_buffer)

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
        mock_agent_loop = MagicMock()
        mock_agent_loop.rollout_ctl.continue_generation.remote = AsyncMock(return_value=None)
        mock_agent_loop.rollout_ctl.pause_generation.remote = AsyncMock(return_value=None)
        mock_agent_loop.rollout_ctl.get_rollout_metadata.remote = AsyncMock(return_value={"server_url_dict": {}})

        async def mock_gen(rs, **kwargs):
            await asyncio.sleep(0.01 * rs[0].id) 
            for r in rs:
                r.status = Status.COMPLETED
            return rs
        mock_agent_loop.generate_group = mock_gen

        sampler_cfg = SamplerConfig.model_construct(dataloader_cfg=self.mock_dataloader_cfg)
        produce_strategy_cfg = SyncProduceStrategyConfig()

        sampler = sampler_cfg.build(self.mock_tokenizer, self.replay_buffer)
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

    async def test_produce_batch_keeps_extra_completed_leftovers(self):
        task_name = "test_task"
        mock_agent_loop = MagicMock()
        mock_agent_loop.rollout_ctl.continue_generation.remote = AsyncMock(return_value=None)
        mock_agent_loop.rollout_ctl.pause_generation.remote = AsyncMock(return_value=None)
        mock_agent_loop.rollout_ctl.get_rollout_metadata.remote = AsyncMock(return_value={"server_url_dict": {}})

        async def mock_gen(rs, **kwargs):
            await asyncio.sleep(0.01 * rs[0].id)
            for r in rs:
                r.status = Status.COMPLETED
            return rs

        mock_agent_loop.generate_group = mock_gen

        sampler_cfg = SamplerConfig.model_construct(dataloader_cfg=self.mock_dataloader_cfg)
        produce_strategy_cfg = AsyncProduceStrategyConfig(over_sample_threshold=1)
        sampler = sampler_cfg.build(self.mock_tokenizer, self.replay_buffer)
        strategy = produce_strategy_cfg.build()

        await strategy.produce_batch(
            mock_agent_loop,
            sampler,
            self.replay_buffer,
            batch_size=2,
            task_name=task_name,
        )

        completed_groups = await self.replay_buffer.get(10, task_name, Status.COMPLETED)
        self.assertGreaterEqual(len(completed_groups), 4)
        self.assertEqual(sorted(group[0].id for group in completed_groups), list(range(len(completed_groups))))

    async def test_produce_batch_passes_partial_rollout_flag(self):
        task_name = "test_task"
        mock_agent_loop = MagicMock()
        mock_agent_loop.rollout_ctl.continue_generation.remote = AsyncMock(return_value=None)
        mock_agent_loop.rollout_ctl.pause_generation.remote = AsyncMock(return_value=None)
        mock_agent_loop.rollout_ctl.get_rollout_metadata.remote = AsyncMock(return_value={"server_url_dict": {}})

        seen_flags = []

        async def mock_gen(rs, **kwargs):
            seen_flags.append(kwargs.get("enable_partial_rollout"))
            for r in rs:
                r.status = Status.COMPLETED
            return rs

        mock_agent_loop.generate_group = mock_gen

        sampler_cfg = SamplerConfig.model_construct(dataloader_cfg=self.mock_dataloader_cfg)
        produce_strategy_cfg = AsyncProduceStrategyConfig(enable_partial_rollout=True)
        sampler = sampler_cfg.build(self.mock_tokenizer, self.replay_buffer)
        strategy = produce_strategy_cfg.build()

        await strategy.produce_batch(
            mock_agent_loop,
            sampler,
            self.replay_buffer,
            batch_size=1,
            task_name=task_name,
        )

        self.assertTrue(seen_flags)
        self.assertTrue(all(flag is True for flag in seen_flags))

    async def test_async_produce_strategy_config_uses_produce_side_fields(self):
        produce_strategy_cfg = AsyncProduceStrategyConfig(over_sample_threshold=1.5, enable_partial_rollout=True)

        self.assertEqual(produce_strategy_cfg.over_sample_threshold, 1.5)
        self.assertTrue(produce_strategy_cfg.enable_partial_rollout)

        strategy = produce_strategy_cfg.build()
        self.assertEqual(strategy.over_sample_threshold, 1.5)
        self.assertTrue(strategy.enable_partial_rollout)
