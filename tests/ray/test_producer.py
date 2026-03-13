import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from xtuner.v1.rl.base.sampler import SamplerConfig, Sampler
from xtuner.v1.rl.base.producer import SyncProduceStrategyConfig, OverProduceStrategyConfig
from xtuner.v1.rl.base.replay_buffer import AsyncReplayBufferConfig
from xtuner.v1.data_proto.rl_data import RolloutState, Status

class MockRolloutState:
    def __init__(self, id, seq_staleness=1, status=Status.COMPLETED):
        self.id = id
        self.status = status
        self.seq_staleness = seq_staleness

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
        replay_buffer_cfg = AsyncReplayBufferConfig(min_staleness=1, max_staleness=5)
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
        
        data = await sampler.sample(task_name)
        self.assertEqual(data[0].id, 999)

    async def test_sync_produce_strategy(self):
        task_name = "test_task"
        mock_agent_loop = MagicMock()
        async def mock_gen(rs):
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
        task_name = "test_task"
        call_count = 0
        async def mock_gen(rs):
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
        produce_strategy_cfg = OverProduceStrategyConfig(staleness_threshold = 1)
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


class TestAgentLoopManager(unittest.IsolatedAsyncioTestCase):

    async def test_restart_inactive_workers_before_produce_batch(self):
        from xtuner.v1.rl.base.agent_loop_manager import AgentLoopManager

        mock_agent_loop = MagicMock()
        mock_agent_loop.rollout_ctl = MagicMock()
        mock_agent_loop.rollout_ctl.restart_inactive_workers = MagicMock()
        mock_agent_loop.rollout_ctl.restart_inactive_workers.remote = AsyncMock()

        mock_strategy = MagicMock()
        mock_strategy.produce_batch = AsyncMock()

        mock_sampler = MagicMock()
        replay_buffer = MagicMock()
        replay_buffer.get = AsyncMock(return_value=[[MockRolloutState(1)]])

        manager = AgentLoopManager(
            agent_loop=mock_agent_loop,
            produce_strategy=mock_strategy,
            sampler=mock_sampler,
            replay_buffer=replay_buffer,
            task_name="test_task",
        )

        await manager.produce_batch(batch_size=1)

        mock_agent_loop.rollout_ctl.restart_inactive_workers.remote.assert_awaited_once()
        mock_strategy.produce_batch.assert_awaited_once()

