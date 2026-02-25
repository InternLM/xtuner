import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from xtuner.v1.rl.base.producer import Sampler, SamplerWithReplayBuffer, SyncProduceStrategy, AsyncProduceStrategy
from xtuner.v1.rl.base.replay_buffer import ReplayBuffer, StalenessBackend
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
        self.backend = StalenessBackend(limit=10, max_staleness=5)
        self.replay_buffer = ReplayBuffer(storage_backend=self.backend)

    async def test_sampler_with_replay_buffer(self):
        sampler = SamplerWithReplayBuffer("test_task", self.mock_dataloader_cfg, self.mock_tokenizer, self.replay_buffer)
        
        # 场景 A: ReplayBuffer 为空，从 Dataloader 拿
        data = await sampler.sample()
        self.assertEqual(data.id, 0)

        # 场景 B: ReplayBuffer 有 ABORTED 数据，优先拿
        aborted_item = MockRolloutState(999, status=Status.ABORTED)
        await self.replay_buffer.put([aborted_item], "test_task")
        
        data = await sampler.sample()
        self.assertEqual(data[0].id, 999)

    async def test_sync_produce_strategy(self):
        # 1. 模拟 AgentLoop
        mock_agent_loop = MagicMock()
        mock_agent_loop.task_name = "test_task"
        # generate_group 返回的是 List[RolloutState]
        async def mock_gen(rs, k):
            rs.status = Status.COMPLETED
            return [rs]
        mock_agent_loop.generate_group = mock_gen

        sampler = Sampler("test_task", self.mock_dataloader_cfg, self.mock_tokenizer)
        strategy = SyncProduceStrategy(self.replay_buffer)

        # 执行：生产 batch_size 为 2 的数据
        await strategy.produce_batch(mock_agent_loop, sampler, batch_size=2, prompt_k=1)

        # 验证：ReplayBuffer 中应该有 2 条 COMPLETED 数据
        final_data = await self.replay_buffer.get(10, "test_task", Status.COMPLETED)
        self.assertEqual(len(final_data), 2)
        self.assertEqual(final_data[0].id, 0)
        self.assertEqual(final_data[1].id, 1)

    async def test_async_produce_strategy(self):
        # 这个async_produce_strategy的测试主要验证超发逻辑 + staleness 优先get的逻辑
        # 异步的其他功能如 partial_rollout, tail_batch不在这里进行验证 
        mock_agent_loop = MagicMock()
        mock_agent_loop.task_name = "test_task"

        call_count = 0
        async def mock_gen(rs, k):
            nonlocal call_count
            call_count += 1
            if isinstance(rs, list):
                for r in rs:
                    r.seq_staleness = 5  
                    r.status = Status.COMPLETED
                return rs
            else:
                rs.seq_staleness = call_count
                rs.status = Status.COMPLETED
                return [rs]
        mock_agent_loop.generate_group = mock_gen

        sampler = SamplerWithReplayBuffer("test_task", self.mock_dataloader_cfg, self.mock_tokenizer, self.replay_buffer)
        strategy = AsyncProduceStrategy(self.replay_buffer, staleness_threshold = 1)
        # 预处理
        aborted_item = MockRolloutState(999, status=Status.ABORTED)
        await self.replay_buffer.put([aborted_item], "test_task")
        # 执行
        await strategy.produce_batch(mock_agent_loop, sampler, batch_size=2, prompt_k=1)

        # 验证：ReplayBuffer 中应该有 4 条 COMPLETED 数据，
        # NOTE(@duanyanhui): 目前还没实现暂停功能，所以4条都会推理完成,4条数据按照新鲜度顺序排列，999 是最旧的，0 是最新的
        final_data = await self.replay_buffer.get(10, "test_task", Status.COMPLETED)
        self.assertEqual(len(final_data), 4)
        self.assertEqual(final_data[0].id, 999)
        self.assertEqual(final_data[1].id, 2)
        self.assertEqual(final_data[2].id, 1)
        self.assertEqual(final_data[3].id, 0)