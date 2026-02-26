import unittest
import asyncio
from xtuner.v1.rl.base.replay_buffer import ReplayBuffer, StorageIndices, FIFOBackend, StalenessBackend
from xtuner.v1.data_proto.rl_data import RolloutState, Status

class MockState:
    def __init__(self, id, staleness=0):
        self.id = id
        self.seq_staleness = staleness
        self.status = Status.COMPLETED

class TestReplayBuffer(unittest.IsolatedAsyncioTestCase):
    async def test_fifo_backend(self):
        backend = FIFOBackend()
        buffer = ReplayBuffer(storage_backend=backend)
        group_states1 = [MockState(i) for i in range(1, 4)]
        group_states2 = [MockState(i) for i in range(5, 7)]
        
        await buffer.put(group_states1, "task1")
        await buffer.put(group_states2, "task1")
        res = await buffer.get(2, "task1", Status.COMPLETED)
        
        self.assertEqual(len(res), 2)
        self.assertEqual(len(res[0]), 3)
        self.assertEqual(len(res[1]), 2)
        self.assertEqual(res[0][0].id, 1)
        self.assertEqual(res[1][0].id, 5)

    async def test_staleness_priority(self):
        backend = StalenessBackend(min_staleness=0, max_staleness=5)
        buffer = ReplayBuffer(storage_backend=backend)
        
        s1 = MockState(id="low", staleness=1)
        s5 = MockState(id="high", staleness=5)
        
        await buffer.put([s1], "task1")
        await buffer.put([s5], "task1")
        
        res = await buffer.get(2, "task1", Status.COMPLETED)
        self.assertEqual(res[0][0].id, "high")
        self.assertEqual(res[1][0].id, "low")

    async def test_multi_task(self):
        buffer = ReplayBuffer()
        await buffer.put([MockState(100)], "task_a")
        await buffer.put([MockState(200)], "task_b")
        
        res_a = await buffer.get(10, "task_a", Status.COMPLETED)
        res_b = await buffer.get(10, "task_b", Status.COMPLETED)
        self.assertEqual(len(res_a), 1)
        self.assertEqual(len(res_a[0]), 1)
        self.assertEqual(res_a[0][0].id, 100)
        self.assertEqual(len(res_b), 1)
        self.assertEqual(len(res_b[0]), 1)
        self.assertEqual(res_b[0][0].id, 200)