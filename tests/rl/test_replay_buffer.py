import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from xtuner.v1.data_proto.rl_data import Status
from xtuner.v1.rl.replay_buffer import AsyncReplayBufferConfig, SyncReplayBufferConfig


class MockState:
    def __init__(self, state_id, staleness=0, input_ids=None, status=Status.COMPLETED):
        self.id = state_id
        self.seq_staleness = staleness
        self.status = status
        self.input_ids = input_ids if input_ids is not None else [state_id]


class TestReplayBuffer(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _get_sorted_input_ids(data_groups):
        return sorted(tuple(state.input_ids) for group in data_groups for state in group)
        
    async def _run_roundtrip_input_ids_case(self, replay_buffer_cfg, put_groups, task_name, sample_size):
        with TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir)
            original = replay_buffer_cfg.build()
            for group in put_groups:
                await original.put(group, task_name)
            await original.save(save_path)

            old_sampled = await original.get(sample_size, task_name, Status.COMPLETED)

            resumed = replay_buffer_cfg.build()
            await resumed.resume(save_path)
            new_sampled = await resumed.get(sample_size, task_name, Status.COMPLETED)

            self.assertEqual(self._get_sorted_input_ids(old_sampled), self._get_sorted_input_ids(new_sampled))

    async def test_basic_ordering_and_task_isolation(self):
        fifo_cfg = SyncReplayBufferConfig()
        fifo = fifo_cfg.build()
        await fifo.put([MockState(1), MockState(2)], "task1")
        await fifo.put([MockState(3)], "task1")
        await fifo.put([MockState(200)], "task2")

        res_task1 = await fifo.get(2, "task1", Status.COMPLETED)
        res_task2 = await fifo.get(1, "task2", Status.COMPLETED)
        self.assertEqual([s.id for s in res_task1[0]], [1, 2])
        self.assertEqual([s.id for s in res_task1[1]], [3])
        self.assertEqual([s.id for s in res_task2[0]], [200])

        staleness_cfg = AsyncReplayBufferConfig()
        staleness = staleness_cfg.build()
        await staleness.put([MockState("low", staleness=1)], "task")
        await staleness.put([MockState("high", staleness=5)], "task")
        sampled = await staleness.get(2, "task", Status.COMPLETED)
        self.assertEqual(sampled[0][0].id, "high")
        self.assertEqual(sampled[1][0].id, "low")

    async def test_save_resume_keeps_query_behavior_fifo(self):
        replay_buffer_cfg = SyncReplayBufferConfig()
        with TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir)
            buffer = replay_buffer_cfg.build()
            await buffer.put([MockState("a1", status=Status.COMPLETED, input_ids=[11, 12])], "task_a")
            await buffer.put([MockState("a2", status=Status.FAILED, input_ids=[21])], "task_a")
            await buffer.put([MockState("b1", status=Status.COMPLETED, input_ids=[31])], "task_b")
            await buffer.save(save_path)

            resumed = replay_buffer_cfg.build()
            await resumed.resume(save_path)

            self.assertEqual(await resumed.count("task_a", Status.COMPLETED), 1)
            self.assertEqual(await resumed.count("task_a", Status.FAILED), 1)
            self.assertEqual(await resumed.count("task_b", Status.COMPLETED), 1)
            self.assertEqual(await resumed.count("task_b", Status.FAILED), 0)

            completed = await resumed.get(5, "task_a", Status.COMPLETED)
            failed = await resumed.get(5, "task_a", Status.FAILED)
            self.assertEqual([s.id for s in completed[0]], ["a1"])
            self.assertEqual([s.id for s in failed[0]], ["a2"])

            await resumed.put([MockState("a3", input_ids=[41])], "task_a")
            next_completed = await resumed.get(1, "task_a", Status.COMPLETED)
            self.assertEqual([s.id for s in next_completed[0]], ["a3"])

    async def test_save_resume_keeps_query_behavior_staleness(self):
        replay_buffer_cfg = AsyncReplayBufferConfig()
        with TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir)
            buffer = replay_buffer_cfg.build()
            await buffer.put([MockState("done_low", staleness=1, status=Status.COMPLETED, input_ids=[101])], "task")
            await buffer.put([MockState("failed_high", staleness=10, status=Status.FAILED, input_ids=[201])], "task")
            await buffer.put([MockState("done_mid", staleness=5, status=Status.COMPLETED, input_ids=[301, 302])], "task")
            await buffer.save(save_path)

            resumed = replay_buffer_cfg.build()
            await resumed.resume(save_path)

            self.assertEqual(await resumed.count("task", Status.COMPLETED), 2)
            self.assertEqual(await resumed.count("task", Status.FAILED), 1)

            completed = await resumed.get(2, "task", Status.COMPLETED)
            failed = await resumed.get(1, "task", Status.FAILED)
            self.assertEqual(completed[0][0].id, "done_mid")
            self.assertEqual(completed[1][0].id, "done_low")
            self.assertEqual(failed[0][0].id, "failed_high")

    async def test_save_resume_sample_keeps_input_ids_fifo(self):
        await self._run_roundtrip_input_ids_case(
            replay_buffer_cfg=SyncReplayBufferConfig(),
            put_groups=[
                [MockState(1, input_ids=[101, 102]), MockState(2, input_ids=[201])],
                [MockState(3, input_ids=[301, 302, 303])],
            ],
            task_name="task",
            sample_size=2,
        )

    async def test_save_resume_sample_keeps_input_ids_staleness(self):
        await self._run_roundtrip_input_ids_case(
            replay_buffer_cfg=AsyncReplayBufferConfig(),
            put_groups=[
                [MockState("mid", staleness=3, input_ids=[301, 302])],
                [MockState("high", staleness=5, input_ids=[501])],
                [MockState("low", staleness=1, input_ids=[101, 102, 103])],
            ],
            task_name="task",
            sample_size=3,
        )