import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from xtuner.v1.rl.agent_loop_manager.agent_turn_pause_policy import AgentTurnPausePolicy
from xtuner.v1.rl.agent_loop_manager.disagg_agent_loop_manager import DisaggAgentLoopManager
from xtuner.v1.rl.agent_loop_manager.disagg_producer import DisaggAsyncProduceStrategy
from xtuner.v1.rl.rollout.request_gate import RequestGate, SettlementReceipt


class QuiesceTests(unittest.IsolatedAsyncioTestCase):
    def manager(self):
        strategy = DisaggAsyncProduceStrategy(0, False, -1, 3, None, 1, lambda *args: True)
        manager = DisaggAgentLoopManager.__new__(DisaggAgentLoopManager)
        manager._group_admission = asyncio.Event()
        manager._group_admission.set()
        manager._producer_idle = asyncio.Event()
        manager._producer_idle.set()
        manager.task_runners = [SimpleNamespace(task_name="swe", produce_strategy=strategy)]
        results = []

        async def put(result):
            results.append(result)
            return True

        ctx = SimpleNamespace(put_generated_group=put)
        manager._pending_context = lambda task: ctx
        return manager, strategy, results

    async def test_quiesce_keeps_requests_open_and_collects_the_original_episode(self):
        manager, strategy, results = self.manager()
        gate = RequestGate(wait_timeout=1)
        finish = asyncio.Event()

        async def episode():
            await finish.wait()
            async with gate.request("s") as lease:
                lease.settlement = SettlementReceipt(True, True, "trace_committed")
            return "original"

        async def spawn():
            return asyncio.create_task(episode())

        await strategy._pending_tasks.schedule_one(max_pending=1, should_abort=lambda: False, spawn_one=spawn)
        q = asyncio.create_task(manager.quiesce(1))
        await asyncio.sleep(0)
        self.assertTrue(gate.snapshot()["open"])
        self.assertFalse(q.done())
        self.assertFalse(manager._group_admission.is_set())
        self.assertTrue(strategy.scheduling_paused)
        finish.set()
        await q
        self.assertEqual(results, ["original"])
        self.assertEqual(strategy.pending_task_count(), 0)

    async def test_waits_for_a_producer_claim_that_has_not_finished_putting(self):
        manager, strategy, results = self.manager()
        manager._producer_idle.clear()
        q = asyncio.create_task(manager.quiesce(1))
        await asyncio.sleep(0)
        self.assertFalse(q.done())
        results.append("producer-put-complete")
        manager._producer_idle.set()
        await q
        self.assertEqual(results, ["producer-put-complete"])

    async def test_timeout_never_drops_or_cancels_live_episode(self):
        manager, strategy, _ = self.manager()
        released = asyncio.Event()
        started = asyncio.Event()

        async def episode():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                released.set()

        task = asyncio.create_task(episode())
        await started.wait()

        async def spawn():
            return task

        await strategy._pending_tasks.schedule_one(max_pending=1, should_abort=lambda: False, spawn_one=spawn)
        with self.assertRaises(TimeoutError):
            await manager.quiesce(0.01)
        self.assertFalse(task.done())
        self.assertEqual(strategy.pending_task_count(), 1)
        await strategy.close()
        self.assertTrue(released.is_set())
        self.assertEqual(strategy.pending_task_count(), 0)

    async def test_only_checkpoint_sync_quiesces(self):
        manager = SimpleNamespace(quiesce=AsyncMock())
        policy = AgentTurnPausePolicy(freeze_timeout=1, drain_timeout=2, quiesce_timeout=3)
        await policy.before_sync(manager, False)
        manager.quiesce.assert_not_awaited()
        await policy.before_sync(manager, True)
        manager.quiesce.assert_awaited_once_with(3)
