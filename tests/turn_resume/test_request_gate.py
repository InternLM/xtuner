import asyncio
import importlib.util
import sys
import unittest
from pathlib import Path

source = Path(__file__).resolve().parents[2] / 'xtuner/v1/rl/rollout/request_gate.py'
spec = importlib.util.spec_from_file_location('request_gate', source)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
RequestGate, Receipt = m.RequestGate, m.SettlementReceipt


class GateTests(unittest.IsolatedAsyncioTestCase):
    async def test_atomic_admission_drains_commit_and_publishes(self):
        gate = RequestGate(wait_timeout=1)
        entered, commit = asyncio.Event(), asyncio.Event()
        versions = []
        async def first():
            async with gate.request('s') as lease:
                versions.append(lease.model_step)
                entered.set()
                await commit.wait()
                lease.settlement = Receipt(True, True, 'trace_committed')
        task = asyncio.create_task(first())
        await entered.wait()
        await gate.seal(1)
        drain = asyncio.create_task(gate.drain(1))
        async def next_turn():
            async with gate.request('s') as lease:
                versions.append(lease.model_step)
                lease.settlement = Receipt(True, True, 'trace_committed')
        queued = asyncio.create_task(next_turn())
        await asyncio.sleep(0)
        self.assertFalse(drain.done())
        self.assertEqual(len(gate.snapshot()['active']), 1)
        commit.set()
        await task
        await drain
        self.assertFalse(queued.done())
        await gate.install_version(1, 2)
        await gate.reopen(1)
        await queued
        self.assertEqual(versions, [0, 2])

    async def test_unknown_backend_or_uncommitted_trace_poison_barrier(self):
        for receipt in (None, Receipt(False, True, 'abort_200'), Receipt(True, False, 'trace_failed')):
            gate = RequestGate(wait_timeout=1)
            async with gate.request('s') as lease:
                lease.settlement = receipt
            await gate.seal(1)
            with self.assertRaises(RuntimeError):
                await gate.drain(1)
            self.assertEqual(len(gate.snapshot()['active']), 1)
            with self.assertRaises(RuntimeError):
                await gate.install_version(1, 1)
            with self.assertRaises(RuntimeError):
                async with gate.request('other'):
                    self.fail('poisoned gate admitted a request')

    async def test_handler_cancellation_never_releases_unknown_backend(self):
        gate = RequestGate(wait_timeout=1)
        entered = asyncio.Event()
        async def run():
            async with gate.request('s'):
                entered.set()
                await asyncio.Event().wait()
        task = asyncio.create_task(run())
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        await gate.seal(1)
        with self.assertRaises(RuntimeError):
            await gate.drain(1)

    async def test_safe_failure_before_backend_start_releases(self):
        gate = RequestGate(wait_timeout=1)
        async with gate.request('s') as lease:
            lease.settlement = Receipt(True, True, 'not_started')
        await gate.seal(1)
        await gate.drain(1)

    async def test_wait_deadline_does_not_count_as_inflight(self):
        gate = RequestGate(wait_timeout=0.01)
        await gate.seal(1)
        with self.assertRaises(TimeoutError):
            async with gate.request('s'):
                self.fail('sealed')
        self.assertEqual(gate.snapshot()['active'], [])

    async def test_epoch_idempotency_and_version_conflict(self):
        gate = RequestGate(wait_timeout=1)
        await gate.seal(1)
        await gate.seal(1)
        await gate.install_version(1, 2)
        await gate.install_version(1, 2)
        with self.assertRaises(ValueError):
            await gate.install_version(1, 3)
        await gate.reopen(1)
        await gate.reopen(1)
        with self.assertRaises(ValueError):
            await gate.seal(1)
        with self.assertRaises(ValueError):
            await gate.seal(0)

    async def test_same_session_overlap_rejected(self):
        gate = RequestGate(wait_timeout=1)
        async with gate.request('s') as lease:
            with self.assertRaises(RuntimeError):
                async with gate.request('s'):
                    self.fail('overlap')
            lease.settlement = Receipt(True, True, 'trace_committed')


if __name__ == '__main__':
    unittest.main()
