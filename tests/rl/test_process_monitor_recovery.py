"""Dependency-free monitor policy tests with bounded real asyncio deadlines."""
import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import time
import unittest
import httpx

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'xtuner/v1/rl/agent_loop/sandbox_agent_loop/process_monitor.py'
spec = importlib.util.spec_from_file_location('tested_monitor', SOURCE)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)

class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    def client(self, outcomes, state='running'):
        values = iter(outcomes)
        async def probe(pid):
            value = next(values, outcomes[-1])
            if isinstance(value, BaseException): raise value
            return value
        async def lifecycle(): return state
        return SimpleNamespace(is_pid_running=probe, get_lifecycle_state=lifecycle)

    async def run_monitor(self, client, rc=None, **kw):
        async def read_rc(): return rc
        return await monitor.observe_process(client, pid=123, timeout=kw.get('timeout',.5),
            recovery_timeout=kw.get('recovery_timeout',.1), poll_interval=.01,
            retry_interval=.01, rc_wait_timeout=.03, rc_poll_interval=.01, read_rc=read_rc)

    async def test_more_than_three_errors_can_recover(self):
        result = await self.run_monitor(self.client([httpx.RemoteProtocolError('fixture')]*5+[False]), rc=0)
        # rc is already trusted, so completion can occur before probe recovery.
        self.assertEqual(result[0], 'entry_completed')
        seen = 0
        client = self.client([httpx.RemoteProtocolError('fixture')]*5+[False])
        async def read_rc():
            nonlocal seen
            seen += 1
            return 0 if seen > 5 else None
        result = await monitor.observe_process(client,pid=123,timeout=.5,recovery_timeout=.2,
            poll_interval=.01,retry_interval=.01,rc_wait_timeout=.03,rc_poll_interval=.01,read_rc=read_rc)
        self.assertEqual(result[0], 'entry_completed')
        self.assertEqual(result[2]['pid_check_errors_total'],5)

    async def test_persistent_unknown_and_terminal(self):
        started=time.monotonic()
        result=await self.run_monitor(self.client([httpx.ReadError('fixture')]), recovery_timeout=.05)
        self.assertEqual(result[0],'pid_monitor_error')
        self.assertLess(time.monotonic()-started,.2)
        self.assertTrue(result[2]['recovery_deadline_reached'])
        result=await self.run_monitor(self.client([httpx.ReadError('fixture')],state='failed'))
        self.assertEqual(result[0],'sandbox_unavailable')

    async def test_entry_deadline_and_cancellation(self):
        result=await self.run_monitor(self.client([httpx.ReadError('fixture')]), timeout=.04,recovery_timeout=120)
        self.assertEqual(result[0],'pid_monitor_error')
        self.assertTrue(result[2]['deadline_reached'])
        with self.assertRaises(asyncio.CancelledError):
            await self.run_monitor(self.client([asyncio.CancelledError()]))

    async def test_exit_rc_missing_zombie_and_unknown_boolean(self):
        self.assertEqual((await self.run_monitor(self.client([False]),rc=7))[:2],('entry_failed',7))
        self.assertEqual((await self.run_monitor(self.client([False])))[0],'rc_missing')
        self.assertEqual((await self.run_monitor(self.client([True]),timeout=.03))[0],'timeout')
        self.assertEqual((await self.run_monitor(self.client([None]),recovery_timeout=.03))[0],'pid_monitor_error')

if __name__=='__main__': unittest.main(verbosity=2)
