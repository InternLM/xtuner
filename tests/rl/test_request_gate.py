"""RequestGate 的完整请求准入协议测试。

Good Tests:
- 只通过 request/seal/drain/install_version/reopen 这些 public 入口验证状态机。
- 使用真实的 asyncio 并发路径观察等待、结算和失败传播。

Bad Tests:
- 不读取 _active/_failures 等内部字典。
- 不断言内部 condition 的实现细节。
"""

import asyncio
import unittest

from xtuner.v1.rl.rollout.request_gate import RequestGate, SettlementReceipt


class TestRequestGate(unittest.IsolatedAsyncioTestCase):
    async def test_closed_gate_holds_requests_until_publish(self):
        gate = RequestGate(wait_timeout=5, initially_open=False)
        admitted: list[int] = []

        async def serve() -> None:
            async with gate.request("session-0") as lease:
                admitted.append(lease.model_step)

        server = asyncio.create_task(serve())
        await asyncio.sleep(0.05)
        self.assertEqual(admitted, [])

        await gate.seal(1)
        await gate.drain(timeout=1)
        await gate.install_version(1, 0)
        await gate.reopen(1)
        await asyncio.wait_for(server, timeout=1)
        self.assertEqual(admitted, [0])

    async def test_drain_waits_for_settled_request(self):
        gate = RequestGate(wait_timeout=5, initially_open=True)
        entered = asyncio.Event()
        release = asyncio.Event()

        async def serve() -> None:
            async with gate.request("session-0") as lease:
                entered.set()
                await release.wait()
                lease.settlement = SettlementReceipt(True, True, "trace_committed")

        server = asyncio.create_task(serve())
        await entered.wait()
        await gate.seal(1)
        drain = asyncio.create_task(gate.drain(timeout=5))
        await asyncio.sleep(0.05)
        self.assertFalse(drain.done())

        release.set()
        await asyncio.wait_for(server, timeout=1)
        await asyncio.wait_for(drain, timeout=1)
        self.assertEqual(gate.snapshot()["active"], [])

    async def test_unsettled_request_poisons_later_admission(self):
        gate = RequestGate(wait_timeout=5, initially_open=True)

        async def serve() -> None:
            async with gate.request("session-0"):
                pass

        await asyncio.wait_for(serve(), timeout=1)
        await gate.seal(1)
        with self.assertRaisesRegex(RuntimeError, "Unsettled requests"):
            await gate.drain(timeout=1)
        with self.assertRaisesRegex(RuntimeError, "Request barrier failed"):
            async with gate.request("session-1"):
                pass

    async def test_drain_timeout_poisons_barrier(self):
        gate = RequestGate(wait_timeout=5, initially_open=True)
        release = asyncio.Event()

        async def serve() -> None:
            async with gate.request("session-0"):
                await release.wait()

        server = asyncio.create_task(serve())
        await asyncio.sleep(0.05)
        await gate.seal(1)
        with self.assertRaises(TimeoutError):
            await gate.drain(timeout=0.05)
        with self.assertRaisesRegex(RuntimeError, "Unsettled requests"):
            await gate.drain(timeout=1)
        release.set()
        await asyncio.wait_for(server, timeout=1)

    async def test_same_session_cannot_generate_concurrently(self):
        gate = RequestGate(wait_timeout=5, initially_open=True)
        entered = asyncio.Event()
        release = asyncio.Event()

        async def serve() -> None:
            async with gate.request("session-0") as lease:
                entered.set()
                await release.wait()
                lease.settlement = SettlementReceipt(True, True, "trace_committed")

        first = asyncio.create_task(serve())
        await entered.wait()
        with self.assertRaisesRegex(RuntimeError, "Concurrent generation"):
            async with gate.request("session-0"):
                pass
        release.set()
        await asyncio.wait_for(first, timeout=1)

    async def test_version_publication_requires_drained_current_epoch(self):
        gate = RequestGate(wait_timeout=5, initially_open=True)
        with self.assertRaisesRegex(RuntimeError, "drained current barrier"):
            await gate.install_version(0, 1)

        await gate.seal(1)
        await gate.drain(timeout=1)
        await gate.install_version(1, 2)
        with self.assertRaisesRegex(RuntimeError, "Cannot reopen"):
            await gate.reopen(0)
        await gate.reopen(1)
        with self.assertRaisesRegex(RuntimeError, "Version publication"):
            await gate.install_version(1, 3)

    async def test_model_versions_cannot_move_backwards(self):
        gate = RequestGate(wait_timeout=5, initially_open=True)
        await gate.seal(1)
        await gate.drain(timeout=1)
        await gate.install_version(1, 5)
        await gate.reopen(1)

        await gate.seal(2)
        await gate.drain(timeout=1)
        with self.assertRaisesRegex(ValueError, "cannot move backwards"):
            await gate.install_version(2, 4)

    async def test_request_wait_timeout(self):
        gate = RequestGate(wait_timeout=0.05, initially_open=False)
        with self.assertRaises(TimeoutError):
            async with gate.request("session-0"):
                pass
