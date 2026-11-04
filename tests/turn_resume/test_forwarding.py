import asyncio
import unittest

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from xtuner.v1.rl.rollout.request_gate import RequestGate
from xtuner.v1.rl.rollout.session_server import SessionServer


class ForwardingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.sent = asyncio.Event()
        self.finish = asyncio.Event()
        self.commit = asyncio.Event()
        self.preparation_error = False
        self.trace_error = False
        self.reason = "stop"
        self.timeout = False

        async def upstream(request):
            self.sent.set()
            await self.finish.wait()
            return web.json_response(
                dict(
                    id="backend-id",
                    choices=[dict(finish_reason=self.reason, message=dict(role="assistant", content="ok"))],
                )
            )

        app = web.Application()
        app.router.add_post("/v1/chat/completions", upstream)
        self.upstream = TestServer(app)
        await self.upstream.start_server()
        self.server = SessionServer.__new__(SessionServer)
        self.server.worker_base_url = str(self.upstream.make_url("")).rstrip("/")
        self.server.request_timeout = 1
        self.server.read_bufsize = 4096
        self.server.stop_word = "<eos>"
        self.server.request_gate = RequestGate(wait_timeout=1)

        async def on_request(body, fmt, **kwargs):
            if self.preparation_error:
                raise ValueError("preparation")
            return body

        async def on_response(*args):
            await self.commit.wait()
            if self.trace_error:
                raise ValueError("trace failed")

        self.server.on_request = on_request
        self.server.on_response = on_response
        app = web.Application()
        app.router.add_route("*", "/{path:.*}", self.server._handle_request)
        self.client = TestClient(TestServer(app))
        await self.client.start_server()

    async def asyncTearDown(self):
        self.finish.set()
        self.commit.set()
        await self.client.close()
        await self.upstream.close()

    async def send(self):
        return await self.client.post("/v1/chat/completions", json=dict(session_id="s", messages=[], stream=False))

    async def test_trace_commit_is_inside_drain_and_request_id_reaches_client(self):
        task = asyncio.create_task(self.send())
        await self.sent.wait()
        gate = self.server.request_gate
        await gate.seal(1)
        drain = asyncio.create_task(gate.drain(1))
        self.finish.set()
        await asyncio.sleep(0.01)
        self.assertFalse(drain.done())
        self.commit.set()
        await drain
        response = await task
        self.assertEqual(response.status, 200)
        self.assertTrue((await response.json())["id"].startswith("chatcmpl-"))
        self.assertEqual(gate.snapshot()["active"], [])

    async def test_preparation_failure_proves_backend_not_started(self):
        self.preparation_error = True
        response = await self.send()
        self.assertEqual(response.status, 500)
        self.assertFalse(self.sent.is_set())
        gate = self.server.request_gate
        await gate.seal(1)
        await gate.drain(1)

    async def test_trace_failure_after_complete_backend_is_safe_failed_attempt(self):
        self.trace_error = True
        self.finish.set()
        self.commit.set()
        response = await self.send()
        self.assertEqual(response.status, 500)
        gate = self.server.request_gate
        await gate.seal(1)
        await gate.drain(1)

    async def test_bad_generation_terminal_is_never_successful_training_response(self):
        self.reason = "parse_error"
        self.finish.set()
        self.commit.set()
        response = await self.send()
        self.assertEqual(response.status, 500)
        gate = self.server.request_gate
        await gate.seal(1)
        await gate.drain(1)

    async def test_timeout_with_unknown_backend_end_blocks_update(self):
        self.server.request_timeout = 0.02
        response = await self.send()
        self.assertGreaterEqual(response.status, 500)
        gate = self.server.request_gate
        await gate.seal(1)
        with self.assertRaises(RuntimeError):
            await gate.drain(1)
        self.assertEqual(len(gate.snapshot()["active"]), 1)

    async def test_stream_and_eval_generation_rejected_before_backend(self):
        for body in (dict(session_id="s", messages=[], stream=True), dict(messages=[])):
            response = await self.client.post("/v1/chat/completions", json=body)
            self.assertEqual(response.status, 400)
        self.assertFalse(self.sent.is_set())

    async def test_unsupported_format_does_not_poison_gate(self):
        response = await self.client.post("/v1/messages", json=dict(session_id="s", messages=[]))
        self.assertEqual(response.status, 400)
        self.assertFalse(self.sent.is_set())
        self.assertEqual(self.server.request_gate.snapshot()["active"], [])
        self.assertEqual(self.server.request_gate.snapshot()["failures"], {})
