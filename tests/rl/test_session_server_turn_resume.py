"""SessionServer 的完整请求准入与结算测试。

Good Tests:
- 使用真实 SessionServer 转发路径、真实 RequestGate 和真实 HTTP 上游。
- 只通过 HTTP 状态、trace 写入和 gate 的 public 状态观察行为。

Bad Tests:
- 不 mock SessionServer._forward_request 或 gate 内部状态。
- 不断言内部 lease 字段以外的实现细节。
"""

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from xtuner.v1.rl.rollout.request_gate import RequestGate
from xtuner.v1.rl.rollout.session_server import SessionServer


def _chat_response(*, finish_reason: str = "stop", status: int = 200) -> web.Response:
    return web.json_response(
        {
            "id": "upstream-id",
            "model": "test-model",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": finish_reason}
            ],
        },
        status=status,
    )


class TestSessionServerTurnResume(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.upstream_calls: list[dict] = []
        self.upstream_response = _chat_response()
        self.upstream_gate: asyncio.Event | None = None

        async def upstream_handler(request: web.Request) -> web.Response:
            body = await request.read()
            self.upstream_calls.append({"path": request.path, "body": body})
            if self.upstream_gate is not None:
                await self.upstream_gate.wait()
            return self.upstream_response

        upstream_app = web.Application()
        upstream_app.router.add_route("*", "/{path:.*}", upstream_handler)
        self.upstream_server = TestServer(upstream_app)
        await self.upstream_server.start_server()

        self.session_server = SessionServer.__new__(SessionServer)
        self.session_server.worker_base_url = str(self.upstream_server.make_url("")).rstrip("/")
        self.session_server.request_timeout = 10.0
        self.session_server.read_bufsize = 2**20
        self.session_server.enable_return_routed_experts = False
        self.session_server.stop_word = "<eos>"
        self.session_server.tokenizer = Mock()
        self.session_server.tokenizer.apply_chat_template.return_value = "rendered prompt"
        self.session_server.tokenizer.encode.return_value = [101, 102]
        self.session_server.store = SimpleNamespace(search=SimpleNamespace(remote=AsyncMock(return_value=("", []))))
        self.session_server.request_gate = RequestGate(wait_timeout=5, initially_open=False)
        self.on_response = AsyncMock(return_value=None)
        self.session_server.on_response = self.on_response

        proxy_app = web.Application()
        proxy_app.router.add_route("*", "/{path:.*}", self.session_server._handle_request)
        self.proxy_client = TestClient(TestServer(proxy_app))
        await self.proxy_client.start_server()

    async def asyncTearDown(self):
        await self.proxy_client.close()
        await self.upstream_server.close()

    async def _publish(self, model_step: int = 0) -> None:
        gate = self.session_server.request_gate
        await gate.seal(1)
        await gate.drain(timeout=1)
        await gate.install_version(1, model_step)
        await gate.reopen(1)

    def _payload(self, *, session_id: str = "session-0", **overrides) -> dict:
        payload = {
            "model": "test-model",
            "session_id": session_id,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 16,
        }
        payload.update(overrides)
        return payload

    async def test_request_waits_closed_and_returns_the_lease_id_after_publish(self):
        request = asyncio.create_task(self.proxy_client.post("/v1/chat/completions", json=self._payload()))
        await asyncio.sleep(0.05)
        self.assertEqual(self.upstream_calls, [])
        self.assertFalse(request.done())

        await self._publish(model_step=0)
        response = await asyncio.wait_for(request, timeout=2)

        self.assertEqual(response.status, 200)
        body = await response.json()
        self.assertTrue(body["id"].startswith("chatcmpl-"))
        self.assertEqual(len(self.upstream_calls), 1)
        lease = self.on_response.await_args.kwargs["lease"]
        self.assertEqual(lease.model_step, 0)
        self.assertEqual(self.session_server.request_gate.snapshot()["active"], [])

    async def test_unsuccessful_terminal_poisons_the_gate(self):
        await self._publish()
        self.upstream_response = _chat_response(finish_reason="error")
        response = await self.proxy_client.post("/v1/chat/completions", json=self._payload())
        self.assertEqual(response.status, 200)

        gate = self.session_server.request_gate
        self.assertTrue(gate.snapshot()["failures"])
        await gate.seal(2)
        with self.assertRaisesRegex(RuntimeError, "Unsettled requests"):
            await gate.drain(timeout=1)

    async def test_known_context_rejection_settles_safely(self):
        await self._publish()
        self.upstream_response = web.json_response(
            {
                "object": "error",
                "type": "invalid_request_error",
                "code": 400,
                "message": "Input length (100) must be smaller than the model context length (50).",
            },
            status=400,
        )
        response = await self.proxy_client.post("/v1/chat/completions", json=self._payload())
        self.assertEqual(response.status, 400)

        gate = self.session_server.request_gate
        self.assertEqual(gate.snapshot()["failures"], {})
        await gate.seal(2)
        await gate.drain(timeout=1)

    async def test_turn_resume_rejects_untraced_or_streaming_generation(self):
        await self._publish()

        anthropic = await self.proxy_client.post("/v1/messages", json=self._payload())
        streaming = await self.proxy_client.post("/v1/chat/completions", json=self._payload(stream=True))
        untraced = await self.proxy_client.post("/v1/chat/completions", json=self._payload(return_token_ids=False))

        self.assertEqual([anthropic.status, streaming.status, untraced.status], [400, 400, 400])
        self.assertEqual(self.upstream_calls, [])

    async def test_private_control_fields_are_stripped_from_the_wire_body(self):
        await self._publish()
        response = await self.proxy_client.post(
            "/v1/chat/completions",
            json=self._payload(_xt_request_id="forged", _xt_model_step=99),
        )
        self.assertEqual(response.status, 200)

        forwarded = json.loads(self.upstream_calls[0]["body"])
        self.assertNotIn("_xt_request_id", forwarded)
        self.assertNotIn("_xt_model_step", forwarded)
        self.assertNotIn("session_id", forwarded)

    async def test_concurrent_generation_in_one_session_is_rejected(self):
        await self._publish()
        self.upstream_gate = asyncio.Event()

        first = asyncio.create_task(self.proxy_client.post("/v1/chat/completions", json=self._payload()))
        for _ in range(100):
            if self.upstream_calls:
                break
            await asyncio.sleep(0.01)
        second = await self.proxy_client.post("/v1/chat/completions", json=self._payload())
        self.assertEqual(second.status, 500)

        self.upstream_gate.set()
        self.assertEqual((await asyncio.wait_for(first, timeout=2)).status, 200)
        self.assertEqual(self.session_server.request_gate.snapshot()["failures"], {})


if __name__ == "__main__":
    unittest.main()
