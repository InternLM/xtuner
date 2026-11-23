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
import logging
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from aiohttp import ClientSession, ConnectionTimeoutError, web
from aiohttp.test_utils import TestClient, TestServer

from xtuner.v1.rl.rollout.controller import RolloutController, RolloutRequestState
from xtuner.v1.rl.rollout.cross_weight_update_rollout import CrossWeightUpdateRolloutConfig
from xtuner.v1.rl.rollout.request_gate import RequestGate
from xtuner.v1.rl.rollout.session_server import SessionServer, SessionServerActor


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


class TestSessionServerCrossWeightUpdateRollout(unittest.IsolatedAsyncioTestCase):
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
        self.session_server._forward_timeouts = {}
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

    async def test_cross_weight_update_rollout_rejects_untraced_or_streaming_generation(self):
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

    async def test_never_sent_connection_timeout_fails_without_poisoning_publication(self):
        await self._publish()
        target = self.session_server.worker_base_url
        original = ClientSession.request

        def request(client, method, url, **kwargs):
            if str(url).startswith(target):
                raise ConnectionTimeoutError("connector never sent headers")
            return original(client, method, url, **kwargs)

        with patch("xtuner.v1.rl.rollout.session_server.ClientSession.request", new=request):
            response = await self.proxy_client.post("/v1/chat/completions", json=self._payload())
            await response.read()
        self.assertGreaterEqual(response.status, 500)
        self.assertEqual(self.upstream_calls, [])
        gate = self.session_server.request_gate
        await gate.seal(2)
        await gate.drain(1)
        await gate.install_version(2, 1)
        await gate.reopen(2)
        response = await self.proxy_client.post("/v1/chat/completions", json=self._payload())
        self.assertEqual(response.status, 200)
        self.assertEqual(self.on_response.await_args.kwargs["lease"].model_step, 1)

    async def _timeout_controller(self, fault=None):
        # Run real controller + actor methods; only Ray transport and native backend sleep are doubles.
        loop = asyncio.get_running_loop()

        def remote(method):
            async def invoke(*args, **kwargs):
                result = method(*args, **kwargs)
                if asyncio.iscoroutine(result):
                    result = await result
                return result

            return SimpleNamespace(remote=lambda *a, **kw: asyncio.run_coroutine_threadsafe(invoke(*a, **kw), loop))

        actor = SessionServerActor.__new__(SessionServerActor)
        actor.server = self.session_server
        server = SimpleNamespace(
            **{
                name: remote(getattr(actor, name))
                for name in (
                    "request_status",
                    "seal_requests",
                    "drain_requests",
                    "settle_after_backend_sleep",
                    "install_request_version",
                    "reopen_requests",
                )
            }
        )
        stopped = False

        async def sleep(epoch):
            nonlocal stopped
            if fault == "sleep":
                raise RuntimeError("native sleep failed")
            stopped = True
            if fault == "topology":
                worker.session_url = "http://replacement"
            self.upstream_gate.set()
            return dict(
                epoch=epoch,
                backend_url="wrong" if fault == "identity" else self.session_server.worker_base_url,
                level=2,
                native_sleep_returned=True,
                is_sleeping=True,
            )

        async def release(_sessions):
            self.assertTrue(stopped, "trace cleanup preceded backend stop")
            if fault == "trace":
                raise RuntimeError("trace cleanup failed")
            return _sessions

        self.session_server.store.release_sessions = SimpleNamespace(remote=release)
        self.session_server.store.list_sessions = SimpleNamespace(remote=AsyncMock(return_value=[]))
        worker = SimpleNamespace(
            url=self.session_server.worker_base_url,
            session_url="http://session",
            actor=SimpleNamespace(get_session_server=remote(lambda: server), sleep_for_timeout_recovery=remote(sleep)),
        )
        controller = RolloutController.__new__(RolloutController)
        controller.config = SimpleNamespace(cross_weight_update_rollout=CrossWeightUpdateRolloutConfig())
        controller.registry = SimpleNamespace(active_entrypoints=lambda: (worker,), active_workers=lambda: (worker,))
        controller.health_manager = SimpleNamespace(
            pause=lambda: None, wait_recovery_done=lambda **kw: True, resume=lambda: None
        )
        controller._request_state = RolloutRequestState()
        controller.logger = logging.getLogger(__name__)
        controller._broadcast_to_workers = lambda *a, **kw: [True]
        return controller

    async def test_sent_timeout_recovers_through_pause_and_publishes_next_version(self):
        await self._publish()
        self.upstream_gate = asyncio.Event()
        self.session_server.request_timeout = 0.1
        responses = await asyncio.gather(
            *(
                self.proxy_client.post("/v1/chat/completions", json=self._payload(session_id=f"session-{i}"))
                for i in range(2)
            )
        )
        for response in responses:
            await response.read()
            self.assertGreaterEqual(response.status, 500)
        self.assertEqual(len(self.upstream_calls), 2)
        self.assertTrue(self.session_server.request_gate.snapshot()["failures"])
        controller = await self._timeout_controller()
        # The fixture published epoch 1; controller must own the next barrier epoch.
        controller._request_state.epoch = 1

        def resolve(refs, timeout=None):
            return [ref.result(timeout=timeout) for ref in refs]

        try:
            with patch("xtuner.v1.rl.rollout.controller.ray.get", side_effect=resolve):
                await asyncio.to_thread(controller.pause_generation)
                self.assertFalse(self.session_server.request_gate.snapshot()["active"])
                await asyncio.to_thread(controller.continue_generation, model_step=1)
            response = await self.proxy_client.post("/v1/chat/completions", json=self._payload(session_id="next"))
            self.assertEqual(response.status, 200)
            self.assertEqual(self.on_response.await_args.kwargs["lease"].model_step, 1)
        finally:
            self.upstream_gate.set()

    async def test_failed_recovery_never_allows_publication(self):
        for fault in ("sleep", "trace", "identity", "topology", "unknown"):
            with self.subTest(fault=fault):
                self.session_server.request_gate = RequestGate(wait_timeout=5, initially_open=False)
                self.session_server._forward_timeouts = {}
                await self._publish()
                self.upstream_gate = asyncio.Event()
                self.session_server.request_timeout = 0.1
                response = await self.proxy_client.post("/v1/chat/completions", json=self._payload())
                await response.read()
                if fault == "unknown":
                    self.session_server._forward_timeouts.clear()
                controller = await self._timeout_controller(fault)
                controller._request_state.epoch = 1
                try:
                    with patch(
                        "xtuner.v1.rl.rollout.controller.ray.get",
                        side_effect=lambda refs, timeout=None: [r.result(timeout) for r in refs],
                    ):
                        with self.assertRaises(RuntimeError):
                            await asyncio.to_thread(controller.pause_generation)
                    with self.assertRaisesRegex(RuntimeError, "unsettled"):
                        await self.session_server.request_gate.install_version(2, 1)
                finally:
                    self.upstream_gate.set()


if __name__ == "__main__":
    unittest.main()
