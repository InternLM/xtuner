"""SessionAffinityRouter 的真实 HTTP 路由测试。

Good Tests:
- 启动真实 aiohttp 后端和真实 router 线程，只通过 HTTP 观察路由行为。
- 用上游请求计数验证固定绑定、失败关闭和不重放。

Bad Tests:
- 不读取 bindings/retired 等内部计数。
- 不 mock router 的 HTTP 处理路径。
"""

import json
import unittest

from aiohttp import web
from aiohttp.test_utils import TestServer

from xtuner.v1.rl.rollout.affinity_router import SessionAffinityRouter


class _Backend:
    def __init__(self, name: str):
        self.name = name
        self.calls: list[dict] = []
        self.status = 200

    async def handle(self, request: web.Request) -> web.Response:
        body = await request.read()
        self.calls.append({"method": request.method, "path": request.path, "body": body})
        if self.status != 200:
            return web.json_response({"error": {"code": "backend_error"}}, status=self.status)
        return web.json_response({"backend": self.name, "path": request.path})


class TestSessionAffinityRouter(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.backends = [_Backend("a"), _Backend("b")]
        self.servers: list[TestServer] = []
        for backend in self.backends:
            app = web.Application()
            app.router.add_route("*", "/{path:.*}", backend.handle)
            server = TestServer(app)
            await server.start_server()
            self.servers.append(server)
        self.urls = [str(server.make_url("")).rstrip("/") for server in self.servers]
        self.router = SessionAffinityRouter("127.0.0.1")
        self.router.replace_backends(self.urls)

    async def asyncTearDown(self):
        self.router.close()
        for server in self.servers:
            await server.close()

    async def _post(self, session_id: str | None, *, path: str = "/v1/chat/completions"):
        import aiohttp

        payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        if session_id is not None:
            payload["session_id"] = session_id
        async with aiohttp.ClientSession() as client:
            async with client.post(f"{self.router.url}{path}", json=payload) as response:
                return response.status, await response.json()

    async def test_session_stays_on_one_backend_across_requests(self):
        _, first = await self._post("session-0")
        _, second = await self._post("session-0")
        _, other = await self._post("session-1")

        self.assertEqual(first["backend"], second["backend"])
        self.assertEqual(len(self.backends[0].calls) + len(self.backends[1].calls), 3)
        self.assertIn(other["backend"], {"a", "b"})

    async def test_replacing_backends_keeps_existing_bindings(self):
        _, first = await self._post("session-0")
        bound = self.backends[0] if first["backend"] == "a" else self.backends[1]

        self.router.replace_backends(list(reversed(self.urls)))
        _, second = await self._post("session-0")

        self.assertEqual(second["backend"], first["backend"])
        self.assertEqual(len(bound.calls), 2)

    async def test_dead_bound_backend_fails_closed_without_rebinding(self):
        _, first = await self._post("session-0")
        bound_index = 0 if first["backend"] == "a" else 1
        other_backend = self.backends[1 - bound_index]

        self.router.set_backend(self.urls[bound_index], False)
        status, payload = await self._post("session-0")

        self.assertEqual(status, 503)
        self.assertEqual(payload["error"]["code"], "session_backend_unavailable")
        self.assertEqual(other_backend.calls, [])

    async def test_retired_session_gets_410_and_never_reaches_backend(self):
        await self._post("session-0")
        self.router.retire_sessions(["session-0"])
        status, payload = await self._post("session-0")

        self.assertEqual(status, 410)
        self.assertEqual(payload["error"]["code"], "session_retired")
        self.assertEqual(sum(len(backend.calls) for backend in self.backends), 1)

    async def test_missing_session_id_is_rejected(self):
        status, payload = await self._post(None)

        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "valid_session_id_required")

    async def test_upstream_error_is_forwarded_without_replay(self):
        self.backends[0].status = 500
        self.backends[1].status = 500
        status, _ = await self._post("session-0")
        await self._post("session-0")

        self.assertEqual(status, 500)
        self.assertEqual(sum(len(backend.calls) for backend in self.backends), 2)

    async def test_request_body_is_forwarded_unchanged(self):
        await self._post("session-0")
        backend = self.backends[0] if self.backends[0].calls else self.backends[1]
        forwarded = json.loads(backend.calls[0]["body"])
        self.assertEqual(forwarded["session_id"], "session-0")
        self.assertEqual(forwarded["messages"], [{"role": "user", "content": "hi"}])

    async def test_model_listing_is_forwarded_without_a_session(self):
        import aiohttp

        async with aiohttp.ClientSession() as client:
            async with client.get(f"{self.router.url}/v1/models") as response:
                self.assertEqual(response.status, 200)
                payload = await response.json()
        self.assertIn(payload["backend"], {"a", "b"})
        self.assertEqual(payload["path"], "/v1/models")


if __name__ == "__main__":
    unittest.main()
