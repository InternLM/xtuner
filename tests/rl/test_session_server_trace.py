import unittest
from unittest.mock import AsyncMock

from aiohttp import ClientConnectionResetError, web
from aiohttp.test_utils import TestClient, TestServer

from xtuner.v1.rl.rollout.session_server import SessionServer, _prepare_stream_response


class TestSessionServerTraceHandling(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.upstream_status = 200
        self.upstream_body = b"{}"

        async def upstream_handler(_request):
            return web.Response(
                status=self.upstream_status,
                content_type="application/json",
                body=self.upstream_body,
            )

        upstream_app = web.Application()
        upstream_app.router.add_route("*", "/{path:.*}", upstream_handler)
        self.upstream_server = TestServer(upstream_app)
        await self.upstream_server.start_server()

        self.session_server = SessionServer.__new__(SessionServer)
        self.session_server.worker_base_url = str(self.upstream_server.make_url("")).rstrip("/")
        self.session_server.request_timeout = 10.0
        self.session_server.read_bufsize = 2**20
        self.session_server.stop_word = "<eos>"
        self.session_server.on_request = AsyncMock(side_effect=lambda body, _fmt, **_kwargs: body)
        self.session_server.on_response = AsyncMock(return_value=None)

        proxy_app = web.Application()
        proxy_app.router.add_route("*", "/{path:.*}", self.session_server._handle_request)
        self.proxy_client = TestClient(TestServer(proxy_app))
        await self.proxy_client.start_server()

    async def asyncTearDown(self):
        await self.proxy_client.close()
        await self.upstream_server.close()

    @staticmethod
    def _request_payload():
        return {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "session_id": "trace-session",
        }

    async def test_malformed_generation_response_fails_closed(self):
        self.upstream_body = b'{"choices": []}'
        self.session_server.on_response = AsyncMock(side_effect=KeyError("choices"))

        response = await self.proxy_client.post("/v1/chat/completions", json=self._request_payload())

        self.assertEqual(response.status, 500)
        self.session_server.on_response.assert_awaited_once()

    async def test_non_object_generation_response_fails_closed(self):
        self.upstream_body = b"[]"

        response = await self.proxy_client.post("/v1/chat/completions", json=self._request_payload())

        self.assertEqual(response.status, 500)
        self.session_server.on_response.assert_not_awaited()

    async def test_non_json_generation_response_fails_closed_with_native_error(self):
        self.upstream_body = b"not-json"

        response = await self.proxy_client.post("/v1/messages", json=self._request_payload())

        self.assertEqual(response.status, 500)
        error = await response.json()
        self.assertEqual(error["type"], "error")
        self.assertEqual(error["error"]["type"], "internal_server_error")
        self.session_server.on_response.assert_not_awaited()

    async def test_upstream_error_is_returned_unchanged(self):
        self.upstream_status = 400
        self.upstream_body = b'{"type":"error","error":{"type":"bad_request","message":"unsupported"}}'

        response = await self.proxy_client.post("/v1/messages", json=self._request_payload())

        self.assertEqual(response.status, 400)
        self.assertEqual(await response.read(), self.upstream_body)
        self.session_server.on_response.assert_not_awaited()


class TestPrepareStreamResponse(unittest.IsolatedAsyncioTestCase):
    async def test_prepared_client_remains_alive(self):
        response = unittest.mock.Mock()
        response.prepare = AsyncMock(return_value=None)
        request = unittest.mock.Mock()

        self.assertTrue(await _prepare_stream_response(response, request))
        response.prepare.assert_awaited_once_with(request)

    async def test_disconnect_before_headers_marks_client_dead(self):
        response = unittest.mock.Mock()
        response.prepare = AsyncMock(side_effect=ClientConnectionResetError("closing transport"))
        request = unittest.mock.Mock()

        self.assertFalse(await _prepare_stream_response(response, request))
        response.prepare.assert_awaited_once_with(request)


if __name__ == "__main__":
    unittest.main()
