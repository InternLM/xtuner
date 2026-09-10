import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from aiohttp import ClientConnectionResetError, web
from aiohttp.test_utils import TestClient, TestServer

from xtuner.v1.rl.rollout import session_server as session_server_module
from xtuner.v1.rl.rollout.session_server import SessionServer


class TestSessionServerTraceHandling(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.upstream_status = 200
        self.upstream_body = b'{"choices":[{"message":{"role":"assistant","content":"ok"}}]}'
        self.upstream_content_type = "application/json"
        self.upstream_headers = {"X-Upstream-Header": "preserved"}
        self.upstream_calls = 0

        async def upstream_handler(_request):
            self.upstream_calls += 1
            return web.Response(
                status=self.upstream_status,
                content_type=self.upstream_content_type,
                headers=self.upstream_headers,
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
    def _request_payload(stream=False):
        return {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "session_id": "trace-session",
            "return_token_ids": True,
            "stream": stream,
        }

    @staticmethod
    def _openai_stream(*, include_done=True):
        body = (
            b'data: {"id":"completion","choices":[{"delta":{"content":"ok"},'
            b'"output_ids":[1],"output_token_logprobs":[[-0.1,1]],"finish_reason":"stop"}]}\n\n'
        )
        if include_done:
            body += b"data: [DONE]\n\n"
        return body

    async def test_successful_training_response_runs_response_hook(self):
        self.upstream_body = json.dumps(
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "output_ids": [1],
                        "output_token_logprobs": [[-0.1, 1]],
                    }
                ]
            }
        ).encode()

        response = await self.proxy_client.post("/v1/chat/completions", json=self._request_payload())

        self.assertEqual(response.status, 200)
        self.session_server.on_request.assert_awaited_once()
        self.session_server.on_response.assert_awaited_once()

    async def test_successful_training_stream_sends_done_after_trace(self):
        self.upstream_content_type = "text/event-stream"
        self.upstream_body = self._openai_stream()

        response = await self.proxy_client.post("/v1/chat/completions", json=self._request_payload(stream=True))

        self.assertEqual(response.status, 200)
        body = await response.read()
        self.assertIn(b'"content": "ok"', body)
        self.assertIn(b"data: [DONE]\n\n", body)
        self.session_server.on_response.assert_awaited_once()

    async def test_upstream_error_json_is_returned_unchanged(self):
        self.upstream_body = b'{"object":"error","message":"upstream failed"}'

        response = await self.proxy_client.post("/v1/chat/completions", json=self._request_payload())

        self.assertEqual(response.status, 200)
        self.assertEqual(await response.read(), self.upstream_body)
        self.session_server.on_response.assert_not_awaited()

    async def test_upstream_error_stream_is_not_wrapped_again(self):
        self.upstream_content_type = "text/event-stream"
        self.upstream_body = b'data: {"error":{"type":"server_error","message":"upstream failed"}}\n\ndata: [DONE]\n\n'

        response = await self.proxy_client.post("/v1/chat/completions", json=self._request_payload(stream=True))
        body = await response.read()

        self.assertEqual(response.status, 200)
        self.assertEqual(body, self.upstream_body)
        self.assertNotIn(b"SessionServer", body)
        self.session_server.on_response.assert_not_awaited()

    async def test_non_success_json_is_forwarded_without_cleaning_or_hook(self):
        self.upstream_status = 400
        self.upstream_body = json.dumps(
            {
                "choices": [{"message": {"role": "assistant", "content": "bad<eos>"}, "output_ids": [1]}],
                "routed_experts": [[0]],
            }
        ).encode()

        response = await self.proxy_client.post("/v1/chat/completions", json=self._request_payload())

        self.assertEqual(response.status, 400)
        self.assertEqual(await response.read(), self.upstream_body)
        self.assertEqual(response.headers["X-Upstream-Header"], "preserved")
        self.session_server.on_response.assert_not_awaited()

    async def test_non_success_stream_is_forwarded_byte_for_byte(self):
        self.upstream_status = 500
        self.upstream_content_type = "text/event-stream"
        self.upstream_body = (
            b'data: {"choices":[{"delta":{"content":"bad<eos>"},"output_ids":[1]}]}\n\ndata: [DONE]\n\n'
        )

        response = await self.proxy_client.post("/v1/chat/completions", json=self._request_payload(stream=True))

        self.assertEqual(response.status, 500)
        self.assertEqual(await response.read(), self.upstream_body)
        self.session_server.on_response.assert_not_awaited()

    async def test_malformed_or_non_object_success_response_fails_closed(self):
        for body in (b"not-json", b"[]"):
            with self.subTest(body=body):
                self.upstream_status = 200
                self.upstream_content_type = "application/json"
                self.upstream_body = body
                self.session_server.on_response.reset_mock()

                response = await self.proxy_client.post("/v1/chat/completions", json=self._request_payload())
                error = await response.json()

                self.assertEqual(response.status, 500)
                self.assertEqual(error["object"], "error")
                self.session_server.on_response.assert_not_awaited()

    async def test_response_hook_failure_returns_native_error(self):
        self.session_server.on_response = AsyncMock(side_effect=RuntimeError("missing routed experts"))

        response = await self.proxy_client.post("/v1/chat/completions", json=self._request_payload())
        error = await response.json()

        self.assertEqual(response.status, 500)
        self.assertEqual(error["object"], "error")
        self.assertIn("missing routed experts", error["message"])
        self.session_server.on_response.assert_awaited_once()

    async def test_response_hook_failure_stream_sends_error_without_done(self):
        self.upstream_content_type = "text/event-stream"
        self.upstream_body = self._openai_stream()
        self.session_server.on_response = AsyncMock(side_effect=RuntimeError("trace write failed"))

        response = await self.proxy_client.post("/v1/chat/completions", json=self._request_payload(stream=True))
        body = await response.read()

        self.assertEqual(response.status, 200)
        self.assertIn(b'"object": "error"', body)
        self.assertNotIn(b"data: [DONE]", body)
        self.session_server.on_response.assert_awaited_once()

    async def test_incomplete_stream_sends_error_without_done(self):
        self.upstream_content_type = "text/event-stream"
        self.upstream_body = self._openai_stream(include_done=False)

        response = await self.proxy_client.post("/v1/chat/completions", json=self._request_payload(stream=True))
        body = await response.read()

        self.assertEqual(response.status, 200)
        self.assertIn(b'"object": "error"', body)
        self.assertNotIn(b"data: [DONE]", body)
        self.session_server.on_response.assert_not_awaited()

    async def test_disconnect_before_prepare_still_completes_trace(self):
        self.upstream_content_type = "text/event-stream"
        self.upstream_body = self._openai_stream()

        class Request:
            is_proxy = True
            method = "POST"
            match_info = {"path": "v1/chat/completions"}
            query_string = ""
            headers = {"Content-Type": "application/json"}

            async def read(self):
                return json.dumps(TestSessionServerTraceHandling._request_payload(stream=True)).encode()

        class FakeStreamResponse:
            def __init__(self, status, headers):
                self.status = status
                self.headers = headers
                self.prepare_calls = []

            async def prepare(self, request):
                self.prepare_calls.append(request)
                raise ClientConnectionResetError("downstream closed")

            async def write_eof(self):
                pass

        fake_web = SimpleNamespace(
            StreamResponse=FakeStreamResponse,
            Response=web.Response,
            json_response=web.json_response,
        )
        with patch.object(session_server_module, "web", fake_web):
            response = await self.session_server._handle_request(Request())

        self.assertEqual(response.status, 200)
        self.assertEqual(len(response.prepare_calls), 1)
        self.session_server.on_response.assert_awaited_once()

    async def test_disconnect_midstream_still_completes_trace(self):
        self.upstream_content_type = "text/event-stream"
        self.upstream_body = self._openai_stream()

        class Request:
            is_proxy = True
            method = "POST"
            match_info = {"path": "v1/chat/completions"}
            query_string = ""
            headers = {"Content-Type": "application/json"}

            async def read(self):
                return json.dumps(TestSessionServerTraceHandling._request_payload(stream=True)).encode()

        class FakeStreamResponse:
            def __init__(self, status, headers):
                self.status = status
                self.headers = headers
                self.prepare_calls = []
                self.write_calls = []

            async def prepare(self, request):
                self.prepare_calls.append(request)

            async def write(self, data):
                self.write_calls.append(data)
                raise ClientConnectionResetError("downstream closed")

            async def write_eof(self):
                pass

        fake_web = SimpleNamespace(
            StreamResponse=FakeStreamResponse,
            Response=web.Response,
            json_response=web.json_response,
        )
        with patch.object(session_server_module, "web", fake_web):
            response = await self.session_server._handle_request(Request())

        self.assertEqual(response.status, 200)
        self.assertEqual(len(response.prepare_calls), 1)
        self.assertEqual(len(response.write_calls), 1)
        self.session_server.on_response.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
