import json
import unittest
from unittest.mock import AsyncMock

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from xtuner.v1.rl.rollout.session_server import (
    FMT_ANTHROPIC,
    FMT_OPENAI,
    SessionServer,
    _detect_format,
    _is_generation_endpoint,
)


class TestEndpointClassification(unittest.TestCase):
    def test_only_generation_endpoints_are_classified(self):
        cases = [
            ("POST", "v1/messages"),
            ("POST", "/proxy/v1/messages?beta=true"),
            ("POST", "/v1/chat/completions"),
        ]
        for method, path in cases:
            with self.subTest(method=method, path=path):
                self.assertTrue(_is_generation_endpoint(method, path))

    def test_non_generation_paths_and_wrong_methods_are_not_classified(self):
        cases = [
            ("POST", "v1/messages/count_tokens"),
            ("POST", "prefix/v1/responses"),
            ("HEAD", "/api/hello"),
            ("GET", "/v1/models"),
            ("GET", "/v1/messages"),
            ("POST", "/terminate"),
            ("POST", "/update_weights"),
            ("POST", "/sleep"),
            ("POST", "/wakeup"),
        ]
        for method, path in cases:
            with self.subTest(method=method, path=path):
                self.assertFalse(_is_generation_endpoint(method, path))

    def test_count_tokens_uses_anthropic_format(self):
        self.assertEqual(_detect_format("v1/messages/count_tokens"), FMT_ANTHROPIC)
        self.assertEqual(_detect_format("v1/messages/batches"), FMT_ANTHROPIC)
        self.assertEqual(_detect_format("v1/chat/completions"), FMT_OPENAI)


class TestSessionServerEndpointHandling(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.upstream_calls = []
        self.upstream_body = b'{"input_tokens": 42}'
        self.upstream_content_type = "application/json"

        async def upstream_handler(request):
            self.upstream_calls.append(
                {
                    "method": request.method,
                    "path": request.path,
                    "query": request.query_string,
                    "headers": dict(request.headers),
                    "body": await request.read(),
                }
            )
            return web.Response(
                status=200,
                content_type=self.upstream_content_type,
                headers={"X-Upstream-Header": "preserved"},
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

    async def test_count_tokens_is_transparent_and_untraced(self):
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "system": "system",
            "tools": [
                {"name": "tool", "input_schema": {"type": "object"}},
                {"type": "web_search_20250305", "name": "web_search"},
            ],
            "session_id": "trace-session",
            "return_token_ids": True,
            "return_logprob": True,
        }

        response = await self.proxy_client.post("/v1/messages/count_tokens?beta=true", json=payload)

        self.assertEqual(response.status, 200)
        self.assertEqual(response.headers["X-Upstream-Header"], "preserved")
        self.assertEqual(await response.read(), b'{"input_tokens": 42}')
        self.assertEqual(len(self.upstream_calls), 1)
        self.assertEqual(self.upstream_calls[0]["headers"]["anthropic-version"], "2023-06-01")
        forwarded = json.loads(self.upstream_calls[0]["body"])
        self.assertEqual(forwarded, {key: value for key, value in payload.items() if key != "session_id"})
        self.assertEqual(self.upstream_calls[0]["query"], "beta=true")
        self.session_server.on_request.assert_not_awaited()
        self.session_server.on_response.assert_not_awaited()

    async def test_generation_still_runs_both_hooks(self):
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
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "session_id": "trace-session",
            "return_token_ids": True,
            "return_logprob": True,
        }

        response = await self.proxy_client.post("/v1/chat/completions", json=payload)

        self.assertEqual(response.status, 200)
        self.session_server.on_request.assert_awaited_once()
        self.assertTrue(self.session_server.on_request.await_args.kwargs["trace_enabled"])
        self.session_server.on_response.assert_awaited_once()

    async def test_generation_evaluation_request_runs_only_request_hook(self):
        self.upstream_body = b'{"choices":[{"message":{"role":"assistant","content":"ok"}}]}'
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "session_id": "evaluation-session",
            "return_token_ids": False,
        }

        response = await self.proxy_client.post("/v1/chat/completions", json=payload)

        self.assertEqual(response.status, 200)
        self.session_server.on_request.assert_awaited_once()
        self.assertFalse(self.session_server.on_request.await_args.kwargs["trace_enabled"])
        self.session_server.on_response.assert_not_awaited()

    async def test_non_generation_endpoints_reach_upstream_without_trace(self):
        hello = await self.proxy_client.head("/api/hello")
        unknown = await self.proxy_client.post("/terminate", json={})
        wrong_method = await self.proxy_client.get("/v1/messages")

        self.assertEqual(hello.status, 200)
        self.assertEqual(unknown.status, 200)
        self.assertEqual(wrong_method.status, 200)
        self.assertEqual([call["path"] for call in self.upstream_calls], ["/api/hello", "/terminate", "/v1/messages"])
        self.session_server.on_request.assert_not_awaited()
        self.session_server.on_response.assert_not_awaited()

    async def test_start_registers_handle_request_directly(self):
        self.session_server.host = "127.0.0.1"
        self.session_server.port = 0
        self.session_server._site = None
        self.session_server._runner = None
        self.session_server._app = None

        await self.session_server.start()
        try:
            route = next(route for route in self.session_server._app.router.routes() if route.method == "*")
            self.assertIs(route.handler.__func__, SessionServer._handle_request)
        finally:
            await self.session_server.stop()

    async def test_non_generation_stream_is_forwarded_without_hooks(self):
        self.upstream_content_type = "text/event-stream"
        self.upstream_body = (
            b"event: ping"
            + bytes([10])
            + b'data: {"choices":[{"delta":{"content":"hello<eos>"}}],"output_ids":[9]}'
            + bytes([10, 10])
            + b"data: [DONE]"
            + bytes([10, 10])
        )

        response = await self.proxy_client.post("/future/stream", json={"stream": True, "session_id": "old-client"})

        self.assertEqual(response.status, 200)
        self.assertEqual(await response.read(), self.upstream_body)
        self.assertEqual(json.loads(self.upstream_calls[0]["body"]), {"stream": True})
        self.session_server.on_request.assert_not_awaited()
        self.session_server.on_response.assert_not_awaited()

    async def test_non_generation_non_object_body_is_forwarded_unchanged(self):
        body = b'[{"session_id":"should-stay-in-array"}]'

        response = await self.proxy_client.post("/future", data=body, headers={"Content-Type": "application/json"})

        self.assertEqual(response.status, 200)
        self.assertEqual(self.upstream_calls[0]["body"], body)
        self.session_server.on_request.assert_not_awaited()
        self.session_server.on_response.assert_not_awaited()

    async def test_non_generation_object_without_session_id_preserves_body_bytes(self):
        body = b' { "stream": false, "value": [1, 2] } '

        response = await self.proxy_client.post("/future", data=body, headers={"Content-Type": "application/json"})

        self.assertEqual(response.status, 200)
        self.assertEqual(self.upstream_calls[0]["body"], body)
        self.session_server.on_request.assert_not_awaited()
        self.session_server.on_response.assert_not_awaited()

    async def test_non_generation_json_does_not_inject_return_fields(self):
        payload = {"messages": [{"role": "user", "content": "hello"}]}

        response = await self.proxy_client.post("/future", json=payload)

        self.assertEqual(response.status, 200)
        forwarded = json.loads(self.upstream_calls[0]["body"])
        self.assertEqual(forwarded, payload)
        self.assertFalse(any(key.startswith("return_") for key in forwarded))
        self.session_server.on_request.assert_not_awaited()
        self.session_server.on_response.assert_not_awaited()

    async def test_responses_keeps_preexisting_rejection(self):
        responses = await self.proxy_client.post("/v1/responses", json={})

        self.assertEqual(responses.status, 501)
        self.assertEqual(self.upstream_calls, [])


if __name__ == "__main__":
    unittest.main()
