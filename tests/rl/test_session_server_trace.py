import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from aiohttp import ClientConnectionResetError, web
from aiohttp.test_utils import TestClient, TestServer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from transformers import PreTrainedTokenizerFast
from xtuner.v1.rl.rollout import session_server as session_server_module
from xtuner.v1.rl.rollout.chat_template import canonicalize_messages_for_chat_template
from xtuner.v1.rl.rollout.session_server import SessionServer
from xtuner.v1.rl.rollout.trace_store import Trie


class TestSessionServerTraceHandling(unittest.IsolatedAsyncioTestCase):
    async def test_raw_anthropic_tokens_keep_output_tool_schema(self):
        server = SessionServer.__new__(SessionServer)
        server.tokenizer = Mock()
        server.tokenizer.apply_chat_template.return_value = "rendered tool prompt"
        server.tokenizer.encode.return_value = [11, 12, 13]
        server.stop_word = "<eos>"
        server.store = SimpleNamespace(search=SimpleNamespace(remote=AsyncMock(return_value=("", []))))
        request = {
            "model": "test-model",
            "session_id": "tools-session",
            "system": "Use the provided tools.",
            "messages": [{"role": "user", "content": "Run pwd"}],
            "tools": [{"name": "Bash", "input_schema": {"type": "object", "properties": {}}}],
            "tool_choice": {"type": "tool", "name": "Bash"},
            "max_tokens": 128,
        }
        forwarded = await server.on_request(request, session_server_module.FMT_ANTHROPIC)
        self.assertEqual(forwarded["input_ids"], [11, 12, 13])
        self.assertEqual(forwarded["messages"], [])
        self.assertNotIn("system", forwarded)
        self.assertEqual(forwarded["tools"], request["tools"])
        self.assertEqual(forwarded["tool_choice"], request["tool_choice"])
        self.assertTrue(forwarded["return_logprob"])
        self.assertEqual(request["messages"], [{"role": "user", "content": "Run pwd"}])

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


class TestAnthropicToolTrace(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _events(parts: list[str], name: str = "Write") -> list[dict]:
        events = [
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "call_1", "name": name, "input": {}},
            }
        ]
        for index, part in enumerate(parts):
            events.append(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": part},
                    "output_ids": [index + 10],
                    "output_token_logprobs": [[-0.25, index + 10]],
                }
            )
        events.extend(
            [
                {"type": "content_block_stop", "index": 0},
                {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}},
                {"type": "message_stop"},
            ]
        )
        return events

    def test_tool_arguments_and_tokens_survive_stream_aggregation(self) -> None:
        cases = [
            (['{"path": "a.py",', '"content": "unfinished'], '{"path": "a.py","content": "unfinished'),
            (['{"command": "printf \\', '"你好'], '{"command": "printf \\"你好'),
            (['{"path": "a.py",', '"content": "ok"}'], {"path": "a.py", "content": "ok"}),
            (["{}"], {}),
            ([""], {}),
        ]
        for parts, expected in cases:
            with self.subTest(parts=parts):
                events = self._events(parts)
                original = copy.deepcopy(events)
                response = SessionServer._parse_anthropic_stream(events)
                self.assertIsNotNone(response)
                self.assertEqual(response["content"][0]["input"], expected)
                self.assertNotIn("_partial_json", response["content"][0])
                self.assertEqual(response["output_ids"], list(range(10, 10 + len(parts))))
                self.assertEqual(response["output_token_logprobs"], [[-0.25, i + 10] for i in range(len(parts))])
                self.assertEqual(response["stop_reason"], "max_tokens")
                self.assertEqual(events, original)

    def test_missing_message_stop_remains_incomplete(self) -> None:
        self.assertIsNone(SessionServer._parse_anthropic_stream(self._events(['{"path": "unfinished'])[:-1]))

    def test_missing_output_ids_still_fails(self) -> None:
        events = self._events(['{"path": "unfinished'])
        for event in events:
            event.pop("output_ids", None)
        with self.assertRaisesRegex(RuntimeError, "without output_ids"):
            SessionServer._parse_anthropic_stream(events)

    async def test_truncated_tool_trace_matches_recorded_message(self) -> None:
        # Exercise the real HF renderer without downloading a model or starting Ray.
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
            chat_template=(
                "{% for message in messages %}{{ message['role'] }}:{{ message.get('content') or '' }}"
                "{% for call in message.get('tool_calls', []) %}{{ call['function'] | tojson }}{% endfor %}"
                "<eos>{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}"
            ),
        )
        request = {
            "model": "test-model",
            "max_tokens": 8192,
            "session_id": "truncated-tool",
            "messages": [{"role": "user", "content": "Run the tool"}],
        }
        for name, parts in (
            ("Write", ['{"path": "a.py",', '"content": "unfinished']),
            ("Bash", ['{"command": "printf \\', '"你好']),
        ):
            with self.subTest(name=name):
                trie = Trie()
                server = SessionServer.__new__(SessionServer)
                server.tokenizer = tokenizer
                server.stop_word = "<eos>"
                server.enable_return_routed_experts = False
                server.store = SimpleNamespace(
                    search=SimpleNamespace(
                        remote=AsyncMock(side_effect=lambda session_id, key, **kwargs: trie.search(key, **kwargs))
                    ),
                    insert=SimpleNamespace(
                        remote=AsyncMock(side_effect=lambda session_id, key, value: trie.insert(key, value))
                    ),
                )
                response = SessionServer._parse_anthropic_stream(self._events(parts, name))
                await server.on_response(response, session_server_module.FMT_ANTHROPIC, request)

                # The agent transcript keeps invalid arguments as the original string,
                # independently of SessionServer's Anthropic -> OpenAI conversion.
                recorded_messages = [
                    *request["messages"],
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": name, "arguments": "".join(parts)},
                            }
                        ],
                    },
                ]
                original = copy.deepcopy(recorded_messages)
                lookup_key = tokenizer.apply_chat_template(
                    canonicalize_messages_for_chat_template(recorded_messages),
                    add_generation_prompt=False,
                    tokenize=False,
                ).rstrip()
                matched, nodes = trie.search(lookup_key, filter_none=True)
                self.assertEqual(matched, lookup_key)
                segment = nodes[-1].value
                self.assertEqual(segment.token_ids, [10, 11])
                self.assertEqual(segment.labels, [10, 11])
                self.assertEqual(segment.logprobs, [-0.25, -0.25])
                self.assertEqual(recorded_messages, original)

                # Matching must not collapse a genuinely different malformed call.
                recorded_messages[-1]["tool_calls"][0]["function"]["arguments"] += "different"
                different_key = tokenizer.apply_chat_template(
                    canonicalize_messages_for_chat_template(recorded_messages),
                    add_generation_prompt=False,
                    tokenize=False,
                ).rstrip()
                self.assertNotEqual(trie.search(different_key, filter_none=True)[0], different_key)


if __name__ == "__main__":
    unittest.main()
