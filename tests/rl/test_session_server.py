import json
import unittest
from types import SimpleNamespace

from xtuner.v1.rl.rollout import session_server as session_server_mod


class _RemoteMethod:
    def __init__(self, func):
        self._func = func

    async def remote(self, *args, **kwargs):
        return self._func(*args, **kwargs)


class _FakeTokenizer:
    eos_token = "<eos>"

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=False, tokenize=False):
        del tools, tokenize
        rendered = []
        for message in messages:
            role = message["role"]
            content = message.get("content") or ""
            if role == "user":
                rendered.append(f"user:{content}\n")
            elif role == "assistant":
                rendered.append(f"assistant:{content}")
            else:
                rendered.append(f"{role}:{content}\n")
        text = "".join(rendered)
        if add_generation_prompt:
            text += "assistant:"
        else:
            text += self.eos_token
        return text

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(ch) for ch in text]


class _FakeTraceStore:
    def __init__(self, search_results):
        self.search_results = list(search_results)
        self.search_calls = []
        self.inserts = []
        self.search = _RemoteMethod(self._search)
        self.insert = _RemoteMethod(self._insert)

    def _search(self, session_id, text, filter_none=False):
        self.search_calls.append((session_id, text, filter_none))
        if self.search_results:
            return self.search_results.pop(0)
        return "", []

    def _insert(self, *args, **kwargs):
        self.inserts.append((args, kwargs))


class _FakeWorkerResponse:
    def __init__(self, payload, *, status=200):
        self.status = status
        self.headers = {"content-type": "application/json", "content-length": "999"}
        self._payload = payload

    async def read(self):
        return json.dumps(self._payload).encode("utf-8")


class TestSessionServerModes(unittest.IsolatedAsyncioTestCase):
    def _rollout_request(self, data, *, stream=False, trace_enabled=False):
        return session_server_mod._RolloutRequest(
            original_request=None,
            method="POST",
            path="/v1/chat/completions",
            query_string="",
            headers={},
            body=json.dumps(data).encode("utf-8"),
            data=data,
            return_logprob=False,
            return_token_ids=False,
            return_routed_experts=True,
            trace_enabled=trace_enabled,
            session_id=data.get("session_id"),
            messages=data.get("messages"),
            tools=data.get("tools"),
            stream=stream,
        )

    async def test_default_proxy_mode_preserves_default_proxy_options(self):
        caller_data = {
            "session_id": "session-1",
            "messages": [{"role": "user", "content": "hello"}],
            "logprobs": False,
            "top_logprobs": 2,
            "stream": False,
        }
        rollout_request = self._rollout_request(caller_data)
        mode = session_server_mod._DefaultProxyMode(
            worker_base_url="http://worker",
            stop_word="",
        )

        worker_request = await mode.prepare_worker_request(rollout_request)
        worker_payload = json.loads(worker_request.body)

        self.assertEqual(worker_request.method, "POST")
        self.assertEqual(worker_request.target_url, "http://worker/v1/chat/completions")
        self.assertNotIn("session_id", worker_payload)
        self.assertNotIn("top_logprobs", worker_payload)
        self.assertEqual(worker_payload["return_logprob"], False)
        self.assertEqual(worker_payload["return_token_ids"], False)
        self.assertEqual(worker_payload["return_routed_experts"], True)
        self.assertIn("session_id", caller_data)

    def test_clean_caller_payload_uses_original_caller_options(self):
        rollout_request = self._rollout_request({})
        rollout_request.return_routed_experts = False
        payload = {
            "output_ids": [1],
            "output_token_logprobs": [[-0.1, 1]],
            "routed_experts": "top-level-experts",
            "choices": [
                {
                    "message": {"content": "hello<eos>"},
                    "delta": {"content": "world<eos>"},
                    "output_ids": [1],
                    "output_token_logprobs": [[-0.1, 1]],
                    "routed_experts": "choice-experts",
                    "logprobs": {"content": []},
                }
            ],
        }
        mode = session_server_mod._DefaultProxyMode(worker_base_url="http://worker", stop_word="<eos>")

        cleaned = mode._clean_caller_payload(payload, rollout_request)

        self.assertNotIn("output_ids", cleaned)
        self.assertNotIn("output_token_logprobs", cleaned)
        self.assertNotIn("routed_experts", cleaned)
        choice = cleaned["choices"][0]
        self.assertNotIn("output_ids", choice)
        self.assertNotIn("output_token_logprobs", choice)
        self.assertNotIn("routed_experts", choice)
        self.assertNotIn("logprobs", choice)
        self.assertEqual(choice["message"]["content"], "hello")
        self.assertEqual(choice["delta"]["content"], "world")

    def test_parse_sse_to_complete_response_builds_complete_message(self):
        raw = b"".join(
            [
                b'data: {"id":"cmpl-1","model":"m","choices":[{"delta":{"content":"he"},'
                b'"output_ids":[10],"output_token_logprobs":[[-0.1,10]]}]}\n\n',
                b'data: {"choices":[{"delta":{"content":"llo","reasoning_content":"r"},'
                b'"output_ids":[11],"output_token_logprobs":[[-0.2,11]],'
                b'"routed_experts":"expert-key","finish_reason":"stop"}],"usage":{"completion_tokens":2}}\n\n',
                b"data: [DONE]\n\n",
            ]
        )
        response = session_server_mod._parse_sse_to_complete_response(raw)

        self.assertEqual(response["id"], "cmpl-1")
        self.assertEqual(response["model"], "m")
        choice = response["choices"][0]
        self.assertEqual(choice["message"]["content"], "hello")
        self.assertEqual(choice["message"]["reasoning_content"], "r")
        self.assertEqual(choice["output_ids"], [10, 11])
        self.assertEqual(choice["output_token_logprobs"], [[-0.1, 10], [-0.2, 11]])
        self.assertEqual(choice["routed_experts"], "expert-key")
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertEqual(response["usage"], {"completion_tokens": 2})

    async def test_trace_store_request_prepares_incremental_input_ids_and_forces_trace_options(self):
        tokenizer = _FakeTokenizer()
        messages = [{"role": "user", "content": "hello"}]
        prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prefix = "user:hello\n"
        prefix_node = SimpleNamespace(
            value=session_server_mod.TokenizedSegment(text=prefix, token_ids=[101, 102])
        )
        trace_store = _FakeTraceStore(search_results=[(prefix, [prefix_node])])
        mode = session_server_mod._TraceStoreMode(
            tokenizer=tokenizer,
            trace_store=trace_store,
            worker_base_url="http://worker",
            stop_word=tokenizer.eos_token,
        )
        rollout_request = self._rollout_request(
            {
                "session_id": "session-1",
                "messages": messages,
                "logprobs": False,
                "top_logprobs": 5,
                "return_token_ids": True,
                "temperature": 0.7,
            },
            trace_enabled=True,
        )

        worker_request = await mode.prepare_worker_request(rollout_request)
        worker_payload = json.loads(worker_request.body)

        delta = prompt_text[len(prefix) :]
        self.assertEqual(worker_payload["input_ids"], [101, 102, *tokenizer.encode(delta)])
        self.assertEqual(worker_payload["messages"], [])
        self.assertEqual(worker_payload["return_token_ids"], True)
        self.assertEqual(worker_payload["return_routed_experts"], True)
        self.assertEqual(worker_payload["return_logprob"], True)
        self.assertEqual(worker_payload["include_stop_str_in_output"], True)
        self.assertEqual(worker_payload["temperature"], 0.7)
        self.assertNotIn("session_id", worker_payload)
        self.assertNotIn("top_logprobs", worker_payload)
        self.assertEqual(trace_store.search_calls, [("session-1", prompt_text, True)])
        insert_args, insert_kwargs = trace_store.inserts[0]
        self.assertEqual(insert_args[:2], ("session-1", prompt_text))
        self.assertEqual(insert_kwargs, {})
        inserted_segment = insert_args[2]
        self.assertEqual(inserted_segment.text, delta)
        self.assertEqual(inserted_segment.token_ids, tokenizer.encode(delta))

    async def test_trace_store_response_records_raw_worker_tokens_while_cleaning_caller_response(self):
        tokenizer = _FakeTokenizer()
        trace_store = _FakeTraceStore(search_results=[])
        mode = session_server_mod._TraceStoreMode(
            tokenizer=tokenizer,
            trace_store=trace_store,
            worker_base_url="http://worker",
            stop_word=tokenizer.eos_token,
        )
        messages = [{"role": "user", "content": "hello"}]
        rollout_request = self._rollout_request(
            {"session_id": "session-1", "messages": messages},
            trace_enabled=True,
        )
        worker_payload = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "world"},
                    "output_ids": [11, 12],
                    "output_token_logprobs": [[-0.1, 11], [-0.2, 12]],
                    "logprobs": {"content": []},
                }
            ]
        }

        response = await mode.handle_worker_response(rollout_request, _FakeWorkerResponse(worker_payload))

        caller_payload = json.loads(response.body)
        self.assertNotIn("output_ids", caller_payload["choices"][0])
        self.assertNotIn("output_token_logprobs", caller_payload["choices"][0])
        self.assertNotIn("logprobs", caller_payload["choices"][0])
        insert_args, insert_kwargs = trace_store.inserts[0]
        self.assertEqual(insert_args, ("session-1",))
        self.assertEqual(insert_kwargs["key"], "user:hello\nassistant:world<eos>")
        inserted_segment = insert_kwargs["value"]
        self.assertEqual(inserted_segment.text, "world<eos>")
        self.assertEqual(inserted_segment.token_ids, [11, 12])
        self.assertEqual(inserted_segment.labels, [11, 12])
        self.assertEqual(inserted_segment.logprobs, [-0.1, -0.2])

    async def test_trace_store_response_returns_error_when_required_trace_fields_are_missing(self):
        tokenizer = _FakeTokenizer()
        trace_store = _FakeTraceStore(search_results=[])
        mode = session_server_mod._TraceStoreMode(
            tokenizer=tokenizer,
            trace_store=trace_store,
            worker_base_url="http://worker",
            stop_word=tokenizer.eos_token,
        )
        rollout_request = self._rollout_request(
            {
                "session_id": "session-1",
                "messages": [{"role": "user", "content": "hello"}],
            },
            trace_enabled=True,
        )

        response = await mode.handle_worker_response(
            rollout_request,
            _FakeWorkerResponse({"choices": [{"message": {"role": "assistant", "content": "world"}}]}),
        )

        error_payload = json.loads(response.text)
        self.assertEqual(response.status, 500)
        self.assertEqual(error_payload["object"], "error")
        self.assertIn("no output_ids", error_payload["message"])
        self.assertEqual(trace_store.inserts, [])

    def test_parse_sse_to_complete_response_merges_tool_call_deltas(self):
        raw = b"".join(
            [
                b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1",'
                b'"type":"function","function":{"name":"get_"}}]}}]}\n\n',
                b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
                b'"function":{"name":"weather","arguments":"{\\"city\\""}}]},'
                b'"output_ids":[10],"output_token_logprobs":[[-0.1,10]]}]}\n\n',
                b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
                b'"function":{"arguments":":\\"HZ\\"}"}}]},'
                b'"output_ids":[11],"output_token_logprobs":[[-0.2,11]],'
                b'"finish_reason":"tool_calls"}]}\n\n',
                b"data: [DONE]\n\n",
            ]
        )

        response = session_server_mod._parse_sse_to_complete_response(raw)

        choice = response["choices"][0]
        tool_call = choice["message"]["tool_calls"][0]
        self.assertEqual(tool_call["id"], "call-1")
        self.assertEqual(tool_call["type"], "function")
        self.assertEqual(tool_call["function"]["name"], "get_weather")
        self.assertEqual(tool_call["function"]["arguments"], '{"city":"HZ"}')
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(choice["output_ids"], [10, 11])

    def test_sse_traceability_rejects_incomplete_or_error_streams(self):
        complete = (
            b'data: {"choices":[{"delta":{"content":"ok"},"output_ids":[1],'
            b'"finish_reason":"stop"}]}\n\n'
            b"data: [DONE]\n\n"
        )
        missing_done = b'data: {"choices":[{"delta":{"content":"ok"},"output_ids":[1],"finish_reason":"stop"}]}\n\n'
        error_choice = (
            b'data: {"choices":[{"delta":{"content":"bad"},"finish_reason":"error"}]}\n\n'
            b"data: [DONE]\n\n"
        )
        upstream_error = b'data: {"object":"error","message":"bad"}\n\n'

        self.assertTrue(session_server_mod._has_complete_traceable_sse_response(complete))
        self.assertFalse(session_server_mod._has_complete_traceable_sse_response(missing_done))
        self.assertFalse(session_server_mod._has_complete_traceable_sse_response(error_choice))
        self.assertFalse(session_server_mod._has_complete_traceable_sse_response(upstream_error))
        with self.assertRaisesRegex(RuntimeError, "without \\[DONE\\]"):
            session_server_mod._parse_sse_to_complete_response(missing_done)
        with self.assertRaisesRegex(RuntimeError, "finished with error"):
            session_server_mod._parse_sse_to_complete_response(error_choice)
