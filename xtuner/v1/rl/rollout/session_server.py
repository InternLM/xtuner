from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import reduce
from operator import add
from typing import Any, AsyncIterator

import numpy as np
import ray
from aiohttp import ClientConnectionResetError, ClientResponse, ClientSession, ClientTimeout, web

from transformers import AutoTokenizer
from xtuner.v1.utils import get_logger

from .chat_template import canonicalize_messages_for_chat_template
from .trace_store import TokenizedSegment, get_store


@dataclass
class _RolloutRequest:
    original_request: web.Request | None
    method: str
    path: str
    query_string: str
    headers: dict[str, str]
    body: bytes
    data: dict[str, Any] | None
    return_logprob: bool
    return_token_ids: bool
    return_routed_experts: bool
    trace_enabled: bool
    session_id: str | None
    messages: list[dict[str, Any]] | None
    tools: Any | None
    stream: bool


@dataclass
class _WorkerRequest:
    method: str = ""
    target_url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""


class SessionServer:
    """SessionServer is both a server for rollout callers and a client to the
    worker.

    It exposes an OpenAI-compatible HTTP server to rollout code, then forwards
    each request as an HTTP client to an already running worker such as lmdeploy,
    sglang, or vllm. _build_rollout_request parses caller requests, the selected
    response handler writes caller responses, and _WorkerClient owns the worker
    client transport.

    Args:
        worker_base_url (str): The base URL of the real worker, for example "http://127.0.0.1:8000".
        tokenizer_path (str): The path to the tokenizer model.
        host (str): Host for this session server to listen on.
        port (int): Port for this session server to listen on.
        request_timeout (float): Total timeout in seconds for forwarding requests to the worker.
        read_bufsize (int): Buffer limit for line reader in ClientSession.
    """

    def __init__(
        self,
        worker_base_url: str,
        tokenizer_path: str,
        host: str = "127.0.0.1",
        port: int = 8080,
        request_timeout: float = 1200.0,
        read_bufsize: int = 2**26,
    ):
        self.worker_base_url = worker_base_url.rstrip("/")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.host = host
        self.port = port
        self.request_timeout = request_timeout
        self.read_bufsize = read_bufsize
        self.store = get_store()
        self.stop_word = self.tokenizer.eos_token or ""

        self.default_proxy_mode = _DefaultProxyMode(worker_base_url=self.worker_base_url, stop_word=self.stop_word)
        self.trace_store_mode = _TraceStoreMode(
            tokenizer=self.tokenizer,
            trace_store=self.store,
            worker_base_url=self.worker_base_url,
            stop_word=self.stop_word,
        )
        self.client = _WorkerClient(
            request_timeout=self.request_timeout,
            read_bufsize=self.read_bufsize,
        )

        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None

    @property
    def url(self) -> str:
        """The bound URL for the SessionServer.

        Returns:
            str: The HTTP URL this SessionServer listens on.
        """

        return f"http://{self.host}:{self.port}"

    async def start(self) -> None:
        """Start the SessionServer proxy application."""

        if self._site is not None:
            return

        await self.client.start()

        self._app = web.Application()
        self._app.router.add_route("*", "/{path:.*}", self._handle_request)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()
        get_logger().info(f"SessionServer listening on {self.url} (Forwarding to {self.worker_base_url})")

    async def stop(self) -> None:
        """Cleanly stop the SessionServer application."""

        if self._runner:
            await self._runner.cleanup()
        await self.client.stop()
        self._site = None
        self._runner = None
        self._app = None
        get_logger().info("SessionServer stopped.")

    async def _handle_request(self, request: web.Request) -> web.StreamResponse | web.Response:
        """Aiohttp entrypoint for one caller request through the worker
        proxy."""

        try:
            rollout_request = await _build_rollout_request(request)
            mode = self.trace_store_mode if rollout_request.trace_enabled else self.default_proxy_mode
            worker_request = await mode.prepare_worker_request(rollout_request)
        except _PrepareRequestError as exc:
            get_logger().error(exc.message)
            return web.json_response(
                {
                    "message": exc.message,
                    "type": "internal_server_error",
                    "code": exc.status,
                    "object": "error",
                },
                status=exc.status,
            )

        async with self.client.request(worker_request) as worker_response:
            return await mode.handle_worker_response(rollout_request, worker_response)


class SessionServerActor:
    """Ray actor wrapper that owns one SessionServer instance."""

    def __init__(self, worker_base_url: str, tokenizer_path: str, host: str, port: int, request_timeout: float):
        self.worker_base_url = worker_base_url
        self.tokenizer_path = tokenizer_path
        self.host = host
        self.port = port
        self.request_timeout = request_timeout
        self.server: SessionServer | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    async def start(self) -> str:
        if self.server is not None:
            return self.url

        self.server = SessionServer(
            worker_base_url=self.worker_base_url,
            tokenizer_path=self.tokenizer_path,
            host=self.host,
            port=self.port,
            request_timeout=self.request_timeout,
        )
        await self.server.start()
        return self.server.url

    async def stop(self) -> None:
        if self.server is not None:
            await self.server.stop()
            self.server = None


class _WorkerClient:
    """Worker-facing HTTP transport owned by SessionServer."""

    def __init__(
        self,
        *,
        request_timeout: float,
        read_bufsize: int,
    ):
        self.request_timeout = request_timeout
        self.read_bufsize = read_bufsize
        self._session: ClientSession | None = None

    async def start(self) -> None:
        if self._session is not None and not self._session.closed:
            return
        timeout = ClientTimeout(total=self.request_timeout, sock_connect=30)
        self._session = ClientSession(read_bufsize=self.read_bufsize, timeout=timeout)

    async def stop(self) -> None:
        if self._session is None:
            return
        await self._session.close()
        self._session = None

    @asynccontextmanager
    async def request(self, worker_request: _WorkerRequest) -> AsyncIterator[ClientResponse]:
        if self._session is None or self._session.closed:
            raise RuntimeError("Worker client must be started before forwarding requests.")
        async with self._session.request(
            method=worker_request.method,
            url=worker_request.target_url,
            headers=worker_request.headers,
            data=worker_request.body,
        ) as response:
            yield response


async def _build_rollout_request(request: web.Request) -> _RolloutRequest:
    request_body = await request.read()
    request_data = None
    if request_body:
        try:
            request_data = json.loads(request_body)
        except json.JSONDecodeError:
            request_data = None

    return_logprob = False
    return_token_ids = False
    return_routed_experts = True
    trace_enabled = False
    session_id = None
    messages = None
    tools = None
    stream = False
    if request_data is not None:
        return_logprob = request_data.get("return_logprob", request_data.get("logprobs")) is True
        return_token_ids = request_data.get("return_token_ids") is True
        return_routed_experts_value = request_data.get("return_routed_experts")
        return_routed_experts = True if return_routed_experts_value is None else return_routed_experts_value is True
        trace_return_token_ids = request_data.get("return_token_ids")
        trace_enabled = (
            request_data.get("session_id") is not None
            and "messages" in request_data
            and (True if trace_return_token_ids is None else trace_return_token_ids is True)
        )
        session_id = request_data.get("session_id")
        messages = request_data.get("messages")
        tools = request_data.get("tools", None)
        stream = request_data.get("stream") is True

    return _RolloutRequest(
        original_request=request,
        method=request.method,
        path=request.match_info["path"],
        query_string=request.query_string,
        headers=dict(request.headers),
        body=request_body,
        data=request_data,
        return_logprob=return_logprob,
        return_token_ids=return_token_ids,
        return_routed_experts=return_routed_experts,
        trace_enabled=trace_enabled,
        session_id=session_id,
        messages=messages,
        tools=tools,
        stream=stream,
    )


class _BaseProxyMode:
    def __init__(self, *, worker_base_url: str, stop_word: str = ""):
        self.worker_base_url = worker_base_url.rstrip("/")
        self.stop_word = stop_word

    async def prepare_worker_request(self, rollout_request: _RolloutRequest) -> _WorkerRequest:
        worker_request = _WorkerRequest(
            method=rollout_request.method,
            target_url=f"{self.worker_base_url}/{rollout_request.path.lstrip('/')}",
            headers=dict(rollout_request.headers),
        )
        if rollout_request.query_string:
            worker_request.target_url += f"?{rollout_request.query_string}"
        for header in ("Host", "host", "Content-Length", "content-length"):
            worker_request.headers.pop(header, None)

        if rollout_request.data is None:
            worker_request.body = rollout_request.body
            return worker_request

        try:
            worker_payload = await self._build_worker_payload(rollout_request)
        except Exception as exc:
            message = f"SessionServer request hook failed: {type(exc).__name__}: {exc}"
            raise _PrepareRequestError(message) from exc

        worker_request.body = json.dumps(worker_payload).encode("utf-8")
        return worker_request

    async def _build_worker_payload(self, rollout_request: _RolloutRequest) -> dict[str, Any]:
        raise NotImplementedError

    async def handle_worker_response(
        self,
        rollout_request: _RolloutRequest,
        worker_response: ClientResponse,
    ) -> web.StreamResponse | web.Response:
        if rollout_request.stream:
            response, _ = await self._relay_stream_response(
                rollout_request,
                worker_response,
                capture_raw_response=False,
                delay_done=False,
            )
            try:
                await response.write_eof()
            except (ConnectionError, ClientConnectionResetError):
                pass
            return response

        response, _ = await self._read_non_stream_response(rollout_request, worker_response)
        return response

    def _clean_caller_payload(
        self,
        payload: dict[str, Any],
        rollout_request: _RolloutRequest,
    ) -> dict[str, Any]:
        for key, drop in [
            ("output_token_logprobs", not rollout_request.return_logprob),
            ("output_ids", not rollout_request.return_token_ids),
            ("routed_experts", not rollout_request.return_routed_experts),
        ]:
            if drop:
                payload.pop(key, None)
                for choice in payload.get("choices", []):
                    choice.pop(key, None)

        for choice in payload.get("choices", []):
            choice.pop("logprobs", None)
            self._remove_stop_word(choice.get("message"))
            self._remove_stop_word(choice.get("delta"))
        return payload

    async def _relay_stream_response(
        self,
        rollout_request: _RolloutRequest,
        worker_response: ClientResponse,
        *,
        capture_raw_response: bool,
        delay_done: bool,
    ) -> tuple[web.StreamResponse, bytes]:
        assert rollout_request.original_request is not None
        raw_response_chunks: list[bytes] = []
        response = web.StreamResponse(
            status=worker_response.status,
            headers=_filter_response_headers(worker_response.headers),
        )
        await response.prepare(rollout_request.original_request)

        client_alive = True
        async for line in worker_response.content:
            if capture_raw_response:
                raw_response_chunks.append(line)

            if rollout_request.data is not None and line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                try:
                    payload = json.loads(line.decode("utf-8")[6:])
                    payload = self._clean_caller_payload(payload, rollout_request)
                    line = ("data: " + json.dumps(payload) + "\n").encode("utf-8")
                except Exception:
                    pass

            if client_alive and (not delay_done or line.strip() != b"data: [DONE]"):
                try:
                    await response.write(line)
                except (ConnectionError, ClientConnectionResetError):
                    client_alive = False

        return response, b"".join(raw_response_chunks)

    async def _read_non_stream_response(
        self,
        rollout_request: _RolloutRequest,
        worker_response: ClientResponse,
    ) -> tuple[web.Response, bytes]:
        raw_response = await worker_response.read()
        final_raw_response = raw_response
        if rollout_request.data is not None:
            try:
                payload = json.loads(raw_response)
                payload = self._clean_caller_payload(payload, rollout_request)
                final_raw_response = json.dumps(payload).encode("utf-8")
            except Exception:
                pass

        return (
            web.Response(
                status=worker_response.status,
                headers=_filter_response_headers(worker_response.headers),
                body=final_raw_response,
            ),
            raw_response,
        )

    def _remove_stop_word(self, message: Any) -> None:
        if not self.stop_word or not isinstance(message, dict):
            return
        content = message.get("content")
        if isinstance(content, str) and self.stop_word in content:
            message["content"] = content.replace(self.stop_word, "")


class _DefaultProxyMode(_BaseProxyMode):
    async def _build_worker_payload(self, rollout_request: _RolloutRequest) -> dict[str, Any]:
        assert rollout_request.data is not None
        worker_req = {k: v for k, v in rollout_request.data.items() if k not in {"session_id"}}
        if "logprobs" in worker_req:
            worker_req.setdefault("return_logprob", worker_req.pop("logprobs"))
        if worker_req.get("return_logprob") is not True:
            worker_req.pop("top_logprobs", None)
            worker_req["return_logprob"] = False
        worker_req["return_token_ids"] = False
        worker_req.setdefault("return_routed_experts", True)
        return worker_req


class _TraceStoreMode(_BaseProxyMode):
    def __init__(self, *, tokenizer: Any, trace_store: Any, worker_base_url: str, stop_word: str = ""):
        super().__init__(worker_base_url=worker_base_url, stop_word=stop_word)
        self.tokenizer = tokenizer
        self.trace_store = trace_store
        self._lmdeploy_actor: ray.actor.ActorHandle | None = None

    async def _build_worker_payload(self, rollout_request: _RolloutRequest) -> dict[str, Any]:
        assert rollout_request.data is not None
        input_ids = await self._prepare_input_ids(rollout_request)
        return {
            **{
                k: v
                for k, v in rollout_request.data.items()
                if k not in {"session_id", "messages", "logprobs", "top_logprobs"}
            },
            "messages": [],
            "input_ids": input_ids,
            "return_token_ids": True,
            "return_routed_experts": True,
            "return_logprob": True,
            "include_stop_str_in_output": True,
        }

    async def _prepare_input_ids(self, rollout_request: _RolloutRequest) -> list[int]:
        if rollout_request.session_id is None or rollout_request.messages is None:
            raise RuntimeError("Trace-store requests require session_id and messages.")

        prompt_text = self.tokenizer.apply_chat_template(
            canonicalize_messages_for_chat_template(rollout_request.messages),
            tools=rollout_request.tools,
            add_generation_prompt=True,
            tokenize=False,
        )

        prefix, nodes = await self.trace_store.search.remote(rollout_request.session_id, prompt_text, filter_none=True)
        if prefix:
            get_logger().debug(f"Hit prefix cache for session {rollout_request.session_id}")
        delta = prompt_text[len(prefix) :]
        delta_ids = []
        if delta:
            delta_ids = self.tokenizer.encode(delta, add_special_tokens=False)
            await self.trace_store.insert.remote(
                rollout_request.session_id,
                prompt_text,
                TokenizedSegment(text=delta, token_ids=delta_ids),
            )
        return reduce(add, [node.value.token_ids for node in nodes] + [delta_ids])

    async def handle_worker_response(
        self,
        rollout_request: _RolloutRequest,
        worker_response: ClientResponse,
    ) -> web.StreamResponse | web.Response:
        if rollout_request.stream:
            return await self._handle_stream_worker_response(rollout_request, worker_response)

        response, raw_response = await self._read_non_stream_response(rollout_request, worker_response)
        try:
            await self._record_raw_response(rollout_request, raw_response=raw_response, stream=False)
        except Exception as exc:
            return web.json_response(
                {
                    "message": f"SessionServer response failed: {type(exc).__name__}: {exc}",
                    "type": "internal_server_error",
                    "code": 500,
                    "object": "error",
                },
                status=500,
            )
        return response

    async def _handle_stream_worker_response(
        self,
        rollout_request: _RolloutRequest,
        worker_response: ClientResponse,
    ) -> web.StreamResponse:
        response, raw_response = await self._relay_stream_response(
            rollout_request,
            worker_response,
            capture_raw_response=True,
            delay_done=True,
        )
        should_write_done = _has_complete_traceable_sse_response(raw_response)
        error = None
        if should_write_done:
            try:
                await self._record_raw_response(rollout_request, raw_response=raw_response, stream=True)
            except Exception as exc:
                error = exc

        try:
            if error is not None:
                error_payload = {
                    "message": f"SessionServer response failed: {type(error).__name__}: {error}",
                    "type": "internal_server_error",
                    "code": 500,
                    "object": "error",
                }
                await response.write(
                    ("data: " + json.dumps(error_payload, ensure_ascii=False) + "\n\n").encode("utf-8")
                )
            elif should_write_done:
                await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
        except (ConnectionError, ClientConnectionResetError):
            pass
        return response

    async def _record_raw_response(
        self,
        rollout_request: _RolloutRequest,
        *,
        raw_response: bytes,
        stream: bool,
    ) -> None:
        worker_response = self._parse_trace_response(raw=raw_response, stream=stream)
        if worker_response is not None:
            await self._record_response(rollout_request, worker_response)

    def _parse_trace_response(self, *, raw: bytes, stream: bool) -> dict[str, Any] | None:
        if stream:
            response = _parse_sse_to_complete_response(raw)
        else:
            response = json.loads(raw)

        if _is_error_payload(response):
            return None
        return self._validate_and_normalize_trace_response(response)

    def _validate_and_normalize_trace_response(self, response: dict[str, Any]) -> dict[str, Any]:
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError("SessionServer response has no choices; cannot export a training trace.")
        choice = choices[0]
        output_ids = choice.get("output_ids")
        if output_ids is None:
            raise RuntimeError(
                "SessionServer response choice has no output_ids; "
                "cannot export a training trace for this assistant turn."
            )
        self._extract_output_logprobs(choice, output_ids)
        return choice

    async def _record_response(self, rollout_request: _RolloutRequest, worker_response: dict[str, Any]) -> None:
        if rollout_request.session_id is None or rollout_request.messages is None:
            raise RuntimeError("Trace-store responses require session_id and messages.")

        output_ids = worker_response["output_ids"]
        output_logprobs = self._extract_output_logprobs(worker_response, output_ids)
        old_prompt = self.tokenizer.apply_chat_template(
            canonicalize_messages_for_chat_template(rollout_request.messages),
            tools=rollout_request.tools,
            add_generation_prompt=True,
            tokenize=False,
        )
        messages = [*rollout_request.messages, worker_response["message"]]
        new_prompt = (
            self.tokenizer.apply_chat_template(
                canonicalize_messages_for_chat_template(messages),
                tools=rollout_request.tools,
                add_generation_prompt=False,
                tokenize=False,
            )
        ).rstrip()
        assert new_prompt.startswith(old_prompt) and new_prompt.endswith(self.stop_word)

        routed_experts = None
        if worker_response.get("routed_experts") is not None:
            routed_experts = await self._decode_and_split_routed_experts(
                session_id=rollout_request.session_id,
                old_prompt=old_prompt,
                output_ids=output_ids,
                routed_experts=worker_response["routed_experts"],
            )

        await self.trace_store.insert.remote(
            rollout_request.session_id,
            key=new_prompt,
            value=TokenizedSegment(
                text=new_prompt[len(old_prompt) :],
                token_ids=output_ids,
                logprobs=output_logprobs,
                labels=output_ids,
                expert_key=routed_experts,
                length=len(output_ids),
            ),
        )

    async def _decode_and_split_routed_experts(
        self,
        *,
        session_id: str,
        old_prompt: str,
        output_ids: list[int],
        routed_experts: Any,
    ) -> Any:
        if isinstance(routed_experts, str):
            if self._lmdeploy_actor is None:
                self._lmdeploy_actor = ray.get_actor("shared_store", namespace="lmdeploy")
            assert self._lmdeploy_actor is not None, "LMDeploy actor should be available in the shared store."
            routed_experts = await self._lmdeploy_actor.get.remote(routed_experts)
        raw_routed_expert = np.asarray(routed_experts)
        if len(raw_routed_expert) > 0:
            num_layers = raw_routed_expert.shape[1]
            topk_experts = raw_routed_expert.shape[2]
            dummy_expert = np.full((1, num_layers, topk_experts), 0, dtype=raw_routed_expert.dtype)
            raw_routed_expert = np.concatenate([dummy_expert, raw_routed_expert], axis=0)

        _, nodes = await self.trace_store.search.remote(session_id, old_prompt, filter_none=True)
        if not nodes:
            return ray.put(raw_routed_expert)

        delta_node_val: TokenizedSegment = nodes[-1].value
        delta_len = len(delta_node_val.token_ids)
        prefix_len = sum(len(n.value.token_ids) for n in nodes[:-1])
        assert prefix_len + delta_len + len(output_ids) == len(raw_routed_expert)

        delta_expert = raw_routed_expert[prefix_len : prefix_len + delta_len]
        response_expert = raw_routed_expert[prefix_len + delta_len :]
        if delta_len > 0:
            delta_node_val.expert_key = ray.put(delta_expert)
            await self.trace_store.insert.remote(session_id, old_prompt, delta_node_val)
        return ray.put(response_expert)

    def _extract_output_logprobs(self, choice: dict, output_token_ids: list[int]) -> list[float]:
        if not output_token_ids:
            return []

        output_token_logprobs = choice.get("output_token_logprobs")
        if output_token_logprobs is None:
            raise RuntimeError(
                "SessionServer response choice has no output_token_logprobs; "
                "the return_logprob protocol is required for training traces."
            )

        logprob_token_ids = [item[1] for item in output_token_logprobs]
        if logprob_token_ids != output_token_ids:
            raise RuntimeError(
                "SessionServer response choice has mismatched output_token_logprobs: "
                f"output_ids_len={len(output_token_ids)}, logprob_ids_len={len(logprob_token_ids)}"
            )
        return [item[0] for item in output_token_logprobs]


def _parse_sse_to_complete_response(raw: bytes) -> dict[str, Any]:
    text = raw.decode("utf-8", errors="replace")
    events = []
    saw_done = False
    for line in text.split("\n"):
        line = line.strip()
        if line == "data: [DONE]":
            saw_done = True
            continue
        if line.startswith("data: "):
            event = json.loads(line[6:])
            if _is_error_payload(event):
                raise RuntimeError(f"Upstream SSE stream returned error: {json.dumps(event, ensure_ascii=False)}")
            events.append(event)

    if not events:
        return {}
    if not any(event.get("choices") for event in events):
        raise RuntimeError(f"Upstream SSE stream ended without choices: {json.dumps(events, ensure_ascii=False)}")

    message: dict[str, Any] = {"choices": [{"message": {"role": "assistant", "content": ""}}]}
    content_parts: list[str] = []
    tool_calls_map: dict[int, dict[str, Any]] = {}
    usage: dict[str, Any] = {}

    for event in events:
        if event.get("id") and "id" not in message:
            message["id"] = event["id"]
        if event.get("model"):
            message["model"] = event["model"]

        for choice in event.get("choices", []):
            if choice.get("finish_reason") == "error":
                raise RuntimeError(f"Upstream SSE choice finished with error: {json.dumps(event, ensure_ascii=False)}")
            delta = choice.get("delta", {})

            if delta.get("content"):
                content_parts.append(delta["content"])
            if choice.get("output_ids") is not None:
                assistant_choice = message["choices"][0]
                assistant_choice.setdefault("output_ids", []).extend(choice["output_ids"])
            if choice.get("routed_experts") is not None:
                message["choices"][0]["routed_experts"] = choice["routed_experts"]
            if choice.get("output_token_logprobs") is not None:
                assistant_choice = message["choices"][0]
                assistant_choice.setdefault("output_token_logprobs", []).extend(choice["output_token_logprobs"])
            if delta.get("reasoning_content"):
                assistant_msg = message["choices"][0]["message"]
                assistant_msg["reasoning_content"] = (
                    assistant_msg.get("reasoning_content", "") + delta["reasoning_content"]
                )

            for tool_call_delta in delta.get("tool_calls") or []:
                idx = tool_call_delta.get("index", 0)
                tool_call = tool_calls_map.setdefault(
                    idx,
                    {
                        "id": tool_call_delta.get("id", ""),
                        "type": tool_call_delta.get("type", "function"),
                        "function": {"name": "", "arguments": ""},
                    },
                )
                function_delta = tool_call_delta.get("function", {})
                if function_delta.get("name"):
                    tool_call["function"]["name"] += function_delta["name"]
                if function_delta.get("arguments"):
                    tool_call["function"]["arguments"] += function_delta["arguments"]

            if choice.get("finish_reason"):
                message["choices"][0]["finish_reason"] = choice["finish_reason"]

        if event.get("usage") is not None:
            usage = event["usage"]

    assistant_message = message["choices"][0]["message"]
    assistant_message["content"] = "".join(content_parts)
    if tool_calls_map:
        assistant_message["tool_calls"] = [tool_calls_map[i] for i in sorted(tool_calls_map)]
    if usage:
        message["usage"] = usage

    assistant_choice = message["choices"][0]
    if not saw_done:
        raise RuntimeError("Upstream SSE stream ended without [DONE].")
    if not assistant_choice.get("finish_reason"):
        raise RuntimeError("Upstream SSE stream ended without terminal finish_reason.")
    if assistant_choice.get("output_ids") is None:
        raise RuntimeError("Upstream SSE stream ended without output_ids.")

    return message


def _has_complete_traceable_sse_response(raw: bytes) -> bool:
    text = raw.decode("utf-8", errors="replace")
    has_choices = False
    saw_done = False
    saw_terminal_finish = False
    for line in text.split("\n"):
        line = line.strip()
        if line == "data: [DONE]":
            saw_done = True
            continue
        if line.startswith("data: "):
            try:
                event = json.loads(line[6:])
            except json.JSONDecodeError:
                return False
            if _is_error_payload(event):
                return False
            if event.get("choices"):
                if any(choice.get("finish_reason") == "error" for choice in event.get("choices", [])):
                    return False
                if any(choice.get("finish_reason") for choice in event.get("choices", [])):
                    saw_terminal_finish = True
                has_choices = True
    return has_choices and saw_done and saw_terminal_finish


def _is_error_payload(payload: dict[str, Any]) -> bool:
    return payload.get("error") is not None or payload.get("type") == "error" or payload.get("object") == "error"


class _PrepareRequestError(RuntimeError):
    def __init__(self, message: str, *, status: int = 500):
        self.message = message
        self.status = status
        super().__init__(message)


def _filter_response_headers(headers: Any) -> dict[str, str]:
    return {
        k: v
        for k, v in headers.items()
        if k.lower() not in ("transfer-encoding", "content-length", "content-encoding")
    }
