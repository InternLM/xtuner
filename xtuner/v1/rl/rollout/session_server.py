import copy
import json
from functools import reduce
from http import HTTPStatus
from operator import add
from typing import Any, List, Optional

import numpy as np
import ray
from aiohttp import ClientConnectionResetError, ClientSession, ClientTimeout, web
from transformers import AutoTokenizer

from xtuner.v1.utils import get_logger

from .chat_template import canonicalize_messages_for_chat_template
from .trace_store import TokenizedSegment, get_store

FMT_OPENAI = "openai"
FMT_ANTHROPIC = "anthropic"

# Fields the SessionServer consumes locally and never forwards upstream.
_SESSION_SERVER_ONLY_KEYS = {"session_id"}


def _detect_format(req_path: str) -> str:
    if req_path.endswith("/messages") or "/v1/messages" in req_path:
        return FMT_ANTHROPIC
    return FMT_OPENAI


def _is_error_payload(payload: dict) -> bool:
    return payload.get("error") is not None or payload.get("type") == "error" or payload.get("object") == "error"


def _lmdeploy_error_payload(message: str, status: int = 500, error_type: str = "internal_server_error") -> dict:
    return {
        "message": message,
        "type": error_type,
        "code": status,
        "object": "error",
    }


def _bool_request_value(value: Any, default: bool = False) -> bool:
    """Coerce an opaque request-body value (bool / int / str / None) to bool.

    Mirrors how lmdeploy validates flag-style fields — accepts ``"true"``,
    ``"1"`` etc. as truthy and ``"false"``, ``"0"`` as falsy.
    """
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def _request_uses_trace_store(req_body: dict) -> bool:
    """Whether this request should drive the trace_store / token-in-token-out path.

    Evaluate-mode callers opt out by setting ``return_token_ids=False`` on the
    request body; tracing then becomes a passthrough proxy with parameter-name
    hygiene (``logprobs`` → ``return_logprob``) and the upstream worker handles
    its own tokenization.
    """
    return _bool_request_value(req_body.get("return_token_ids"), True)


def _extract_output_logprobs(output_token_logprobs: Optional[list], output_token_ids: list[int]) -> list[float]:
    if not output_token_ids:
        return []

    if output_token_logprobs is None:
        raise RuntimeError(
            "SessionServer response has no output_token_logprobs; "
            "the return_logprob protocol is required for training traces."
        )

    logprob_token_ids = [item[1] for item in output_token_logprobs]
    if logprob_token_ids != output_token_ids:
        raise RuntimeError(
            "SessionServer response has mismatched output_token_logprobs: "
            f"output_ids_len={len(output_token_ids)}, logprob_ids_len={len(logprob_token_ids)}"
        )
    return [item[0] for item in output_token_logprobs]


def _maybe_json_loads(value):
    """Parse a JSON string, falling back to the original value on error."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _normalize_tool_call_arguments(messages) -> None:
    """In-place: parse ``tool_calls[].function.arguments`` strings into dicts.

    lmdeploy stringifies tool-call arguments; the standard OpenAI/chat_template
    path expects dicts. Normalize so both paths share one shape.
    """
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function")
            if isinstance(fn, dict) and "arguments" in fn:
                fn["arguments"] = _maybe_json_loads(fn["arguments"])


def _strip_stop_word(content, stop_word: str):
    """Remove ``stop_word`` from a message ``content`` field (any OpenAI shape).

    OpenAI ``content`` accepts both a plain string and a list of typed parts
    (``[{"type": "text", "text": "..."}, ...]``); the latter shows up after we
    pass anthropic messages through ``to_openai_messages`` — multiple anthropic
    text blocks inside one assistant turn become multiple ``text`` parts. The
    stop_word can land in any of those ``text`` fields, so we walk both shapes.

    Returns ``(content, modified)``. List parts are mutated in place; for str
    a new string is returned (the original is left untouched).
    """
    if not stop_word or content is None:
        return content, False
    if isinstance(content, str):
        if stop_word in content:
            return content.replace(stop_word, ""), True
        return content, False
    if isinstance(content, list):
        modified = False
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str) and stop_word in part["text"]:
                part["text"] = part["text"].replace(stop_word, "")
                modified = True
        return content, modified
    return content, False


def _anthropic_response_to_assistant_message(response: dict) -> dict:
    """Wrap an Anthropic ``MessagesResponse`` body as an anthropic-shaped assistant message.

    The returned dict can be appended to ``MessagesRequest.messages`` so the
    full conversation (input + this assistant turn) survives a round-trip
    through ``MessagesRequest.model_validate`` + ``to_openai_messages``.
    """
    blocks = response.get("content") or []
    normalized: List[dict] = []
    for block in blocks:
        btype = block.get("type")
        if btype == "text":
            normalized.append({"type": "text", "text": block.get("text", "")})
        elif btype == "thinking":
            nb: dict = {"type": "thinking", "thinking": block.get("thinking", "")}
            if "signature" in block:
                nb["signature"] = block["signature"]
            normalized.append(nb)
        elif btype == "tool_use":
            tool_id = block.get("id")
            name = block.get("name")
            if not tool_id or not name:
                raise ValueError(f"tool_use block missing id/name: {block}")
            normalized.append({"type": "tool_use", "id": tool_id, "name": name, "input": block.get("input", {})})
    return {"role": "assistant", "content": normalized}


def _anthropic_request_to_openai(req_body: dict) -> tuple[list[dict], Optional[list[dict]]]:
    """Anthropic request body → OpenAI (messages, tools).

    Uses lmdeploy's ``MessagesRequest`` + ``to_openai_messages`` /
    ``to_openai_tools`` to keep all anthropic content-block / tool-use /
    tool-result / system handling consistent with what the worker would do.
    """
    from lmdeploy.serve.anthropic.adapter import to_openai_messages, to_openai_tools
    from lmdeploy.serve.anthropic.protocol import MessagesRequest

    req = MessagesRequest.model_validate(req_body)
    messages = to_openai_messages(req)
    converted_tools = to_openai_tools(req.tools)
    tools = [t.model_dump() for t in converted_tools] if converted_tools else None
    _normalize_tool_call_arguments(messages)
    return messages, tools


class SessionServer:
    """SessionServer intercepts and records requests sent to a remote LLM API worker.

    It acts as a reverse-proxy in front of an already running worker (lmdeploy,
    sglang, vllm). Each supported endpoint is transparently forwarded along
    its native path with its native response shape — only the on_request and
    on_response hooks normalize messages to OpenAI form so the tokenizer +
    ``trace_store`` always see one shape.

    Supported endpoints:

    - ``POST /v1/chat/completions`` — passthrough
    - ``POST /v1/messages`` (Anthropic) — passthrough; hooks render the
      request via the lmdeploy adapter to drive prefix-cache lookup, swap
      ``messages``/``system``/``tools``/``tool_choice`` out for raw
      ``input_ids`` on the wire (lmdeploy requires this when ``input_ids`` is
      set), then re-assemble the OpenAI-shaped trace from the anthropic
      response.

    The ``/v1/responses`` endpoint is intentionally rejected (501) because the
    upstream worker does not implement it.

    Args:
        worker_base_url (str): The base URL of the real worker (e.g. "http://127.0.0.1:8000")
        tokenizer_path (str): The path to the tokenizer model.
        host (str): Host for this session server to listen on.
        port (int): Port for this session server to listen on.
        request_timeout (float): Total timeout in seconds for forwarding requests to the worker.
        read_bufsize (int): Buffer limit for line reader in ClientSession. Default is 64MB (2**26).
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

        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._lmdeploy_actor: Optional[ray.actor.ActorHandle] = None

    async def on_request(self, req_body: dict, fmt: str, *, trace_enabled: bool = True) -> dict:
        """Normalize the request to drive the prefix cache + inject extension fields.

        When ``trace_enabled`` is False (evaluate mode), forward the request
        as-is with parameter-name hygiene only — no token-in-token-out
        rewrite — so the upstream worker handles tokenization on its own.

        Args:
            req_body (dict): The original client request body.
            fmt (str): The detected request format (one of ``FMT_OPENAI`` /
                ``FMT_ANTHROPIC``).
            trace_enabled (bool): Whether the request should populate the
                trace_store. Disabled when the caller sets
                ``return_token_ids=False`` (evaluate path).

        Returns:
            dict: The forward-ready request body. The wire shape still
            matches ``fmt``; in trace mode it carries ``input_ids`` plus the
            ``return_*`` extension flags.
        """
        # Evaluate path: forward unchanged except for the legacy ``logprobs``
        # → ``return_logprob`` rename and stripping the SessionServer-only
        # ``session_id`` field.
        if not trace_enabled:
            worker_req = {k: v for k, v in req_body.items() if k not in _SESSION_SERVER_ONLY_KEYS}
            if "logprobs" in worker_req:
                worker_req.setdefault("return_logprob", worker_req.pop("logprobs"))
            if not _bool_request_value(worker_req.get("return_logprob"), False):
                worker_req.pop("top_logprobs", None)
                worker_req["return_logprob"] = False
            worker_req["return_token_ids"] = False
            worker_req.setdefault("return_routed_experts", True)
            return worker_req

        session_id = req_body["session_id"]

        # 1. Render the request as OpenAI chat-template input to drive the
        #    string-prefix cache. For non-OpenAI formats we convert in-memory;
        #    we do NOT mutate req_body["messages"] in those cases — the wire
        #    body keeps its native shape.
        if fmt == FMT_ANTHROPIC:
            openai_messages, openai_tools = _anthropic_request_to_openai(req_body)
        else:
            openai_messages = req_body["messages"]
            openai_tools = req_body.get("tools", None)

        # Defensive scrub: an earlier round may have rendered ``<|im_end|>`` into
        # an assistant text block (e.g. via ``include_stop_str_in_output``); if
        # the client echoes that block back to us, the chat template would emit
        # the stop_word twice and break tokenization.
        for msg in openai_messages:
            if isinstance(msg, dict) and "content" in msg:
                msg["content"], _ = _strip_stop_word(msg["content"], self.stop_word)

        prompt_text = self.tokenizer.apply_chat_template(
            canonicalize_messages_for_chat_template(openai_messages),
            tools=openai_tools,
            add_generation_prompt=True,
            tokenize=False,
        )

        # 2. Prefix-cache lookup; insert the delta tail back into the store.
        prefix, nodes = await self.store.search.remote(session_id, prompt_text, filter_none=True)
        if prefix:
            get_logger().debug(f"Hit prefix cache for session {session_id}")
        delta, delta_ids = prompt_text[len(prefix) :], []
        if delta:
            delta_ids = self.tokenizer.encode(delta, add_special_tokens=False)
            await self.store.insert.remote(session_id, prompt_text, TokenizedSegment(text=delta, token_ids=delta_ids))
        input_ids = reduce(add, [node.value.token_ids for node in nodes] + [delta_ids])

        # 3. Inject extension fields on the forwarded request. The body still
        #    follows ``fmt``'s native schema. ``logprobs``/``top_logprobs`` are
        #    dropped because we always force ``return_logprob=True`` to drive
        #    the trace_store; their original values would just shadow ours.
        worker_req = {
            k: v for k, v in req_body.items() if k not in _SESSION_SERVER_ONLY_KEYS | {"logprobs", "top_logprobs"}
        }
        worker_req["input_ids"] = input_ids
        worker_req["return_token_ids"] = True
        worker_req["return_routed_experts"] = True
        worker_req["return_logprob"] = True
        worker_req["include_stop_str_in_output"] = True

        if fmt == FMT_ANTHROPIC:
            # lmdeploy's /v1/messages requires messages to be empty AND
            # system / tools / tool_choice to be unset whenever input_ids is
            # supplied (raw input_ids bypass message rendering).
            worker_req["messages"] = []
            worker_req.pop("system", None)
            worker_req.pop("tools", None)
            worker_req.pop("tool_choice", None)
        else:
            worker_req["messages"] = []

        return worker_req

    async def on_response(self, worker_resp: dict, fmt: str, orig_req_body: dict) -> dict:
        """Trace the assistant turn into ``trace_store`` (always in OpenAI shape).

        Args:
            worker_resp (dict): The parsed upstream response (still in
                ``fmt`` shape).
            fmt (str): The request format that produced this response.
            orig_req_body (dict): The original client request body, before
                ``on_request`` rewrote it. Needed to reconstruct the full
                conversation for OpenAI-shape tracing because the wire body
                forwarded to the worker no longer carries ``messages`` /
                ``system`` / ``tools``.

        Returns:
            dict: The (possibly trimmed) response — currently unused by the
            caller but kept for symmetry.
        """
        session_id = orig_req_body["session_id"]

        if fmt == FMT_ANTHROPIC:
            output_token_ids = worker_resp.get("output_ids")
            output_token_logprobs = worker_resp.get("output_token_logprobs")
            raw_routed_expert = worker_resp.get("routed_experts")
            assistant_anthropic_msg = _anthropic_response_to_assistant_message(worker_resp)
            full_req = copy.deepcopy(orig_req_body)
            full_req.setdefault("messages", []).append(assistant_anthropic_msg)
            openai_messages, openai_tools = _anthropic_request_to_openai(full_req)
            assistant_msg = openai_messages[-1]
            messages = openai_messages[:-1]
            tools = openai_tools
        else:
            choice = worker_resp["choices"][0]
            output_token_ids = choice.get("output_ids")
            output_token_logprobs = choice.get("output_token_logprobs")
            raw_routed_expert = choice.get("routed_experts")
            assistant_msg = choice["message"]
            messages = orig_req_body["messages"]
            tools = orig_req_body.get("tools", None)

        if output_token_ids is None:
            raise RuntimeError(
                "SessionServer response has no output_ids; cannot export a training trace for this assistant turn."
            )
        output_logprobs = _extract_output_logprobs(output_token_logprobs, output_token_ids)

        if isinstance(assistant_msg.get("content"), (str, list)):
            assistant_msg["content"], _ = _strip_stop_word(assistant_msg["content"], self.stop_word)
        for msg in messages:
            if isinstance(msg, dict) and "content" in msg:
                msg["content"], _ = _strip_stop_word(msg["content"], self.stop_word)

        # Render OpenAI prompts for the prefix-cache key boundary.
        old_prompt = self.tokenizer.apply_chat_template(
            canonicalize_messages_for_chat_template(messages), tools=tools, add_generation_prompt=True, tokenize=False
        )
        full_messages = [*messages, assistant_msg]
        new_prompt = (
            self.tokenizer.apply_chat_template(
                canonicalize_messages_for_chat_template(full_messages),
                tools=tools,
                add_generation_prompt=False,
                tokenize=False,
            )
        ).rstrip()
        assert new_prompt.startswith(old_prompt) and new_prompt.endswith(self.stop_word)

        if raw_routed_expert is not None:
            raw_routed_expert = await self._decode_routed_experts(raw_routed_expert)
            if len(raw_routed_expert) > 0:
                num_layers = raw_routed_expert.shape[1]
                topk_experts = raw_routed_expert.shape[2]
                dummy_expert = np.full((1, num_layers, topk_experts), 0, dtype=raw_routed_expert.dtype)
                raw_routed_expert = np.concatenate([dummy_expert, raw_routed_expert], axis=0)

            _, nodes = await self.store.search.remote(session_id, old_prompt, filter_none=True)

            # last node in nodes corresponds to the delta inserted in on_request (if any)
            if nodes:
                delta_node_val: TokenizedSegment = nodes[-1].value
                delta_len = len(delta_node_val.token_ids)
                prefix_len = sum(len(n.value.token_ids) for n in nodes[:-1])
                assert prefix_len + delta_len + len(output_token_ids) == len(raw_routed_expert)

                delta_expert = raw_routed_expert[prefix_len : prefix_len + delta_len]
                response_expert = raw_routed_expert[prefix_len + delta_len :]

                if delta_len > 0:
                    delta_node_val.expert_key = ray.put(delta_expert)
                    await self.store.insert.remote(session_id, old_prompt, delta_node_val)

                raw_routed_expert = ray.put(response_expert)
            else:
                raw_routed_expert = ray.put(raw_routed_expert)

        await self.store.insert.remote(
            session_id,
            key=new_prompt,
            value=TokenizedSegment(
                text=new_prompt[len(old_prompt) :],
                token_ids=output_token_ids,
                logprobs=output_logprobs,
                labels=output_token_ids,
                expert_key=raw_routed_expert,
                length=len(output_token_ids),
            ),
        )

        return worker_resp

    @property
    def url(self) -> str:
        """The bound URL for the SessionServer."""
        return f"http://{self.host}:{self.port}"

    async def start(self):
        """Start the SessionServer proxy application."""
        if self._site is not None:
            return

        self._app = web.Application()
        self._app.router.add_route("*", "/{path:.*}", self._handle_request)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()
        get_logger().info(f"SessionServer listening on {self.url} (Forwarding to {self.worker_base_url})")

    async def stop(self):
        """Cleanly stop the SessionServer application."""
        if self._runner:
            await self._runner.cleanup()
        self._site = None
        self._runner = None
        self._app = None
        get_logger().info("SessionServer stopped.")

    async def _handle_request(self, request: web.Request) -> web.Response:
        """Proxy handler: detect format, run hooks, forward, stream back."""

        req_path = request.match_info["path"]

        # Reject /v1/responses outright — upstream worker doesn't implement it.
        if req_path.endswith("/responses") or "/v1/responses" in req_path:
            return web.json_response(
                _lmdeploy_error_payload(
                    "/v1/responses is not supported by SessionServer (upstream worker has no responses endpoint).",
                    status=HTTPStatus.NOT_IMPLEMENTED,
                    error_type="not_implemented",
                ),
                status=HTTPStatus.NOT_IMPLEMENTED,
            )

        fmt = _detect_format(req_path)

        # Read the request body
        request_body = await request.read()
        request_data = None
        orig_req_body: Optional[dict] = None
        trace_enabled = False
        orig_return_logprob = orig_return_token_ids = False
        orig_return_routed_experts = True
        if request_body:
            try:
                request_data = json.loads(request_body)
                orig_req_body = copy.deepcopy(request_data)

                trace_enabled = _request_uses_trace_store(request_data)
                # Accept either ``return_logprob`` (canonical) or the legacy
                # ``logprobs`` alias when deciding whether the client wanted
                # logprobs forwarded back.
                orig_return_logprob = _bool_request_value(
                    request_data.get("return_logprob", request_data.get("logprobs")), False
                )
                orig_return_token_ids = _bool_request_value(request_data.get("return_token_ids"), False)
                orig_return_routed_experts = _bool_request_value(request_data.get("return_routed_experts"), True)

                request_data = await self.on_request(request_data, fmt, trace_enabled=trace_enabled)
                request_body = json.dumps(request_data).encode("utf-8")
            except json.JSONDecodeError:
                pass
            except Exception as exc:
                message = f"SessionServer request hook failed: {type(exc).__name__}: {exc}"
                get_logger().error(message)
                return web.json_response(_lmdeploy_error_payload(message), status=500)

        # Build forwarding headers, dropping original Host / Content-Length.
        forward_headers = dict(request.headers)
        forward_headers.pop("Host", None)
        forward_headers.pop("host", None)
        forward_headers.pop("Content-Length", None)
        forward_headers.pop("content-length", None)
        if fmt == FMT_ANTHROPIC:
            forward_headers["anthropic-version"] = "2023-06-01"

        # Path is forwarded verbatim — each endpoint keeps its native shape.
        target_url = f"{self.worker_base_url}/{req_path.lstrip('/')}"
        if request.query_string:
            target_url += f"?{request.query_string}"

        is_stream = request_data.get("stream", False) if request_data else False

        # Build the per-format stream cleaner (strips lmdeploy-injected
        # extension fields that the client didn't ask for, and removes the
        # stop word from any user-visible text).
        clean_data = self._build_data_cleaner(
            fmt,
            orig_return_logprob=orig_return_logprob,
            orig_return_token_ids=orig_return_token_ids,
            orig_return_routed_experts=orig_return_routed_experts,
        )

        timeout = ClientTimeout(total=self.request_timeout, sock_connect=30)
        async with ClientSession(read_bufsize=self.read_bufsize, timeout=timeout) as client:
            async with client.request(
                method=request.method, url=target_url, headers=forward_headers, data=request_body
            ) as resp:
                if is_stream:
                    response_chunks: list[bytes] = []
                    response = web.StreamResponse(
                        status=resp.status,
                        headers={
                            k: v
                            for k, v in resp.headers.items()
                            if k.lower() not in ("transfer-encoding", "content-length", "content-encoding")
                        },
                    )
                    await response.prepare(request)
                    # If the downstream client closes the socket mid-stream
                    # (e.g. AsyncAPIClient bails out on a finish_reason=='error'
                    # chunk after the prompt overflowed the session window),
                    # keep draining the upstream so the trace is still recorded
                    # in full but stop attempting to write to the closed socket.
                    client_alive = True
                    async for line in resp.content:
                        # Only retain chunks when we'll actually need to parse
                        # them for tracing; evaluate-mode requests skip this
                        # so memory does not grow with stream length.
                        if trace_enabled:
                            response_chunks.append(line)

                        if request_data is not None and line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                            try:
                                text = line.decode("utf-8")
                                data = json.loads(text[6:])
                                if clean_data(data):
                                    line = ("data: " + json.dumps(data) + "\n").encode("utf-8")
                            except Exception:
                                pass

                        # Delay [DONE] only while a training trace still needs to be exported.
                        if client_alive and (not trace_enabled or line.strip() != b"data: [DONE]"):
                            try:
                                await response.write(line)
                            except (ConnectionError, ClientConnectionResetError):
                                client_alive = False

                    raw_response = b"".join(response_chunks) if trace_enabled else b""
                else:
                    raw_response = await resp.read()
                    final_raw_response = raw_response

                    if request_data is not None:
                        try:
                            parsed = json.loads(raw_response)
                            if clean_data(parsed):
                                final_raw_response = json.dumps(parsed).encode("utf-8")
                        except Exception:
                            pass

                    response = web.Response(
                        status=resp.status,
                        headers={
                            k: v
                            for k, v in resp.headers.items()
                            if k.lower() not in ("transfer-encoding", "content-length", "content-encoding")
                        },
                        body=final_raw_response,
                    )

        # Apply abstract on_response processing
        response_data: Optional[dict] = None
        skip_done = bool(is_stream and not trace_enabled)
        session_error_msg: Optional[str] = None
        if request_data and trace_enabled and orig_req_body is not None:
            if is_stream:
                try:
                    response_data = self._parse_stream_response(raw_response, fmt)
                    if response_data is None:
                        # Upstream emitted no traceable content — suppress the
                        # synthetic [DONE] line we usually append; the real
                        # stream content has already been forwarded as-is.
                        skip_done = True
                except Exception as exc:
                    session_error_msg = f"SessionServer stream trace failed: {type(exc).__name__}: {exc}"
                    skip_done = True
            else:
                try:
                    response_data = json.loads(raw_response)
                except json.JSONDecodeError:
                    pass
                if isinstance(response_data, dict) and _is_error_payload(response_data):
                    response_data = None

            if response_data is not None:
                try:
                    await self.on_response(response_data, fmt, orig_req_body)
                except Exception as exc:
                    session_error_msg = f"SessionServer response hook failed: {type(exc).__name__}: {exc}"

        if session_error_msg:
            get_logger().error(session_error_msg)

        if is_stream:
            try:
                if session_error_msg:
                    error_payload = _lmdeploy_error_payload(session_error_msg)
                    await response.write(
                        ("data: " + json.dumps(error_payload, ensure_ascii=False) + "\n\n").encode("utf-8")
                    )
                    skip_done = True
                if not skip_done:
                    await response.write(b"data: [DONE]\n\n")
                await response.write_eof()
            except (ConnectionError, ClientConnectionResetError):
                pass
        elif session_error_msg:
            return web.json_response(_lmdeploy_error_payload(session_error_msg), status=500)

        return response

    async def _decode_routed_experts(self, routed_experts: Any) -> np.ndarray:
        if isinstance(routed_experts, str):
            if self._lmdeploy_actor is None:
                self._lmdeploy_actor = ray.get_actor("shared_store", namespace="lmdeploy")
            assert self._lmdeploy_actor is not None, "LMDeploy actor should be available in the shared store."
            routed_experts_data = await self._lmdeploy_actor.get.remote(routed_experts)
            return np.asarray(routed_experts_data)
        return np.asarray(routed_experts)

    def _build_data_cleaner(
        self, fmt: str, *, orig_return_logprob: bool, orig_return_token_ids: bool, orig_return_routed_experts: bool
    ):
        """Return a per-format ``data -> bool`` cleaner stripping injected fields."""
        drops = {
            "output_token_logprobs": not orig_return_logprob,
            "output_ids": not orig_return_token_ids,
            "routed_experts": not orig_return_routed_experts,
        }

        if fmt == FMT_ANTHROPIC:

            def clean(data: dict) -> bool:
                modified = False
                # Non-stream MessagesResponse and stream MessageDeltaEvent put
                # extension fields at the top level. ContentBlockDeltaEvent
                # carries output_ids/output_token_logprobs on the event itself.
                for key, drop in drops.items():
                    if drop and key in data:
                        data.pop(key)
                        modified = True
                # Stop-word scrubbing for text deltas / text content blocks.
                delta = data.get("delta")
                if isinstance(delta, dict) and isinstance(delta.get("text"), str) and self.stop_word in delta["text"]:
                    delta["text"] = delta["text"].replace(self.stop_word, "")
                    modified = True
                for block in data.get("content") or []:
                    if (
                        isinstance(block, dict)
                        and isinstance(block.get("text"), str)
                        and self.stop_word in block["text"]
                    ):
                        block["text"] = block["text"].replace(self.stop_word, "")
                        modified = True
                return modified

            return clean

        # FMT_OPENAI
        def clean_openai(data: dict) -> bool:
            modified = False
            for key, drop in drops.items():
                if drop and key in data:
                    data.pop(key)
                    modified = True
                if drop:
                    for c in data.get("choices", []):
                        if key in c:
                            c.pop(key)
                            modified = True

            for c in data.get("choices", []):
                if "logprobs" in c:
                    c.pop("logprobs")
                    modified = True

            for c in data.get("choices", []):
                msg = c.get("message")
                if isinstance(msg, dict) and "content" in msg:
                    msg["content"], changed = _strip_stop_word(msg["content"], self.stop_word)
                    if changed:
                        modified = True
                delta = c.get("delta")
                if isinstance(delta, dict) and "content" in delta:
                    delta["content"], changed = _strip_stop_word(delta["content"], self.stop_word)
                    if changed:
                        modified = True

            return modified

        return clean_openai

    @staticmethod
    def _parse_stream_response(raw: bytes, fmt: str) -> Optional[dict]:
        """Parse an SSE stream and reconstruct the final response object.

        Returns ``None`` when the stream carried no traceable content (so the
        caller skips the trace hook entirely instead of recording garbage).
        """
        text = raw.decode("utf-8", errors="replace")
        events: list[dict] = []
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
            return None

        if fmt == FMT_ANTHROPIC:
            return SessionServer._parse_anthropic_stream(events)
        return SessionServer._parse_openai_stream(events, saw_done=saw_done)

    @staticmethod
    def _parse_openai_stream(events: list[dict], *, saw_done: bool) -> Optional[dict]:
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
                    raise RuntimeError(
                        f"Upstream SSE choice finished with error: {json.dumps(event, ensure_ascii=False)}"
                    )
                delta = choice.get("delta", {})

                if delta.get("content"):
                    content_parts.append(delta["content"])

                if choice.get("output_ids") is not None:
                    message["choices"][0].setdefault("output_ids", []).extend(choice["output_ids"])

                if choice.get("routed_experts") is not None:
                    message["choices"][0]["routed_experts"] = choice["routed_experts"]

                if choice.get("output_token_logprobs") is not None:
                    message["choices"][0].setdefault("output_token_logprobs", []).extend(
                        choice["output_token_logprobs"]
                    )

                if delta.get("reasoning_content"):
                    assistant_msg = message["choices"][0]["message"]
                    assistant_msg["reasoning_content"] = (
                        assistant_msg.get("reasoning_content", "") + delta["reasoning_content"]
                    )

                for tc_delta in delta.get("tool_calls") or []:
                    idx = tc_delta.get("index", 0)
                    if idx not in tool_calls_map:
                        tool_calls_map[idx] = {
                            "id": tc_delta.get("id", ""),
                            "type": tc_delta.get("type", "function"),
                            "function": {"name": "", "arguments": ""},
                        }
                    tc = tool_calls_map[idx]
                    fn = tc_delta.get("function", {})
                    if fn.get("name"):
                        tc["function"]["name"] += fn["name"]
                    if fn.get("arguments"):
                        tc["function"]["arguments"] += fn["arguments"]

                if choice.get("finish_reason"):
                    message["choices"][0]["finish_reason"] = choice["finish_reason"]

            if event.get("usage") is not None:
                usage = event["usage"]

        msg = message["choices"][0]["message"]
        msg["content"] = "".join(content_parts)
        if tool_calls_map:
            msg["tool_calls"] = [tool_calls_map[i] for i in sorted(tool_calls_map)]
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

    @staticmethod
    def _parse_anthropic_stream(events: list[dict]) -> Optional[dict]:
        """Aggregate Anthropic SSE events into a ``MessagesResponse``-shaped dict.

        Extension fields:
        - ``output_ids`` / ``output_token_logprobs`` live on each
          ``content_block_delta`` event and are concatenated in order.
        - ``routed_experts`` lives on the terminal ``message_delta`` event.

        Returns ``None`` when no ``message_stop`` was seen — that means the
        upstream stream was aborted before producing a complete turn.
        """
        message: dict[str, Any] = {"role": "assistant", "type": "message", "content": []}
        content_blocks: list[dict] = []
        current_block: Optional[dict] = None
        output_ids: list[int] = []
        output_token_logprobs: list = []
        routed_experts = None
        usage: dict[str, Any] = {}
        saw_message_stop = False

        for event in events:
            etype = event.get("type", "")

            if etype == "message_start":
                msg = event.get("message", {})
                if msg.get("id"):
                    message["id"] = msg["id"]
                if msg.get("model"):
                    message["model"] = msg["model"]
                if msg.get("usage"):
                    usage = dict(msg["usage"])
                if msg.get("role"):
                    message["role"] = msg["role"]

            elif etype == "content_block_start":
                current_block = dict(event.get("content_block") or {})

            elif etype == "content_block_delta":
                delta = event.get("delta") or {}
                dtype = delta.get("type")
                if current_block is None:
                    current_block = {}
                if dtype == "text_delta":
                    current_block.setdefault("type", "text")
                    current_block["text"] = current_block.get("text", "") + delta.get("text", "")
                elif dtype == "thinking_delta":
                    current_block.setdefault("type", "thinking")
                    current_block["thinking"] = current_block.get("thinking", "") + delta.get("thinking", "")
                elif dtype == "signature_delta":
                    current_block["signature"] = current_block.get("signature", "") + delta.get("signature", "")
                elif dtype == "input_json_delta":
                    current_block.setdefault("type", "tool_use")
                    current_block["_partial_json"] = current_block.get("_partial_json", "") + delta.get(
                        "partial_json", ""
                    )

                if event.get("output_ids"):
                    output_ids.extend(event["output_ids"])
                if event.get("output_token_logprobs"):
                    output_token_logprobs.extend(event["output_token_logprobs"])

            elif etype == "content_block_stop":
                if current_block is not None:
                    if "_partial_json" in current_block:
                        raw = current_block.pop("_partial_json")
                        try:
                            current_block["input"] = json.loads(raw) if raw else {}
                        except json.JSONDecodeError:
                            current_block["input"] = {}
                    content_blocks.append(current_block)
                    current_block = None

            elif etype == "message_delta":
                d = event.get("delta") or {}
                if d.get("stop_reason"):
                    message["stop_reason"] = d["stop_reason"]
                if d.get("stop_sequence") is not None:
                    message["stop_sequence"] = d["stop_sequence"]
                u = event.get("usage") or {}
                for k, v in u.items():
                    if isinstance(v, (int, float)):
                        usage[k] = usage.get(k, 0) + v
                    else:
                        usage[k] = v
                if event.get("routed_experts") is not None:
                    routed_experts = event["routed_experts"]

            elif etype == "message_stop":
                saw_message_stop = True

        if not saw_message_stop:
            return None
        if not output_ids:
            raise RuntimeError("Upstream anthropic SSE stream ended without output_ids.")

        message["content"] = content_blocks
        message["output_ids"] = output_ids
        message["output_token_logprobs"] = output_token_logprobs or None
        message["routed_experts"] = routed_experts
        if usage:
            message["usage"] = usage
        return message


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
