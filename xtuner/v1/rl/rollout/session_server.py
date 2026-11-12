import copy
import json
import re
import time
from functools import reduce
from http import HTTPStatus
from operator import add
from typing import Any, List, Optional

import numpy as np
import ray
from aiohttp import ClientConnectionResetError, ClientSession, ClientTimeout, ConnectionTimeoutError, TraceConfig, web

from transformers import AutoTokenizer
from xtuner.v1.utils import get_logger

from .chat_template import canonicalize_messages_for_chat_template
from .request_gate import RequestGate, RequestLease, SettlementReceipt
from .trace_store import TokenizedSegment, get_store


FMT_OPENAI = "openai"
FMT_ANTHROPIC = "anthropic"

# Fields the SessionServer consumes locally and never forwards upstream.
_SESSION_SERVER_ONLY_KEYS = {"session_id", "_xt_request_id", "_xt_model_step"}


def _normalize_path(req_path: str) -> str:
    """Return a leading-slash path without a query string."""
    return "/" + req_path.split("?", 1)[0].lstrip("/")


def _is_generation_endpoint(method: str, req_path: str) -> bool:
    """Whether this request uses the SessionServer generation hooks."""
    path = _normalize_path(req_path)
    return method.upper() == "POST" and (path.endswith("/v1/messages") or path.endswith("/v1/chat/completions"))


def _detect_format(req_path: str) -> str:
    path = _normalize_path(req_path)
    if path.endswith("/messages") or "/v1/messages" in path:
        return FMT_ANTHROPIC
    return FMT_OPENAI


def _is_error_payload(payload: dict) -> bool:
    return payload.get("error") is not None or payload.get("type") == "error" or payload.get("object") == "error"


def _is_lmdeploy_context_rejection(status: int, body: bytes) -> bool:
    """Recognize the pinned LMDeploy's input check before engine generation.

    A finished HTTP error alone does not prove that backend execution ended.
    LMDeploy bc2c8da's AsyncEngine rejects input_len >= session_len before
    submitting generation. Its OpenAI response has no stable machine error code,
    so match that specific payload conservatively; unknown errors stay unsafe.
    """
    if status != HTTPStatus.BAD_REQUEST:
        return False
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return False
    if not isinstance(payload, dict) or (
        payload.get("object") != "error"
        or payload.get("type") != "invalid_request_error"
        or payload.get("code") != 400
        or "choices" in payload
        or not isinstance(payload.get("message"), str)
    ):
        return False
    match = re.fullmatch(
        r"Input length \(([0-9]{1,20})\) must be smaller than the model context length \(([0-9]{1,20})\)\.",
        payload["message"],
    )
    return match is not None and int(match[1]) >= int(match[2]) > 0


class _UpstreamResponseError(RuntimeError):
    """Internal marker for an error explicitly returned by the upstream."""


def _error_payload(fmt: str, message: str, status: int = 500, error_type: str = "internal_server_error") -> dict:
    """Error body in the caller's native shape.

    The downstream SessionClient recorder drops a turn only when it recognizes the
    error shape: Anthropic keys on top-level ``type == "error"``, OpenAI/lmdeploy on
    ``object == "error"``. Emitting the matching shape keeps a turn the SessionServer
    dropped (e.g. an ``on_response`` that raised on missing routed experts) out of the
    client's records, so the trace store stays a superset of them and export never
    sees a trajectory whose nodes were never written.
    """
    if fmt == FMT_ANTHROPIC:
        return {"type": "error", "error": {"type": error_type, "message": message}}
    return {"message": message, "type": error_type, "code": status, "object": "error"}


def _bool_request_value(value: Any, default: bool = False) -> bool:
    """Coerce an opaque request-body value (bool / int / str / None) to bool.

    Mirrors how lmdeploy validates flag-style fields — accepts ``"true"``, ``"1"`` etc. as true,
    ``"false"``, ``"0"``, ``""`` etc. as false, and falls back to the default value for unrecognized strings.
    """
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def _request_uses_trace_store(req_body: dict) -> bool:
    """Whether this request should drive the trace_store / token-in-token-out
    path.

    Evaluate-mode callers opt out by setting ``return_token_ids=False`` on the
    request body; tracing then becomes a passthrough proxy with parameter-name
    hygiene (``logprobs`` → ``return_logprob``) and the upstream worker handles
    its own tokenization.
    """
    if req_body.get("session_id") is None or "messages" not in req_body:
        return False
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
    """Remove ``stop_word`` from a message ``content`` field (any OpenAI
    shape).

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
    """Wrap an Anthropic ``MessagesResponse`` body as an anthropic-shaped
    assistant message.

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


def _filter_anthropic_user_tools(tools) -> Optional[list]:
    """Drop Anthropic server-side tools (``web_search_*`` / ``computer_*`` /
    ``bash_*`` / ``text_editor_*``).

    ``ToolParam`` in lmdeploy's anthropic protocol requires ``input_schema``;
    server-side built-ins are managed by Anthropic itself and only carry
    ``type`` + ``name``, so they fail ``MessagesRequest.model_validate`` and
    the worker can't execute them anyway. Keep only entries that have an
    ``input_schema`` (= user-defined tools the worker can route to).

    Returns the filtered list (possibly empty), or ``None`` if input was None.
    """
    if tools is None:
        return None
    if not isinstance(tools, list):
        return tools
    kept = [t for t in tools if isinstance(t, dict) and "input_schema" in t]
    if len(kept) != len(tools):
        dropped = [t.get("type") or t.get("name") for t in tools if t not in kept]
        get_logger().debug(f"Dropped {len(tools) - len(kept)} anthropic server-side tool(s): {dropped}")
    return kept


class SessionServer:
    """SessionServer intercepts and records requests sent to a remote LLM API
    worker.

    It acts as a reverse-proxy in front of an already running worker (lmdeploy,
    sglang, vllm). Each supported endpoint is transparently forwarded along
    its native path with its native response shape — only the on_request and
    on_response hooks normalize messages to OpenAI form so the tokenizer +
    ``trace_store`` always see one shape.

    Supported endpoints:

    - ``POST /v1/chat/completions`` — passthrough
    - ``POST /v1/messages`` (Anthropic) — passthrough; hooks render the
      request via the lmdeploy adapter to drive prefix-cache lookup, swap
      ``messages``/``system`` out for raw
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

    request_gate: RequestGate | None = None

    def __init__(
        self,
        worker_base_url: str,
        tokenizer_path: str,
        host: str = "127.0.0.1",
        port: int = 8080,
        request_timeout: float = 1200.0,
        read_bufsize: int = 2**26,
        enable_return_routed_experts: bool = False,
        turn_resume_wait_timeout: float | None = None,
        turn_resume_initially_closed: bool = False,
    ):
        self.worker_base_url = worker_base_url.rstrip("/")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.host = host
        self.port = port
        self.request_timeout = request_timeout
        self.read_bufsize = read_bufsize
        self.enable_return_routed_experts = enable_return_routed_experts
        self.turn_resume_wait_timeout = turn_resume_wait_timeout
        self.turn_resume_initially_closed = turn_resume_initially_closed
        self.request_gate = (
            RequestGate(wait_timeout=turn_resume_wait_timeout, initially_open=not turn_resume_initially_closed)
            if turn_resume_wait_timeout is not None else None
        )
        self.store = get_store()
        self.stop_word = self.tokenizer.eos_token or ""

        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._lmdeploy_actor: Optional[ray.actor.ActorHandle] = None

    async def on_request(self, req_body: dict, fmt: str, *, trace_enabled: bool = True) -> dict:
        """Normalize the request to drive the prefix cache + inject extension
        fields.

        When ``trace_enabled`` is False (evaluate mode), forward the request
        without token-in-token-out rewriting, so the upstream worker handles
        tokenization. Parameter names are normalized, and routed experts are
        forced off when the engine was started without routed-expert support.

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
            if self.enable_return_routed_experts:
                worker_req.setdefault("return_routed_experts", True)
            else:
                # Engines started without routed experts reject the flag with HTTP 400.
                worker_req["return_routed_experts"] = False
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

        # 2. Prefix-cache lookup. The delta tail is NOT persisted here: writing a
        #    node before generation would leave it with ``expert_key=None`` (routed
        #    experts don't exist yet), which is exactly the orphan the trace store
        #    must never keep. ``delta_ids`` is only needed locally to build
        #    ``input_ids``; ``on_response`` re-derives and persists the delta node
        #    once its routed experts are known (single-phase write).
        prefix, nodes = await self.store.search.remote(session_id, prompt_text, filter_none=True)
        if prefix:
            get_logger().debug(f"Hit prefix cache for session {session_id}")
        delta = prompt_text[len(prefix) :]
        delta_ids = self.tokenizer.encode(delta, add_special_tokens=False) if delta else []
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
        worker_req["return_routed_experts"] = self.enable_return_routed_experts
        worker_req["return_logprob"] = True
        worker_req["include_stop_str_in_output"] = True

        if fmt == FMT_ANTHROPIC:
            # Raw tokens replace message rendering. Retain the tool schema
            # and choice: LMDeploy still needs them to parse/validate output
            # tool calls, even though their prompt is already in input_ids.
            worker_req["messages"] = []
            worker_req.pop("system", None)
        else:
            worker_req["messages"] = []

        return worker_req

    async def on_response(self, worker_resp: dict, fmt: str, orig_req_body: dict) -> dict:
        """Trace the assistant turn into ``trace_store`` (always in OpenAI
        shape).

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
        if orig_req_body.get("_xt_request_id") is not None and (
            not worker_resp.get("choices")
            or any(c.get("finish_reason") not in {"stop", "length", "tool_calls"} for c in worker_resp["choices"])
        ):
            raise RuntimeError(
                "Backend returned an unsuccessful generation terminal: "
                + str([c.get("finish_reason") for c in worker_resp.get("choices", [])])
            )
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

        if isinstance(assistant_msg.get("content"), (str, list)):
            assistant_msg["content"], _ = _strip_stop_word(assistant_msg["content"], self.stop_word)
        for msg in messages:
            if isinstance(msg, dict) and "content" in msg:
                msg["content"], _ = _strip_stop_word(msg["content"], self.stop_word)

        # Render OpenAI prompts for the prefix-cache key boundary.
        old_prompt = self.tokenizer.apply_chat_template(
            canonicalize_messages_for_chat_template(messages), tools=tools, add_generation_prompt=True, tokenize=False
        )
        if output_token_ids is None:
            raise RuntimeError(
                "SessionServer response has no output_ids; cannot export a training trace for this assistant turn."
            )
        output_logprobs = _extract_output_logprobs(output_token_logprobs, output_token_ids)

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

        # Re-derive the input delta the worker processed. on_request no longer
        # persists it (that would leave a None-expert placeholder in the store),
        # so recompute it from the current prefix and split the routed experts
        # across (prefix, delta, response) using the same boundaries on_request
        # used to build ``input_ids``.
        matched_prefix, prefix_nodes = await self.store.search.remote(session_id, old_prompt, filter_none=True)
        delta_text = old_prompt[len(matched_prefix) :]
        delta_ids = self.tokenizer.encode(delta_text, add_special_tokens=False) if delta_text else []
        prefix_len = sum(len(n.value.token_ids) for n in prefix_nodes)
        delta_len = len(delta_ids)

        delta_expert: Any = None
        response_expert: Any = None
        if raw_routed_expert is not None:
            raw_routed_expert = await self._decode_routed_experts(raw_routed_expert)
            if len(raw_routed_expert) > 0:
                num_layers = raw_routed_expert.shape[1]
                topk_experts = raw_routed_expert.shape[2]
                dummy_expert = np.full((1, num_layers, topk_experts), 0, dtype=raw_routed_expert.dtype)
                raw_routed_expert = np.concatenate([dummy_expert, raw_routed_expert], axis=0)
            assert prefix_len + delta_len + len(output_token_ids) == len(raw_routed_expert)
            if delta_len > 0:
                delta_expert = ray.put(raw_routed_expert[prefix_len : prefix_len + delta_len])
            response_expert = ray.put(raw_routed_expert[prefix_len + delta_len :])
        elif self.enable_return_routed_experts and (
            orig_req_body.get("_xt_request_id") is not None
            or _bool_request_value(orig_req_body.get("return_routed_experts"), True)
        ):
            raise RuntimeError(
                "SessionServer response is missing routed_experts, but enable_return_routed_experts is True."
                " The upstream worker may encounter an LLM call error."
            )

        # Single-phase write: nodes are persisted only now, already carrying their
        # final ``expert_key`` (a valid ref for MoE, None for dense). Every assert /
        # raise above runs before any write, so a dropped turn leaves nothing
        # half-written in the store (and no orphan placeholder to clean up).
        if delta_ids:
            await self.store.insert.remote(
                session_id,
                key=old_prompt,
                value=TokenizedSegment(
                    text=delta_text,
                    token_ids=delta_ids,
                    expert_key=delta_expert,
                    request_id=orig_req_body.get("_xt_request_id"),
                    model_step=orig_req_body.get("_xt_model_step"),
                ),
            )
        await self.store.insert.remote(
            session_id,
            key=new_prompt,
            value=TokenizedSegment(
                text=new_prompt[len(old_prompt) :],
                token_ids=output_token_ids,
                logprobs=output_logprobs,
                labels=output_token_ids,
                expert_key=response_expert,
                request_id=orig_req_body.get("_xt_request_id"),
                model_step=orig_req_body.get("_xt_model_step"),
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
        path = request.match_info["path"]
        if self.request_gate is None or not _is_generation_endpoint(request.method, path):
            return await self._forward_request(request)
        fmt = _detect_format(path)
        if fmt != FMT_OPENAI:
            return web.json_response(
                _error_payload(fmt, "turn-resume supports OpenAI chat completions only", 400), status=400
            )
        try:
            body = await request.json()
        except (ValueError, UnicodeDecodeError):
            return web.json_response(_error_payload(fmt, "Invalid generation JSON", 400), status=400)
        if not isinstance(body, dict) or body.get("stream") or not _request_uses_trace_store(body):
            return web.json_response(
                _error_payload(fmt, "turn-resume requires nonstream traced generation", 400), status=400
            )
        async with self.request_gate.request(str(body["session_id"])) as lease:
            started = time.perf_counter()
            try:
                return await self._forward_request(request, lease)
            except BaseException as exc:
                get_logger().error(
                    f"request_forward_failed id={lease.request_id} exception={type(exc).__name__} "
                    f"elapsed_s={time.perf_counter() - started:.6f} "
                    f"settlement_recorded={lease.settlement is not None}"
                )
                raise

    async def _forward_request(self, request: web.Request, lease: RequestLease | None = None) -> web.Response:
        """Proxy handler: detect format, run hooks, forward, stream back."""

        request_start = time.perf_counter()
        prepare_s = 0.0
        trace_s = 0.0
        req_path = request.match_info["path"]
        is_generation_endpoint = _is_generation_endpoint(request.method, req_path)
        fmt = _detect_format(req_path)

        # Reject /v1/responses outright — upstream worker doesn't implement it.
        if req_path.endswith("/responses") or "/v1/responses" in req_path:
            return web.json_response(
                _error_payload(
                    fmt,
                    "/v1/responses is not supported by SessionServer (upstream worker has no responses endpoint).",
                    status=HTTPStatus.NOT_IMPLEMENTED,
                    error_type="not_implemented",
                ),
                status=HTTPStatus.NOT_IMPLEMENTED,
            )

        # Read the request body
        request_body = await request.read()
        request_data = None
        orig_req_body: Optional[dict] = None
        trace_enabled = False
        orig_return_logprob = orig_return_token_ids = False
        orig_return_routed_experts = True
        if request_body and not is_generation_endpoint:
            # Only inspect object-shaped JSON. Lists, scalars and malformed
            # bodies stay byte-for-byte untouched for the upstream validator.
            if request_body.lstrip().startswith(b"{"):
                try:
                    parsed_request_data = json.loads(request_body)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
                else:
                    if isinstance(parsed_request_data, dict):
                        request_data = parsed_request_data
                        if _SESSION_SERVER_ONLY_KEYS.intersection(request_data):
                            request_data = {
                                key: value
                                for key, value in request_data.items()
                                if key not in _SESSION_SERVER_ONLY_KEYS
                            }
                            request_body = json.dumps(request_data).encode("utf-8")
        elif request_body:
            try:
                parsed_request_data = json.loads(request_body)
                if isinstance(parsed_request_data, dict):
                    request_data = parsed_request_data
                    # Anthropic server-side built-ins (web_search_*, computer_, ...)
                    # don't carry ``input_schema`` and the lmdeploy worker can't
                    # route them anyway — drop them before anything downstream
                    # (orig_req_body / on_request / wire forward) sees them.
                    if fmt == FMT_ANTHROPIC and "tools" in request_data:
                        filtered = _filter_anthropic_user_tools(request_data["tools"])
                        if filtered:
                            request_data["tools"] = filtered
                        else:
                            request_data.pop("tools", None)
                    if lease is not None:
                        request_data["_xt_request_id"] = lease.request_id
                        request_data["_xt_model_step"] = lease.model_step
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

                    prepare_start = time.perf_counter()
                    request_data = await self.on_request(request_data, fmt, trace_enabled=trace_enabled)
                    prepare_s = time.perf_counter() - prepare_start
                    request_body = json.dumps(request_data).encode("utf-8")
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
            except Exception as exc:
                message = f"SessionServer request hook failed: {type(exc).__name__}: {exc}"
                get_logger().error(message)
                if lease is not None:
                    # Request preparation failed before any backend send.
                    get_logger().error(f"request_terminal id={lease.request_id} outcome=not_started")
                    lease.settlement = SettlementReceipt(True, True, "not_started")
                return web.json_response(_error_payload(fmt, message), status=500)

        trace_active = is_generation_endpoint and trace_enabled

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
        clean_data = (
            self._build_data_cleaner(
                fmt,
                orig_return_logprob=orig_return_logprob,
                orig_return_token_ids=orig_return_token_ids,
                orig_return_routed_experts=orig_return_routed_experts,
            )
            if is_generation_endpoint and request_data is not None
            else None
        )

        forward_start = time.perf_counter()
        # ConnectionTimeoutError is raised by aiohttp while acquiring a connection,
        # before sending this attempt. Keep a monotonic send latch as redirects or
        # future transport changes may have already sent an earlier attempt.
        headers_may_have_been_sent = False
        trace_configs = []
        if lease is not None:
            trace = TraceConfig()

            async def mark_headers_sent(session, context, params):
                nonlocal headers_may_have_been_sent
                headers_may_have_been_sent = True

            trace.on_request_headers_sent.append(mark_headers_sent)
            trace_configs.append(trace)
        try:
            timeout = ClientTimeout(total=self.request_timeout, sock_connect=30)
            async with ClientSession(read_bufsize=self.read_bufsize, timeout=timeout, trace_configs=trace_configs) as client:
                async with client.request(
                    method=request.method, url=target_url, headers=forward_headers, data=request_body
                ) as resp:
                    upstream_status = resp.status
                    is_success_response = HTTPStatus.OK <= upstream_status < HTTPStatus.MULTIPLE_CHOICES
                    trace_response = trace_active and is_success_response
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
                        # If the downstream client closes the socket mid-stream
                        # (e.g. AsyncAPIClient bails out on a finish_reason=='error'
                        # chunk after the prompt overflowed the session window),
                        # keep draining the upstream so the trace is still recorded
                        # in full but stop attempting to write to the closed socket.
                        try:
                            await response.prepare(request)
                        except (ConnectionError, ClientConnectionResetError):
                            client_alive = False
                        else:
                            client_alive = True
                        skip_blank_after_done = False
                        async for line in resp.content:
                            # Only retain chunks when we'll actually need to parse
                            # them for tracing; evaluate-mode requests skip this
                            # so memory does not grow with stream length.
                            if trace_response:
                                response_chunks.append(line)

                            if (
                                is_success_response
                                and clean_data is not None
                                and request_data is not None
                                and line.startswith(b"data: ")
                                and line.strip() != b"data: [DONE]"
                            ):
                                try:
                                    text = line.decode("utf-8")
                                    data = json.loads(text[6:])
                                    if isinstance(data, dict) and not _is_error_payload(data) and clean_data(data):
                                        line = ("data: " + json.dumps(data) + "\n").encode("utf-8")
                                except Exception:
                                    pass

                            # Delay [DONE] only while a training trace still needs to be exported.
                            if trace_response and skip_blank_after_done:
                                if not line.strip():
                                    continue
                                skip_blank_after_done = False
                            if line.strip() == b"data: [DONE]":
                                if trace_response:
                                    skip_blank_after_done = True
                                    continue
                            if client_alive:
                                try:
                                    await response.write(line)
                                except (ConnectionError, ClientConnectionResetError):
                                    client_alive = False

                        raw_response = b"".join(response_chunks) if trace_response else b""
                    else:
                        raw_response = await resp.read()
                        final_raw_response = raw_response

                        if is_success_response and clean_data is not None and request_data is not None:
                            try:
                                parsed = json.loads(raw_response)
                                if isinstance(parsed, dict) and not _is_error_payload(parsed) and clean_data(parsed):
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

        except ConnectionTimeoutError:
            if lease is not None and not headers_may_have_been_sent:
                # A failed, never-sent attempt has no backend generation to drain.
                # Do not retry or fabricate a successful trace/reward. Preserve
                # the exception for the caller; release only its unsafe gate lease.
                lease.settlement = SettlementReceipt(True, True, "not_started_connect_timeout")
                get_logger().warning(
                    f"request_terminal id={lease.request_id} outcome=not_started_connect_timeout "
                    "backend_ended=True request_headers_sent=False"
                )
            raise
        except TimeoutError as exc:
            # Only HTTP forwarding timeouts qualify. Trace-hook errors, unknown
            # exceptions and caller cancellation must not be relabelled safe.
            if lease is not None and lease.settlement is None:
                if not hasattr(self, "_forward_timeouts"):
                    self._forward_timeouts = {}
                self._forward_timeouts[lease.request_id] = dict(
                    session_id=lease.session_id, epoch=lease.epoch,
                    headers_sent=headers_may_have_been_sent,
                    error_type=type(exc).__name__, forwarding_exited=True,
                )
            raise

        forward_s = time.perf_counter() - forward_start
        # Apply abstract on_response processing
        response_data: Optional[dict] = None
        skip_done = bool(is_stream and not trace_response)
        session_error_msg: Optional[str] = None
        if trace_response and request_data and orig_req_body is not None:
            if is_stream:
                try:
                    response_data = self._parse_stream_response(raw_response, fmt)
                    if response_data is None:
                        raise RuntimeError("Upstream SSE stream ended without a complete response.")
                except _UpstreamResponseError:
                    # An explicit upstream error is already part of the wire
                    # response. Do not wrap it in a second SessionServer error.
                    skip_done = b"data: [DONE]" not in raw_response
                except Exception as exc:
                    session_error_msg = f"SessionServer stream trace failed: {type(exc).__name__}: {exc}"
                    skip_done = True
            else:
                try:
                    parsed_response = json.loads(raw_response)
                    if not isinstance(parsed_response, dict):
                        raise TypeError(
                            f"generation response body must be a JSON object; got {type(parsed_response).__name__}"
                        )
                except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
                    session_error_msg = f"SessionServer response trace failed: {type(exc).__name__}: {exc}"
                else:
                    if not _is_error_payload(parsed_response):
                        response_data = parsed_response

            if response_data is not None:
                try:
                    trace_start = time.perf_counter()
                    await self.on_response(response_data, fmt, orig_req_body)
                    trace_s = time.perf_counter() - trace_start
                    if lease is not None and any(
                        c.get("finish_reason") not in {"stop", "length", "tool_calls"}
                        for c in response_data.get("choices", [])
                    ):
                        raise RuntimeError("Unsuccessful backend generation terminal")
                except Exception as exc:
                    session_error_msg = f"SessionServer response hook failed: {type(exc).__name__}: {exc}"

        if lease is not None:
            complete = (
                not is_stream
                and is_success_response
                and response_data is not None
                and not _is_error_payload(response_data)
                and bool(response_data.get("choices"))
                and all(c.get("finish_reason") is not None for c in response_data["choices"])
            )
            terminal = "trace_committed" if complete and not session_error_msg else "failed"
            # A known pre-generation rejection is a safely ended failed attempt,
            # not a successful trace and not an ambiguous in-flight request.
            rejected_before_generation = not is_stream and _is_lmdeploy_context_rejection(
                upstream_status, raw_response
            )
            backend_ended = complete or rejected_before_generation
            get_logger().info(f"request_terminal id={lease.request_id} outcome={terminal} backend_ended={backend_ended}")
            if rejected_before_generation:
                get_logger().info(
                    f"request_rejected_before_generation id={lease.request_id} reason=context_length_exceeded"
                )
            get_logger().info(
                f"request_timing id={lease.request_id} prepare_s={prepare_s:.6f} "
                f"forward_s={forward_s:.6f} trace_s={trace_s:.6f} "
                f"total_s={time.perf_counter() - request_start:.6f}"
            )
            lease.settlement = SettlementReceipt(backend_ended, True, terminal)
            if terminal == "trace_committed":
                payload = json.loads(response.body)
                payload["id"] = lease.request_id
                response.body = json.dumps(payload).encode("utf-8")

        if session_error_msg:
            get_logger().error(session_error_msg)

        if is_stream:
            try:
                if client_alive:
                    if session_error_msg:
                        error_payload = _error_payload(fmt, session_error_msg)
                        error_line = "data: " + json.dumps(error_payload, ensure_ascii=False) + "\n\n"
                        # Anthropic SSE carries a named ``event: error`` line; the recorder keys on the ``data:`` JSON
                        # either way, but a real Anthropic SDK consumer needs the event name.
                        if fmt == FMT_ANTHROPIC:
                            error_line = "event: error\n" + error_line
                        await response.write(error_line.encode("utf-8"))
                        skip_done = True
                    elif trace_response and not skip_done:
                        await response.write(b"data: [DONE]\n\n")
                    await response.write_eof()
            except (ConnectionError, ClientConnectionResetError):
                pass
        elif session_error_msg:
            return web.json_response(_error_payload(fmt, session_error_msg), status=500)

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
        """Return a per-format ``data -> bool`` cleaner stripping injected
        fields."""
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

        Returns ``None`` when no complete response can be reconstructed. The
        caller treats that as a local trace failure for trace-enabled requests.
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
                    raise _UpstreamResponseError(
                        f"Upstream SSE stream returned error: {json.dumps(event, ensure_ascii=False)}"
                    )
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
                    raise _UpstreamResponseError(
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
        """Aggregate Anthropic SSE events into a ``MessagesResponse``-shaped
        dict.

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

    def __init__(
        self,
        worker_base_url: str,
        tokenizer_path: str,
        host: str,
        port: int,
        request_timeout: float,
        enable_return_routed_experts: bool = False,
        turn_resume_wait_timeout: float | None = None,
        turn_resume_initially_closed: bool = False,
    ):
        self.worker_base_url = worker_base_url
        self.tokenizer_path = tokenizer_path
        self.host = host
        self.port = port
        self.request_timeout = request_timeout
        self.enable_return_routed_experts = enable_return_routed_experts
        self.turn_resume_wait_timeout = turn_resume_wait_timeout
        self.turn_resume_initially_closed = turn_resume_initially_closed
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
            enable_return_routed_experts=self.enable_return_routed_experts,
            turn_resume_wait_timeout=self.turn_resume_wait_timeout,
            turn_resume_initially_closed=self.turn_resume_initially_closed,
        )
        await self.server.start()
        return self.server.url

    def _gate(self) -> RequestGate:
        if self.server is None or self.server.request_gate is None:
            raise RuntimeError("turn-resume gate is not enabled")
        return self.server.request_gate

    async def request_status(self):
        state = self._gate().snapshot()
        state["known_forward_timeouts"] = sorted(getattr(self.server, "_forward_timeouts", {}))
        return state

    async def seal_requests(self, epoch: int):
        await self._gate().seal(epoch)

    async def drain_requests(self, timeout: float):
        return await self._gate().drain(timeout)

    async def settle_after_backend_sleep(self, epoch: int, receipt: dict):
        import asyncio
        from .trace_store import get_existing_store
        if (self.server is None or receipt.get("epoch") != epoch
                or receipt.get("backend_url") != self.worker_base_url
                or receipt.get("level") != 2
                or receipt.get("native_sleep_returned") is not True
                or receipt.get("is_sleeping") is not True):
            raise RuntimeError("Missing or mismatched native backend sleep receipt")
        gate = self._gate()
        # Native sleep aborts running requests and fences queued old work by
        # backend epoch. Wait for their local forwarding coroutines to exit too.
        deadline = asyncio.get_running_loop().time() + 30
        while True:
            state = gate.snapshot()
            if state["open"] or state["epoch"] != epoch:
                raise RuntimeError("Recovery entrypoint was not sealed")
            if set(state["active"]) == set(state["failures"]):
                break
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("Local HTTP forwarding did not finish after backend sleep")
            await asyncio.sleep(0.05)
        ids = set(state["failures"])
        if not ids:
            return dict(recovered=[], gate=state)
        known = getattr(self.server, "_forward_timeouts", {})
        if not ids.issubset(known) or any(not known[i]["forwarding_exited"] for i in ids):
            raise RuntimeError("Unknown non-timeout failures remain unsafe")
        # Whole failed Agent executions are excluded by the existing recipe.
        # Never retain a failed turn as a successful training trace/reward.
        sessions = list({known[i]["session_id"] for i in ids})
        store = get_existing_store()
        if store is None:
            raise RuntimeError("Cannot certify trace cleanup without the existing store")
        await store.release_sessions.remote(sessions)
        for session_id in sessions:
            if await store.keys.remote(session_id):
                raise RuntimeError("Failed Agent trace resources remain live")
        terminals = getattr(self.server, "_recovery_terminals", {})
        for request_id in ids:
            terminals[request_id] = dict(outcome="failed_after_backend_sleep",
                                        session_id=known[request_id]["session_id"],
                                        barrier_epoch=epoch, reward_valid=False)
        self.server._recovery_terminals = terminals
        result = await gate.settle_after_backend_sleep(epoch, ids)
        for request_id in ids:
            del known[request_id]
        get_logger().warning(f"backend_timeout_recovered epoch={epoch} requests={sorted(ids)}")
        return dict(recovered=sorted(ids), gate=result, failed_terminals=terminals)

    async def install_request_version(self, epoch: int, model_step: int):
        await self._gate().install_version(epoch, model_step)

    async def reopen_requests(self, epoch: int):
        await self._gate().reopen(epoch)

    async def stop(self) -> None:
        if self.server is not None:
            await self.server.stop()
            self.server = None
