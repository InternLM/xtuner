from __future__ import annotations

from enum import Enum
from typing import Any, cast

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..adapters import (
    AnthropicChatAdapter,
    AnthropicChatAdapterError,
    AnthropicMessagesRequest,
    ChatCompletionRequest,
    ChatTraceRecord,
    ChatTraceStore,
    OpenAIChatAdapter,
    OpenAIChatAdapterError,
    ResponsesRequest,
    build_api_key_trace_key,
)
from ..adapters.responses import OpenAIResponsesAdapter
from ..backend.protocol import GatewayBackend
from ..core.exceptions import GatewayStateError


def get_openai_adapter(request: Request) -> OpenAIChatAdapter:
    adapter = getattr(request.app.state, "gateway_openai_adapter", None)
    if adapter is None:
        raise GatewayStateError("Gateway OpenAI adapter is not configured.")
    return cast(OpenAIChatAdapter, adapter)


def get_anthropic_adapter(request: Request) -> AnthropicChatAdapter:
    adapter = getattr(request.app.state, "gateway_anthropic_adapter", None)
    if adapter is None:
        raise GatewayStateError("Gateway Anthropic adapter is not configured.")
    return cast(AnthropicChatAdapter, adapter)


def get_responses_adapter(request: Request) -> OpenAIResponsesAdapter:
    adapter = getattr(request.app.state, "gateway_responses_adapter", None)
    if adapter is None:
        raise GatewayStateError("Gateway Responses adapter is not configured.")
    return cast(OpenAIResponsesAdapter, adapter)


def extract_api_key(request: Request) -> str | None:
    authorization = request.headers.get("authorization")
    if authorization:
        scheme, _, credentials = authorization.partition(" ")
        if scheme.lower() == "bearer" and credentials.strip():
            return credentials.strip()
        if authorization.strip():
            return authorization.strip()

    api_key = request.headers.get("x-api-key") or request.headers.get("api-key")
    if api_key and api_key.strip():
        return api_key.strip()
    return None


# ---------------------------------------------------------------------------
# Runtime router  (/livez, /readyz, /capabilities)
# ---------------------------------------------------------------------------


def build_runtime_router() -> APIRouter:
    router = APIRouter()

    @router.get("/livez")
    async def livez() -> dict[str, str]:
        return {"status": "ok"}

    @router.get("/readyz")
    async def readyz(request: Request):
        backend = _get_backend(request)
        health = await backend.health()
        payload = health.model_dump(mode="json")
        if health.ready:
            return payload
        return JSONResponse(status_code=503, content=payload)

    @router.get("/capabilities")
    async def get_capabilities(request: Request):
        backend = _get_backend(request)
        capabilities = await backend.get_capabilities()
        return capabilities.model_dump(mode="json")

    return router


def _get_backend(request: Request) -> GatewayBackend:
    backend = getattr(request.app.state, "gateway_backend", None)
    if backend is None:
        raise GatewayStateError("Gateway backend is not configured.")
    return cast(GatewayBackend, backend)


# ---------------------------------------------------------------------------
# Trace store router  (/trace_store)
# ---------------------------------------------------------------------------


def build_trace_store_router() -> APIRouter:
    router = APIRouter()

    @router.get("/trace_store")
    async def get_trace_records(
        request: Request,
        trace_key: str | None = Query(default=None),
    ) -> dict:
        trace_store = _get_trace_store(request)
        resolved_trace_key = _resolve_trace_key(request, trace_key)
        records = trace_store.get(resolved_trace_key)
        return _build_trace_store_response(resolved_trace_key, records)

    @router.post("/trace_store/pop")
    async def pop_trace_records(
        request: Request,
        trace_key: str | None = Query(default=None),
    ) -> dict:
        trace_store = _get_trace_store(request)
        resolved_trace_key = _resolve_trace_key(request, trace_key)
        records = trace_store.pop(resolved_trace_key)
        return _build_trace_store_response(resolved_trace_key, records)

    @router.post("/trace_store/clear")
    async def clear_trace_records(
        request: Request,
        trace_key: str | None = Query(default=None),
    ) -> dict:
        trace_store = _get_trace_store(request)
        resolved_trace_key = _resolve_trace_key(request, trace_key)
        trace_store.clear(resolved_trace_key)
        return {
            "trace_key": resolved_trace_key,
            "cleared": True,
        }

    return router


def _get_trace_store(request: Request) -> ChatTraceStore:
    trace_store = getattr(request.app.state, "gateway_trace_store", None)
    if trace_store is None:
        raise GatewayStateError("Gateway trace store is not configured.")
    return cast(ChatTraceStore, trace_store)


def _resolve_trace_key(request: Request, trace_key: str | None) -> str:
    if trace_key:
        return trace_key
    return build_api_key_trace_key(extract_api_key(request))


def _build_trace_store_response(trace_key: str, records: list[ChatTraceRecord]) -> dict[str, Any]:
    return {
        "trace_key": trace_key,
        "count": len(records),
        "records": [_serialize_trace_record(record) for record in records],
    }


def _serialize_trace_record(record: ChatTraceRecord) -> dict[str, Any]:
    return {
        "trace_key": record.trace_key,
        "request_snapshot": _serialize_trace_value(record.request_snapshot),
        "response_snapshot": _serialize_trace_value(record.response_snapshot),
        "prompt_ids": list(record.prompt_ids),
        "response_ids": list(record.response_ids),
        "input_text": record.input_text,
        "output_text": record.output_text,
        "logprobs": _serialize_trace_value(record.logprobs),
        "routed_experts": _serialize_trace_value(record.routed_experts),
        "finish_reason": record.finish_reason,
        "status": _serialize_trace_value(record.status),
        "sequence": record.sequence,
        "created_at": record.created_at,
        "request_id": record.request_id,
    }


def _serialize_trace_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, BaseModel):
        try:
            return _serialize_trace_value(value.model_dump(mode="json", exclude_none=True))
        except Exception:
            return _serialize_trace_value(value.model_dump(mode="python", exclude_none=True))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _serialize_trace_value(val) for key, val in value.items() if val is not None}
    if isinstance(value, (list, tuple, set)):
        return [_serialize_trace_value(item) for item in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    try:
        import ray

        if isinstance(value, ray.ObjectRef):
            return str(value)
    except Exception:
        pass
    if hasattr(value, "tolist"):
        try:
            return _serialize_trace_value(value.tolist())
        except Exception:
            pass
    return str(value)


# ---------------------------------------------------------------------------
# OpenAI Chat Completions router  (/v1/chat/completions)
# ---------------------------------------------------------------------------


def build_openai_router() -> APIRouter:
    router = APIRouter()

    @router.post("/v1/chat/completions")
    async def chat_completions(
        request_body: ChatCompletionRequest,
        request: Request,
        adapter: OpenAIChatAdapter = Depends(get_openai_adapter),
    ):
        try:
            return await adapter.chat(request_body, api_key=extract_api_key(request))
        except OpenAIChatAdapterError as exc:
            return JSONResponse(
                status_code=400 if exc.error_type == "invalid_request_error" else 500,
                content={"error": {"message": exc.message, "type": exc.error_type, "code": exc.code}},
            )

    return router


# ---------------------------------------------------------------------------
# Anthropic Messages router  (/v1/messages)
# ---------------------------------------------------------------------------


def build_anthropic_router() -> APIRouter:
    router = APIRouter()

    @router.post("/v1/messages")
    async def messages(
        request_body: AnthropicMessagesRequest,
        request: Request,
        adapter: AnthropicChatAdapter = Depends(get_anthropic_adapter),
    ):
        try:
            return await adapter.messages(request_body, api_key=extract_api_key(request))
        except AnthropicChatAdapterError as exc:
            return JSONResponse(
                status_code=400 if exc.error_type == "invalid_request_error" else 500,
                content={"type": "error", "error": {"type": exc.error_type, "message": exc.message}},
            )

    return router


# ---------------------------------------------------------------------------
# OpenAI Responses router  (/v1/responses)
# ---------------------------------------------------------------------------


def build_responses_router() -> APIRouter:
    router = APIRouter()

    @router.post("/v1/responses")
    async def responses(
        request_body: ResponsesRequest,
        request: Request,
        adapter: OpenAIResponsesAdapter = Depends(get_responses_adapter),
    ):
        try:
            return await adapter.responses(request_body, api_key=extract_api_key(request))
        except OpenAIChatAdapterError as exc:
            return JSONResponse(
                status_code=400 if exc.error_type == "invalid_request_error" else 500,
                content={"error": {"message": exc.message, "type": exc.error_type, "code": exc.code}},
            )

    return router
