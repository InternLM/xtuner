from __future__ import annotations

from typing import cast

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from ..adapters import (
    AnthropicChatAdapter,
    AnthropicChatAdapterError,
    AnthropicMessagesRequest,
    ChatCompletionRequest,
    OpenAIChatAdapter,
    OpenAIChatAdapterError,
    ResponsesRequest,
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
