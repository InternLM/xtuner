from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from xtuner.v1.data_proto.rl_data import RolloutState, Status

from .anthropic_chat import AnthropicChatAdapterError, AnthropicMessagesRequest, AnthropicMessagesResponse
from .openai_chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    OpenAIChatAdapterError,
)
from .utils import ensure_rollout_request_id


if TYPE_CHECKING:
    from .controller import RolloutControllerProxy


def _build_error_response(
    status_code: int,
    message: str,
    error_type: str,
    code: str | None = None,
    request_id: str | None = None,
    protocol: str = "openai",
) -> JSONResponse:
    if protocol == "anthropic":
        payload = {
            "type": "error",
            "error": {
                "type": error_type,
                "message": message,
            },
        }
        if request_id is not None:
            payload["request_id"] = request_id
    else:
        payload = {
            "error": {
                "message": message,
                "type": error_type,
                "code": code,
                "request_id": request_id,
            }
        }
    return JSONResponse(status_code=status_code, content=payload)


def create_rollout_api_app(
    rollout_controller: RolloutControllerProxy,
    logger: Any,
) -> FastAPI:
    """Build the rollout API app around the provided rollout controller."""
    app = FastAPI()

    @app.exception_handler(HTTPException)
    async def handle_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
        request_id = request.headers.get("X-Request-Id")
        if isinstance(exc.detail, dict) and "error" in exc.detail:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        return _build_error_response(
            status_code=exc.status_code,
            message=str(exc.detail),
            error_type="invalid_request_error" if exc.status_code < 500 else "server_error",
            code="http_error",
            request_id=request_id,
        )

    @app.post("/generate")
    async def generate(request: RolloutState) -> RolloutState:
        request_id = ensure_rollout_request_id(request)
        try:
            response = await rollout_controller.generate(request)
            if not response.extra_fields.get("request_id"):
                response.extra_fields["request_id"] = request_id
            return response
        except Exception as exc:
            logger.error(f"Generate failed in API server for request_id={request_id}: {exc}")
            request.status = Status.FAILED
            request.error_msg = f"Generate failed in API server with error: {str(exc)}"
            return request

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest, http_request: Request) -> ChatCompletionResponse:
        try:
            return await rollout_controller.chat(request)
        except OpenAIChatAdapterError as exc:
            status_code = 400 if exc.error_type == "invalid_request_error" else 500
            raise HTTPException(
                status_code=status_code,
                detail={
                    "error": {
                        "message": exc.message,
                        "type": exc.error_type,
                        "code": exc.code,
                        "request_id": exc.request_id,
                    }
                },
            )

    @app.post("/v1/messages")
    async def anthropic_messages(
        request: AnthropicMessagesRequest, http_request: Request
    ) -> AnthropicMessagesResponse:
        try:
            return await rollout_controller.anthropic_messages(request)
        except AnthropicChatAdapterError as exc:
            status_code = 400 if exc.error_type == "invalid_request_error" else 500
            return _build_error_response(
                status_code=status_code,
                message=exc.message,
                error_type=exc.error_type,
                request_id=exc.request_id,
                protocol="anthropic",
            )

    @app.get("/healthz")
    async def healthz():
        is_ready, payload = rollout_controller.get_ready_status()
        if is_ready:
            return {"status": "ok", **payload}
        return JSONResponse(status_code=503, content={"status": "not_ready", **payload})

    @app.get("/metadata")
    async def metadata():
        return rollout_controller.get_rollout_metadata()

    @app.post("/pause")
    async def pause():
        rollout_controller.pause_generation()
        return {"status": "ok", "action": "pause"}

    @app.post("/continue")
    async def continue_generation():
        rollout_controller.continue_generation()
        return {"status": "ok", "action": "continue"}

    @app.post("/offload")
    async def offload():
        rollout_controller.offload()
        return {"status": "ok", "action": "offload"}

    @app.post("/onload")
    async def onload():
        rollout_controller.onload()
        return {"status": "ok", "action": "onload"}

    @app.post("/shutdown")
    async def shutdown():
        rollout_controller.shutdown()
        return {"status": "ok", "action": "shutdown"}

    return app
