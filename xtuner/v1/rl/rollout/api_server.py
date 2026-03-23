import json
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from xtuner.v1.data_proto.rl_data import RolloutState, Status

from .claude_chat import ClaudeChatAdapterError, ClaudeMessagesRequest, ClaudeMessagesResponse
from .openai_chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    OpenAIChatAdapterError,
    ensure_rollout_request_id,
)


def _ensure_request_id(rollout_state: RolloutState) -> str:
    return ensure_rollout_request_id(rollout_state)


def _extract_bearer_token(request: Request) -> str | None:
    authorization = request.headers.get("Authorization", "")
    if not authorization.startswith("Bearer "):
        return None
    return authorization.removeprefix("Bearer ").strip()


def _is_authorized(request: Request, api_key: str | list[str] | None) -> bool:
    if api_key is None:
        return True
    token = _extract_bearer_token(request)
    if token is None:
        return False
    if isinstance(api_key, list):
        return token in api_key
    return token == api_key


def _error_response(
    status_code: int,
    message: str,
    error_type: str,
    code: str,
    request_id: str | None = None,
) -> JSONResponse:
    payload = {
        "error": {
            "message": message,
            "type": error_type,
            "code": code,
            "request_id": request_id,
        }
    }
    return JSONResponse(status_code=status_code, content=payload)


def _authorize_http_request(
    http_request: Request,
    api_key: str | list[str] | None,
    request_id: str | None = None,
) -> JSONResponse | None:
    if _is_authorized(http_request, api_key):
        return None
    return _error_response(
        401,
        "Invalid or missing bearer token",
        "authentication_error",
        "unauthorized",
        request_id,
    )


def _claude_error_response(
    status_code: int,
    message: str,
    error_type: str,
    request_id: str | None = None,
) -> JSONResponse:
    payload = {
        "type": "error",
        "error": {
            "type": error_type,
            "message": message,
        },
    }
    if request_id is not None:
        payload["request_id"] = request_id
    return JSONResponse(status_code=status_code, content=payload)


def create_rollout_api_app(
    rollout_controller: Any,
    logger,
    api_key: str | list[str] | None = None,
) -> FastAPI:
    """Build the rollout API app around the provided rollout controller."""
    app = FastAPI()

    @app.exception_handler(HTTPException)
    async def handle_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
        request_id = request.headers.get("X-Request-Id")
        if isinstance(exc.detail, dict) and "error" in exc.detail:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        return _error_response(
            status_code=exc.status_code,
            message=str(exc.detail),
            error_type="invalid_request_error" if exc.status_code < 500 else "server_error",
            code="http_error",
            request_id=request_id,
        )

    @app.post("/generate")
    async def generate(request: RolloutState, http_request: Request) -> RolloutState:
        request_id = _ensure_request_id(request)
        unauthorized = _authorize_http_request(http_request, api_key, request_id)
        if unauthorized is not None:
            return unauthorized
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
        header_request_id = http_request.headers.get("X-Request-Id")
        unauthorized = _authorize_http_request(http_request, api_key, header_request_id)
        if unauthorized is not None:
            raise HTTPException(status_code=401, detail=json.loads(unauthorized.body.decode()))

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
    async def claude_messages(request: ClaudeMessagesRequest, http_request: Request) -> ClaudeMessagesResponse:
        header_request_id = http_request.headers.get("X-Request-Id")
        unauthorized = _authorize_http_request(http_request, api_key, header_request_id)
        if unauthorized is not None:
            return _claude_error_response(
                401,
                "Invalid or missing bearer token",
                "authentication_error",
                header_request_id,
            )

        try:
            return await rollout_controller.claude_messages(request)
        except ClaudeChatAdapterError as exc:
            status_code = 400 if exc.error_type == "invalid_request_error" else 500
            return _claude_error_response(
                status_code,
                exc.message,
                exc.error_type,
                exc.request_id,
            )

    @app.get("/healthz")
    async def healthz(http_request: Request):
        unauthorized = _authorize_http_request(http_request, api_key, http_request.headers.get("X-Request-Id"))
        if unauthorized is not None:
            return unauthorized
        rollout_controller.check_health()
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(http_request: Request):
        unauthorized = _authorize_http_request(http_request, api_key, http_request.headers.get("X-Request-Id"))
        if unauthorized is not None:
            return unauthorized
        is_ready, payload = rollout_controller.get_ready_status()
        if is_ready:
            return {"status": "ready", **payload}
        return JSONResponse(status_code=503, content={"status": "not_ready", **payload})

    @app.get("/metadata")
    async def metadata(http_request: Request):
        unauthorized = _authorize_http_request(http_request, api_key, http_request.headers.get("X-Request-Id"))
        if unauthorized is not None:
            return unauthorized
        return rollout_controller.get_rollout_metadata()

    @app.post("/pause")
    async def pause(http_request: Request):
        unauthorized = _authorize_http_request(http_request, api_key, http_request.headers.get("X-Request-Id"))
        if unauthorized is not None:
            return unauthorized
        rollout_controller.pause_generation()
        return {"status": "ok", "action": "pause"}

    @app.post("/continue")
    async def continue_generation(http_request: Request):
        unauthorized = _authorize_http_request(http_request, api_key, http_request.headers.get("X-Request-Id"))
        if unauthorized is not None:
            return unauthorized
        rollout_controller.continue_generation()
        return {"status": "ok", "action": "continue"}

    @app.post("/offload")
    async def offload(http_request: Request):
        unauthorized = _authorize_http_request(http_request, api_key, http_request.headers.get("X-Request-Id"))
        if unauthorized is not None:
            return unauthorized
        rollout_controller.offload()
        return {"status": "ok", "action": "offload"}

    @app.post("/onload")
    async def onload(http_request: Request):
        unauthorized = _authorize_http_request(http_request, api_key, http_request.headers.get("X-Request-Id"))
        if unauthorized is not None:
            return unauthorized
        rollout_controller.onload()
        return {"status": "ok", "action": "onload"}

    @app.post("/shutdown")
    async def shutdown(http_request: Request):
        unauthorized = _authorize_http_request(http_request, api_key, http_request.headers.get("X-Request-Id"))
        if unauthorized is not None:
            return unauthorized
        rollout_controller.shutdown()
        return {"status": "ok", "action": "shutdown"}

    return app
