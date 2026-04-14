from __future__ import annotations

import json
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from xtuner.v1.data_proto.rl_data import RolloutState, Status

from .chat_adapter import (
    AnthropicChatAdapterError,
    AnthropicCountTokensRequest,
    AnthropicCountTokensResponse,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
    OpenAIChatAdapterError,
    ResponsesRequest,
    ResponsesResponse,
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

    @app.post("/v1/responses", response_model=None)
    async def responses(request: ResponsesRequest, http_request: Request):
        try:
            if request.stream:
                non_stream_request = request.model_copy(update={"stream": False})
                response = await rollout_controller.responses(non_stream_request)
                return StreamingResponse(
                    _iter_openai_responses_sse_events(response),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
                )
            return await rollout_controller.responses(request)
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

    @app.post("/v1/messages", response_model=None)
    async def anthropic_messages(
        request: AnthropicMessagesRequest, http_request: Request
    ):
        try:
            if request.stream:
                non_stream_request = request.model_copy(update={"stream": False})
                response = await rollout_controller.anthropic_messages(non_stream_request)
                return StreamingResponse(
                    _iter_anthropic_sse_events(response),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
                )
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

    @app.post("/v1/messages/count_tokens")
    async def anthropic_count_tokens(request: AnthropicCountTokensRequest) -> AnthropicCountTokensResponse:
        return await rollout_controller.anthropic_count_tokens(request)

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


def _iter_anthropic_sse_events(response: AnthropicMessagesResponse) -> Iterator[str]:
    output_tokens = response.usage.output_tokens
    message_start = {
        "type": "message_start",
        "message": {
            "id": response.id,
            "type": response.type,
            "role": response.role,
            "content": [],
            "model": response.model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": 1 if output_tokens > 0 else 0,
            },
        },
    }
    yield _format_sse("message_start", message_start)

    chunk_size = 64
    for index, block in enumerate(response.content):
        block_type = block.get("type")
        if block_type == "text":
            yield _format_sse(
                "content_block_start",
                {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}},
            )
            text = str(block.get("text", ""))
            for offset in range(0, len(text), chunk_size):
                chunk = text[offset : offset + chunk_size]
                yield _format_sse(
                    "content_block_delta",
                    {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": chunk}},
                )
            yield _format_sse("content_block_stop", {"type": "content_block_stop", "index": index})
        elif block_type == "tool_use":
            yield _format_sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": block["id"],
                        "name": block["name"],
                        "input": {},
                    },
                },
            )
            input_json = json.dumps(block.get("input", {}), ensure_ascii=False)
            for offset in range(0, len(input_json), chunk_size):
                chunk = input_json[offset : offset + chunk_size]
                yield _format_sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "input_json_delta", "partial_json": chunk},
                    },
                )
            yield _format_sse("content_block_stop", {"type": "content_block_stop", "index": index})

    yield _format_sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": response.stop_reason,
                "stop_sequence": response.stop_sequence,
            },
            "usage": {
                "output_tokens": response.usage.output_tokens,
            },
        },
    )
    yield _format_sse("message_stop", {"type": "message_stop"})


def _format_sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _iter_openai_responses_sse_events(response: ResponsesResponse) -> Iterator[str]:
    sequence_number = 0
    response_snapshot = response.model_dump(mode="python")
    in_progress_response = {**response_snapshot, "status": "in_progress"}
    yield _format_openai_response_sse({"type": "response.created", "sequence_number": sequence_number, "response": in_progress_response})
    sequence_number += 1

    for output_index, item in enumerate(response.output):
        item_type = item.get("type")
        if item_type == "message":
            yield _format_openai_response_sse(
                {
                    "type": "response.output_item.added",
                    "sequence_number": sequence_number,
                    "output_index": output_index,
                    "item": item,
                }
            )
            sequence_number += 1
            for content_index, part in enumerate(item.get("content", [])):
                if part.get("type") != "output_text":
                    continue
                yield _format_openai_response_sse(
                    {
                        "type": "response.content_part.added",
                        "sequence_number": sequence_number,
                        "output_index": output_index,
                        "item_id": item["id"],
                        "content_index": content_index,
                        "part": {"type": "output_text", "text": "", "annotations": []},
                    }
                )
                sequence_number += 1
                text = str(part.get("text", ""))
                chunk_size = 64
                for offset in range(0, len(text), chunk_size):
                    yield _format_openai_response_sse(
                        {
                            "type": "response.output_text.delta",
                            "sequence_number": sequence_number,
                            "output_index": output_index,
                            "item_id": item["id"],
                            "content_index": content_index,
                            "delta": text[offset : offset + chunk_size],
                        }
                    )
                    sequence_number += 1
                yield _format_openai_response_sse(
                    {
                        "type": "response.output_text.done",
                        "sequence_number": sequence_number,
                        "output_index": output_index,
                        "item_id": item["id"],
                        "content_index": content_index,
                        "text": text,
                    }
                )
                sequence_number += 1
                yield _format_openai_response_sse(
                    {
                        "type": "response.content_part.done",
                        "sequence_number": sequence_number,
                        "output_index": output_index,
                        "item_id": item["id"],
                        "content_index": content_index,
                        "part": {"type": "output_text", "text": text, "annotations": part.get("annotations", [])},
                    }
                )
                sequence_number += 1
            yield _format_openai_response_sse(
                {
                    "type": "response.output_item.done",
                    "sequence_number": sequence_number,
                    "output_index": output_index,
                    "item": item,
                }
            )
            sequence_number += 1
        elif item_type == "function_call":
            added_item = {**item, "arguments": "", "status": "in_progress"}
            yield _format_openai_response_sse(
                {
                    "type": "response.output_item.added",
                    "sequence_number": sequence_number,
                    "output_index": output_index,
                    "item": added_item,
                }
            )
            sequence_number += 1
            arguments = str(item.get("arguments", ""))
            chunk_size = 64
            for offset in range(0, len(arguments), chunk_size):
                yield _format_openai_response_sse(
                    {
                        "type": "response.function_call_arguments.delta",
                        "sequence_number": sequence_number,
                        "output_index": output_index,
                        "item_id": item["id"],
                        "delta": arguments[offset : offset + chunk_size],
                    }
                )
                sequence_number += 1
            yield _format_openai_response_sse(
                {
                    "type": "response.function_call_arguments.done",
                    "sequence_number": sequence_number,
                    "output_index": output_index,
                    "item_id": item["id"],
                    "arguments": arguments,
                    "name": item.get("name"),
                }
            )
            sequence_number += 1
            yield _format_openai_response_sse(
                {
                    "type": "response.output_item.done",
                    "sequence_number": sequence_number,
                    "output_index": output_index,
                    "item": item,
                }
            )
            sequence_number += 1

    yield _format_openai_response_sse(
        {"type": "response.completed", "sequence_number": sequence_number, "response": response_snapshot}
    )
    yield "data: [DONE]\n\n"


def _format_openai_response_sse(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
