from __future__ import annotations

import json
import hashlib
import socket
import threading
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from xtuner.v1.utils import get_logger

from .session_worker_selector import RolloutWorkerHandle, RolloutWorkerUrlSource, SessionWorkerSelector
from ..worker import RolloutConfig


@dataclass
class InternalRolloutHttpEntryConfig:
    port: int
    host: str = "0.0.0.0"
    title: str = "XTuner Internal Rollout Router"
    version: str = "0.1.0"
    log_level: str = "warning"
    request_timeout: float | None = None
    stream_timeout: float | None = None
    worker_url_source: RolloutWorkerUrlSource = "backend"


class InternalRolloutHttpEntry:
    def __init__(
        self,
        worker_handles: list[RolloutWorkerHandle],
        rollout_config: RolloutConfig,
        config: InternalRolloutHttpEntryConfig,
    ) -> None:
        self.worker_selector = SessionWorkerSelector(worker_handles)
        self.rollout_config = rollout_config
        self.config = config
        timeout = config.request_timeout or rollout_config.rollout_timeout
        self.client = httpx.AsyncClient(timeout=timeout)
        self.stream_timeout = config.stream_timeout or rollout_config.rollout_timeout
        self.logger = get_logger(log_dir=rollout_config.worker_log_dir, tag="InternalRolloutHttpEntry")

    async def models(self) -> dict[str, Any]:
        model_id = self.rollout_config.model_name or "xtuner-rollout"
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "xtuner",
                }
            ],
        }

    async def chat_completions(self, request: Request) -> Response:
        payload = await request.json()
        session_id = self._extract_session_id(payload, request)
        worker = await self.worker_selector.select(session_id)
        if worker is None:
            raise HTTPException(status_code=503, detail={"error": "No active rollout worker available."})

        if self.config.worker_url_source == "session":
            payload.setdefault("session_id", session_id)
        try:
            worker_base_url = worker.get_generate_url(self.config.worker_url_source)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=503, detail={"error": str(exc)}) from exc
        url = f"{worker_base_url.rstrip('/')}/v1/chat/completions"
        headers = self._forward_headers(request)
        if payload.get("stream") is True:
            return await self._stream_chat_completions(url, payload, headers, worker)
        return await self._post_chat_completions(url, payload, headers, worker)

    def _extract_session_id(self, payload: dict[str, Any], request: Request) -> int:
        for header_name in ("x-session-uid", "x-session-id", "x-request-id"):
            header_value = request.headers.get(header_name)
            if header_value:
                return self._stable_int(header_value)

        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            for key in ("session_uid", "session_id", "conversation_id", "thread_id"):
                if key in metadata and metadata[key] is not None:
                    return self._stable_int(metadata[key])

        for key in ("session_uid", "session_id"):
            if key in payload and payload[key] is not None:
                return self._stable_int(payload[key])

        return uuid4().int

    def _stable_int(self, value: Any) -> int:
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return uuid4().int if not value else self._hash_to_int(value)
        return self._hash_to_int(json.dumps(value, sort_keys=True, default=str))

    def _hash_to_int(self, value: str) -> int:
        return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:16], byteorder="big")

    def _forward_headers(self, request: Request) -> dict[str, str]:
        ignored = {
            "host",
            "content-length",
            "connection",
            "keep-alive",
            "proxy-authenticate",
            "proxy-authorization",
            "te",
            "trailers",
            "transfer-encoding",
            "upgrade",
        }
        headers = {key: value for key, value in request.headers.items() if key.lower() not in ignored}
        if "content-type" not in {key.lower() for key in headers}:
            headers["content-type"] = "application/json"
        return headers

    async def _post_chat_completions(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        worker: RolloutWorkerHandle,
    ) -> Response:
        try:
            response = await self.client.post(url, json=payload, headers=headers)
        except Exception as exc:
            raise HTTPException(status_code=502, detail={"error": str(exc)}) from exc

        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "application/json"),
        )

    async def _stream_chat_completions(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        worker: RolloutWorkerHandle,
    ) -> StreamingResponse:
        async def stream_response():
            try:
                async with self.client.stream("POST", url, json=payload, headers=headers, timeout=self.stream_timeout) as response:
                    if response.status_code >= 500:
                        body = await response.aread()
                        yield body
                        return
                    async for chunk in response.aiter_bytes():
                        yield chunk
            except Exception as exc:
                self.logger.error(f"Streaming chat completion failed for worker {worker.rank}: {exc}")
                yield f'data: {{"error": {json.dumps(str(exc))}}}\n\n'.encode()

        return StreamingResponse(stream_response(), media_type="text/event-stream")


def build_internal_rollout_http_entry_app(
    worker_handles: list[RolloutWorkerHandle],
    rollout_config: RolloutConfig,
    config: InternalRolloutHttpEntryConfig,
) -> FastAPI:
    entry = InternalRolloutHttpEntry(worker_handles=worker_handles, rollout_config=rollout_config, config=config)
    app = FastAPI(title=config.title, version=config.version)
    app.state.internal_rollout_http_entry = entry

    @app.get("/v1/models")
    async def models():
        return await entry.models()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await entry.chat_completions(request)

    return app


def serve_internal_rollout_http_entry(app: FastAPI, config: InternalRolloutHttpEntryConfig) -> None:
    _ensure_port_available(config)
    uvicorn.run(app, host=config.host, port=config.port, log_level=config.log_level)


def serve_internal_rollout_http_entry_in_thread(
    app: FastAPI, config: InternalRolloutHttpEntryConfig
) -> threading.Thread:
    thread = threading.Thread(
        target=serve_internal_rollout_http_entry,
        args=(app, config),
        daemon=True,
        name="internal-rollout-http-entry",
    )
    thread.start()
    return thread


def _ensure_port_available(config: InternalRolloutHttpEntryConfig) -> None:
    host = "127.0.0.1" if config.host in ("", "0.0.0.0") else config.host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        if sock.connect_ex((host, config.port)) == 0:
            raise OSError(f"Internal rollout HTTP entry port already in use: {config.host}:{config.port}")
