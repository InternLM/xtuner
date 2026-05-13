from __future__ import annotations

import json
import hashlib
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import httpx
import ray
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from xtuner.v1.utils import get_logger

from .health_manager import RolloutHealthManagerProxy, RolloutWorkerRouteInfo
from .worker import RolloutConfig


@dataclass
class WorkerHttpRouterConfig:
    port: int
    host: str = "0.0.0.0"
    title: str = "XTuner Worker Router"
    version: str = "0.1.0"
    log_level: str = "warning"
    request_timeout: float | None = None
    stream_timeout: float | None = None


class WorkerHttpRouter:
    def __init__(
        self,
        health_manager: RolloutHealthManagerProxy,
        rollout_config: RolloutConfig,
        config: WorkerHttpRouterConfig,
    ) -> None:
        self.health_manager = health_manager
        self.rollout_config = rollout_config
        self.config = config
        timeout = config.request_timeout or rollout_config.rollout_timeout
        self.client = httpx.AsyncClient(timeout=timeout)
        self.stream_timeout = config.stream_timeout or rollout_config.rollout_timeout
        self.logger = get_logger(log_dir=rollout_config.worker_log_dir, tag="WorkerHttpRouter")

    async def health(self) -> tuple[bool, dict[str, Any]]:
        return await self.health_manager.get_ready_status.remote()

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
        route_info = await self.health_manager.get_worker_route_info.remote(session_id)
        if route_info is None:
            raise HTTPException(status_code=503, detail={"error": "No active rollout worker available."})

        url = f"{route_info.url.rstrip('/')}/v1/chat/completions"
        headers = self._forward_headers(request)
        if payload.get("stream") is True:
            return await self._stream_chat_completions(url, payload, headers, route_info)
        return await self._post_chat_completions(url, payload, headers, route_info)

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
        route_info: RolloutWorkerRouteInfo,
    ) -> Response:
        try:
            response = await self.client.post(url, json=payload, headers=headers)
        except Exception as exc:
            await self.health_manager.report_worker_failure.remote(route_info.rank, str(exc))
            raise HTTPException(status_code=502, detail={"error": str(exc)}) from exc

        if response.status_code >= 500:
            await self.health_manager.report_worker_failure.remote(route_info.rank, response.text)
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
        route_info: RolloutWorkerRouteInfo,
    ) -> StreamingResponse:
        async def stream_response():
            try:
                async with self.client.stream("POST", url, json=payload, headers=headers, timeout=self.stream_timeout) as response:
                    if response.status_code >= 500:
                        body = await response.aread()
                        await self.health_manager.report_worker_failure.remote(route_info.rank, body.decode(errors="replace"))
                        yield body
                        return
                    async for chunk in response.aiter_bytes():
                        yield chunk
            except Exception as exc:
                await self.health_manager.report_worker_failure.remote(route_info.rank, str(exc))
                self.logger.error(f"Streaming chat completion failed for worker {route_info.rank}: {exc}")
                yield f'data: {{"error": {json.dumps(str(exc))}}}\n\n'.encode()

        return StreamingResponse(stream_response(), media_type="text/event-stream")


def build_worker_http_router_app(
    health_manager: RolloutHealthManagerProxy,
    rollout_config: RolloutConfig,
    config: WorkerHttpRouterConfig,
) -> FastAPI:
    router = WorkerHttpRouter(health_manager=health_manager, rollout_config=rollout_config, config=config)
    app = FastAPI(title=config.title, version=config.version)
    app.state.worker_router = router

    @app.get("/livez")
    async def livez() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz():
        ready, details = await router.health()
        payload = {"ready": ready, "status": "ready" if ready else "unavailable", "details": details}
        if ready:
            return payload
        raise HTTPException(status_code=503, detail=payload)

    @app.get("/v1/models")
    async def models():
        return await router.models()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await router.chat_completions(request)

    return app


def serve_worker_http_router(app: FastAPI, config: WorkerHttpRouterConfig) -> None:
    _ensure_port_available(config)
    uvicorn.run(app, host=config.host, port=config.port, log_level=config.log_level)


def serve_worker_http_router_in_thread(app: FastAPI, config: WorkerHttpRouterConfig) -> threading.Thread:
    thread = threading.Thread(
        target=serve_worker_http_router,
        args=(app, config),
        daemon=True,
        name="worker-http-router",
    )
    thread.start()
    return thread


def wait_for_worker_http_router_ready(base_url: str, *, timeout_seconds: float = 180.0) -> None:
    deadline = time.time() + timeout_seconds
    last_error = None
    while time.time() < deadline:
        try:
            response = httpx.get(f"{base_url}/livez", timeout=5.0)
            if response.status_code == 200:
                return
            last_error = response.text
        except Exception as exc:
            last_error = exc
        time.sleep(1)
    raise TimeoutError(f"Worker router did not become ready at {base_url}: {last_error}")


def _ensure_port_available(config: WorkerHttpRouterConfig) -> None:
    host = "127.0.0.1" if config.host in ("", "0.0.0.0") else config.host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        if sock.connect_ex((host, config.port)) == 0:
            raise OSError(f"Worker router port already in use: {config.host}:{config.port}")
