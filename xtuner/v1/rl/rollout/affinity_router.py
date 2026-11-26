"""Job-owned HTTP session affinity in front of the original SessionServers.

The mapping names a concrete DP entrypoint, not an EP group. A session keeps using the backend it was first
bound to across weight updates: a dead binding fails closed with 503 and is never rebound or replayed. This
module never generates tokens, retries a POST, or changes cache/trace/version state.

The router owns a dedicated event loop in its own thread. The Controller process still performs blocking
``ray.get`` calls, so the controller's main thread cannot serve HTTP; the trainer's fit loop is not replacement.
"""

from __future__ import annotations

import asyncio
import threading
from collections import Counter
from typing import Any, Coroutine

import aiohttp
from aiohttp import web

from xtuner.v1.utils import get_logger


_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}

_INFERENCE_PATHS = {"/v1/chat/completions", "/v1/messages", "/v1/messages/count_tokens"}
_MODEL_PATHS = {"/v1/models", "/models"}


def _headers(headers) -> list[tuple[str, str]]:
    blocked = _HOP_HEADERS | {item.strip().lower() for item in headers.get("Connection", "").split(",")}
    return [(key, value) for key, value in headers.items() if key.lower() not in blocked]


class SessionAffinityRouter:
    """Route external HTTP sessions to fixed SessionServer entrypoints.

    Args:
        host (str): Host address for the router's HTTP listener.
        max_sessions (int): Capacity for live bindings plus retired tombstones.
    """

    def __init__(self, host: str, *, max_sessions: int = 1_000_000):
        self.backends: set[str] = set()
        self.bindings: dict[str, str] = {}
        # Tombstones prevent a late retry from starting a second Trial after cleanup.
        self.retired: set[str] = set()
        self.in_flight: Counter[str] = Counter()
        self.assigned: Counter[str] = Counter()
        self.max_sessions = max_sessions
        self.client: aiohttp.ClientSession | None = None
        self.runner: web.AppRunner | None = None
        self.logger = get_logger()

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, name="session-affinity", daemon=True)
        self._thread.start()
        try:
            self.url = self._call(self._start(host))
        except BaseException:
            self.close()
            raise

    def replace_backends(self, urls: list[str]) -> None:
        """Replace the active backend set without dropping existing session
        bindings."""

        async def update() -> None:
            self.backends = {url.rstrip("/") for url in urls}

        self._call(update())

    def set_backend(self, url: str | None, active: bool) -> None:
        """Add or remove one backend; existing bindings keep failing closed
        when it is dead."""

        async def update() -> None:
            if url:
                if active:
                    self.backends.add(url.rstrip("/"))
                else:
                    self.backends.discard(url.rstrip("/"))

        self._call(update())

    def retire_sessions(self, session_ids: list[str]) -> None:
        """Retire sessions so later requests get 410 instead of starting a
        second attempt."""

        async def update() -> None:
            for session_id in map(str, session_ids):
                url = self.bindings.pop(session_id, None)
                if url is not None:
                    self.assigned[url] -= 1
                self.retired.add(session_id)

        self._call(update())

    def close(self) -> None:
        """Stop the HTTP listener and the dedicated event loop."""
        try:
            self._call(self._stop())
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=60)
            if self._thread.is_alive():
                raise TimeoutError("Session affinity loop did not stop")

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()
        self._loop.close()

    def _call(self, coroutine: Coroutine) -> Any:
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result(timeout=60)

    def _acquire(self, session_id: str | None) -> str:
        if session_id in self.retired:
            raise web.HTTPGone(text='{"error":{"code":"session_retired"}}', content_type="application/json")
        url = self.bindings.get(session_id) if session_id is not None else None
        if url is not None and url not in self.backends:
            raise web.HTTPServiceUnavailable(
                text='{"error":{"code":"session_backend_unavailable"}}', content_type="application/json"
            )
        if url is None:
            if not self.backends:
                raise web.HTTPServiceUnavailable(
                    text='{"error":{"code":"no_active_backend"}}', content_type="application/json"
                )
            if session_id is not None and len(self.bindings) + len(self.retired) >= self.max_sessions:
                raise web.HTTPServiceUnavailable(
                    text='{"error":{"code":"session_capacity_exhausted"}}', content_type="application/json"
                )
            url = min(self.backends, key=lambda item: (self.in_flight[item], self.assigned[item], item))
            if session_id is not None:
                self.bindings[session_id] = url
                self.assigned[url] += 1
                self.logger.info("Session affinity assigned session=%s backend=%s", session_id, url)
        self.in_flight[url] += 1
        return url

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        if request.method == "GET" and request.path == "/health":
            return web.json_response({"active_backends": len(self.backends)})
        # Only forward inference APIs. Control-plane gate/version operations remain Ray calls.
        if request.method == "GET" and request.path in _MODEL_PATHS:
            body, session_id = b"", None
        elif request.method == "POST" and request.path in _INFERENCE_PATHS:
            body = await request.read()
            try:
                data = await request.json()
                session_id = data.get("session_id")
                if isinstance(session_id, bool) or not isinstance(session_id, (str, int)) or str(session_id) == "":
                    raise ValueError("missing session_id")
                session_id = str(session_id)
            except (ValueError, TypeError, AttributeError):
                raise web.HTTPBadRequest(
                    text='{"error":{"code":"valid_session_id_required"}}', content_type="application/json"
                )
        else:
            raise web.HTTPNotFound()
        assert self.client is not None
        # No await between selection, binding and load reservation.
        url = self._acquire(session_id)
        downstream = None
        try:
            async with self.client.request(
                request.method,
                url + str(request.rel_url),
                data=body,
                headers=_headers(request.headers),
                allow_redirects=False,
            ) as upstream:
                downstream = web.StreamResponse(status=upstream.status, headers=_headers(upstream.headers))
                await downstream.prepare(request)
                async for chunk in upstream.content.iter_any():
                    await downstream.write(chunk)
                await downstream.write_eof()
                return downstream
        except (aiohttp.ClientError, OSError, asyncio.TimeoutError):
            # Never infer worker death from one timeout and never replay an ambiguous POST.
            if downstream is not None and downstream.prepared:
                if request.transport is not None:
                    request.transport.close()
                return downstream
            raise web.HTTPBadGateway(
                text='{"error":{"code":"session_backend_transport_error"}}', content_type="application/json"
            )
        finally:
            self.in_flight[url] -= 1

    async def _start(self, host: str) -> str:
        self.client = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=30),
            connector=aiohttp.TCPConnector(limit=0),
            auto_decompress=False,
            trust_env=False,
        )
        app = web.Application(client_max_size=16 * 1024**2)
        app.router.add_route("*", "/{path:.*}", self._handle)
        self.runner = web.AppRunner(app, handler_cancellation=True, shutdown_timeout=10, access_log=None)
        await self.runner.setup()
        site = web.TCPSite(self.runner, host, 0)
        await site.start()
        return f"http://{host}:{self.runner.addresses[0][1]}"

    async def _stop(self) -> None:
        if self.runner is not None:
            await self.runner.cleanup()
        if self.client is not None:
            await self.client.close()
