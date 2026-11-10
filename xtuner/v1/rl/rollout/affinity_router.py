"""Job-owned HTTP session affinity in front of the original SessionServers.

The mapping names a concrete DP entrypoint, not an EP group. This module never
generates tokens, retries a POST, or changes cache/trace/version state.
"""
from __future__ import annotations

import asyncio
from collections import Counter
import logging
import threading

import aiohttp
from aiohttp import web


_HOP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
                "te", "trailer", "transfer-encoding", "upgrade", "host", "content-length"}


def _headers(headers):
    blocked = _HOP_HEADERS | {x.strip().lower() for x in headers.get("Connection", "").split(",")}
    return [(k, v) for k, v in headers.items() if k.lower() not in blocked]


class SessionAffinityRouter:
    def __init__(self, *, max_sessions=1000000):
        self.backends: set[str] = set()
        self.bindings: dict[str, str] = {}
        # Tombstones prevent a late retry from starting a second Trial after cleanup.
        self.retired: set[str] = set()
        self.inflight = Counter()
        self.assigned = Counter()
        self.max_sessions = max_sessions
        self.client = None
        self.runner = None
        self.url = None
        self.logger = logging.getLogger(__name__)

    def replace_backends(self, urls):
        self.backends = {url.rstrip("/") for url in urls}
        # Do not discard bindings on pause or failure. A dead binding fails closed.

    def set_backend(self, url, active):
        if url:
            if active:
                self.backends.add(url.rstrip("/"))
            else:
                self.backends.discard(url.rstrip("/"))

    def retire_sessions(self, session_ids):
        for sid in map(str, session_ids):
            url = self.bindings.pop(sid, None)
            if url is not None:
                self.assigned[url] -= 1
            self.retired.add(sid)

    def acquire(self, sid):
        if sid in self.retired:
            raise web.HTTPGone(text='{"error":{"code":"session_retired"}}', content_type="application/json")
        url = self.bindings.get(sid) if sid is not None else None
        if url is not None and url not in self.backends:
            raise web.HTTPServiceUnavailable(text='{"error":{"code":"session_backend_unavailable"}}',
                                             content_type="application/json")
        if url is None:
            if not self.backends:
                raise web.HTTPServiceUnavailable(text='{"error":{"code":"no_active_backend"}}',
                                                 content_type="application/json")
            if sid is not None and len(self.bindings) + len(self.retired) >= self.max_sessions:
                raise web.HTTPServiceUnavailable(text='{"error":{"code":"session_capacity_exhausted"}}',
                                                 content_type="application/json")
            url = min(self.backends, key=lambda u: (self.inflight[u], self.assigned[u], u))
            if sid is not None:
                self.bindings[sid] = url
                self.assigned[url] += 1
                self.logger.info("Session affinity assigned session=%s backend=%s", sid, url)
        self.inflight[url] += 1
        return url

    async def handle(self, request):
        if request.method == "GET" and request.path == "/health":
            return web.json_response({"active_backends": len(self.backends)})
        # Only forward inference APIs. Control-plane gate/version operations remain Ray calls.
        if request.method == "GET" and request.path in {"/v1/models", "/models"}:
            body, sid = b"", None
        elif request.method == "POST" and request.path in {
            "/v1/chat/completions", "/v1/messages", "/v1/messages/count_tokens",
        }:
            body = await request.read()
            try:
                data = await request.json()
                sid = data.get("session_id")
                if isinstance(sid, bool) or not isinstance(sid, (str, int)) or str(sid) == "":
                    raise ValueError("missing session_id")
                sid = str(sid)
            except (ValueError, TypeError, AttributeError):
                raise web.HTTPBadRequest(text='{"error":{"code":"valid_session_id_required"}}',
                                         content_type="application/json")
        else:
            raise web.HTTPNotFound()
        url = self.acquire(sid)  # No await between selection, binding and load reservation.
        downstream = None
        try:
            async with self.client.request(request.method, url + str(request.rel_url), data=body,
                                           headers=_headers(request.headers), allow_redirects=False) as upstream:
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
            raise web.HTTPBadGateway(text='{"error":{"code":"session_backend_transport_error"}}',
                                     content_type="application/json")
        finally:
            self.inflight[url] -= 1

    async def start(self, host, port=0):
        self.client = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=30),
                                           connector=aiohttp.TCPConnector(limit=0),
                                           auto_decompress=False, trust_env=False)
        app = web.Application(client_max_size=16 * 1024**2)
        app.router.add_route("*", "/{path:.*}", self.handle)
        self.runner = web.AppRunner(app, handler_cancellation=True, shutdown_timeout=10, access_log=None)
        await self.runner.setup()
        site = web.TCPSite(self.runner, host, port)
        await site.start()
        self.url = f"http://{host}:{self.runner.addresses[0][1]}"
        return self.url

    async def stop(self):
        if self.runner is not None:
            await self.runner.cleanup()
        if self.client is not None:
            await self.client.close()


class SessionAffinityService:
    """Dedicated event loop: controller's blocking Ray calls cannot stall HTTP."""

    def __init__(self, host):
        self.loop = asyncio.new_event_loop()
        self.router = SessionAffinityRouter()
        self.thread = threading.Thread(target=self.loop.run_forever, name="session-affinity", daemon=True)
        self.thread.start()
        try:
            self.url = self.call(self.router.start(host))
        except BaseException:
            self.close()
            raise

    def call(self, coroutine):
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result(timeout=60)

    def invoke(self, method, *args):
        async def run():
            return getattr(self.router, method)(*args)
        return self.call(run())

    def close(self):
        try:
            self.call(self.router.stop())
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=60)
            if self.thread.is_alive():
                raise RuntimeError("session affinity thread did not stop")
            self.loop.close()
