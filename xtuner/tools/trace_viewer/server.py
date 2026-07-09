from __future__ import annotations

import argparse
import http.server
import json
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request, urlopen

from xtuner.tools.trace_viewer.payload import (
    build_rollout_view_payload_from_jaeger_traces,
    filter_rollout_view_payload_by_train_step,
    load_live_trace_records,
)
from xtuner.tools.trace_viewer.render import render_rollout_trace_html, write_rollout_trace_html
from xtuner.tools.trace_viewer.source import (
    JAEGER_DEFAULT_LIMIT,
    JAEGER_DEFAULT_LOOKBACK_S,
    JAEGER_DEFAULT_QUERY_URL,
    JaegerQuerySource,
    JsonlTraceSource,
    normalize_jaeger_query_url,
    require_jaeger_query_url,
)


_JAEGER_PROXY_PREFIX = "/jaeger"
_PROXY_TIMEOUT_S = 10.0
_HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


class TraceViewerHandle:
    def __init__(
        self,
        server: http.server.ThreadingHTTPServer,
        thread: threading.Thread,
        *,
        host: str,
        port: int,
        url: str,
    ) -> None:
        self.server = server
        self.thread = thread
        self.host = host
        self.port = port
        self.url = url

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


class _TraceViewerPayloadCache:
    def __init__(
        self,
        load_base_payload: Callable[[], dict[str, Any]],
        *,
        source_signature: Callable[[], Any] | None = None,
        max_age_s: float | None = None,
    ) -> None:
        self._load_base_payload = load_base_payload
        self._source_signature = source_signature or (lambda: None)
        self._max_age_s = max_age_s
        self._lock = threading.Lock()
        self._signature: Any = object()
        self._loaded_at_s = 0.0
        self._base_payload: dict[str, Any] | None = None
        self._payloads_by_step: dict[str, dict[str, Any]] = {}

    def get(self, train_step: str | int | None = "latest") -> dict[str, Any]:
        with self._lock:
            signature = self._source_signature()
            now = time.monotonic()
            expired = self._max_age_s is not None and now - self._loaded_at_s >= self._max_age_s
            if self._base_payload is None or signature != self._signature or expired:
                self._signature = signature
                self._base_payload = self._load_base_payload()
                self._loaded_at_s = now
                self._payloads_by_step.clear()
            cache_key = _train_step_cache_key(train_step)
            payload = self._payloads_by_step.get(cache_key)
            if payload is None:
                payload = filter_rollout_view_payload_by_train_step(self._base_payload, train_step)
                self._payloads_by_step[cache_key] = payload
            return payload


def _train_step_cache_key(train_step: str | int | None) -> str:
    if train_step is None:
        return "latest"
    text = str(train_step).strip()
    return text or "latest"


def fetch_jaeger_traces(
    jaeger_query_url: str,
    *,
    service_name: str,
    lookback_s: int = JAEGER_DEFAULT_LOOKBACK_S,
    limit: int = JAEGER_DEFAULT_LIMIT,
    timeout_s: float = 5.0,
) -> list[dict[str, Any]]:
    return JaegerQuerySource(
        query_url=jaeger_query_url,
        service_name=service_name,
        lookback_s=lookback_s,
        limit=limit,
        timeout_s=timeout_s,
    ).load()


def fetch_rollout_view_payload(
    jaeger_query_url: str,
    *,
    jaeger_link_url: str | None = None,
    live_jsonl_path: Path | str | None = None,
    service_name: str,
    run_id: str | None = None,
    lookback_s: int = JAEGER_DEFAULT_LOOKBACK_S,
    limit: int = JAEGER_DEFAULT_LIMIT,
    train_step: str | int | None = "latest",
) -> dict[str, Any]:
    traces = JaegerQuerySource(
        query_url=jaeger_query_url,
        service_name=service_name,
        lookback_s=lookback_s,
        limit=limit,
    ).load()
    payload = build_rollout_view_payload_from_jaeger_traces(
        traces,
        jaeger_query_url=jaeger_query_url,
        jaeger_link_url=jaeger_link_url,
        live_records=load_live_trace_records(live_jsonl_path),
        service_name=service_name,
        run_id=run_id,
        train_step=train_step,
    )
    payload["service_name"] = service_name
    payload["run_id"] = run_id
    payload["lookback_s"] = lookback_s
    payload["limit"] = limit
    return payload


def fetch_rollout_view_payload_from_trace_jsonl(
    trace_jsonl_path: Path | str,
    *,
    jaeger_query_url: str | None = None,
    jaeger_link_url: str | None = None,
    live_jsonl_path: Path | str | None = None,
    service_name: str | None = None,
    run_id: str | None = None,
    train_step: str | int | None = "latest",
) -> dict[str, Any]:
    traces = JsonlTraceSource(trace_jsonl_path).load()
    payload = build_rollout_view_payload_from_jaeger_traces(
        traces,
        jaeger_query_url=jaeger_query_url,
        jaeger_link_url=jaeger_link_url,
        live_records=load_live_trace_records(live_jsonl_path),
        service_name=service_name,
        run_id=run_id,
        train_step=train_step,
    )
    payload["source"] = "trace_jsonl"
    payload["trace_jsonl_path"] = os.fspath(Path(trace_jsonl_path).expanduser())
    payload["service_name"] = service_name
    payload["run_id"] = run_id
    return payload


def start_rollout_trace_viewer(
    jaeger_query_url: str | None = JAEGER_DEFAULT_QUERY_URL,
    *,
    jaeger_link_url: str | None = None,
    service_name: str,
    run_id: str | None = None,
    trace_jsonl_path: Path | str | None = None,
    live_jsonl_path: Path | str | None = None,
    payload_output_path: Path | str | None = None,
    host: str = "127.0.0.1",
    port: int = 0,
    refresh_interval_s: float = 2.0,
    lookback_s: int = JAEGER_DEFAULT_LOOKBACK_S,
    limit: int = JAEGER_DEFAULT_LIMIT,
    train_step: str | int | None = "latest",
) -> TraceViewerHandle:
    jaeger_query_url = normalize_jaeger_query_url(jaeger_query_url)
    jaeger_link_url = normalize_jaeger_query_url(jaeger_link_url)
    viewer_jaeger_link_url = jaeger_link_url or (_JAEGER_PROXY_PREFIX if jaeger_query_url is not None else None)
    if trace_jsonl_path is None:
        jaeger_query_url = require_jaeger_query_url(jaeger_query_url)

    def load_base_payload() -> dict[str, Any]:
        if trace_jsonl_path is not None:
            payload = fetch_rollout_view_payload_from_trace_jsonl(
                trace_jsonl_path,
                jaeger_query_url=jaeger_query_url,
                jaeger_link_url=viewer_jaeger_link_url,
                live_jsonl_path=live_jsonl_path,
                service_name=service_name,
                run_id=run_id,
                train_step="all",
            )
        else:
            payload = fetch_rollout_view_payload(
                jaeger_query_url,
                jaeger_link_url=viewer_jaeger_link_url,
                live_jsonl_path=live_jsonl_path,
                service_name=service_name,
                run_id=run_id,
                lookback_s=lookback_s,
                limit=limit,
                train_step="all",
            )
        return payload

    def current_source_signature() -> Any:
        if trace_jsonl_path is None:
            return ("jaeger", _source_signature(live_jsonl_path))
        return _source_signature(trace_jsonl_path, live_jsonl_path)

    payload_cache = _TraceViewerPayloadCache(
        load_base_payload,
        source_signature=current_source_signature,
        max_age_s=max(refresh_interval_s, 5.0) if trace_jsonl_path is None else None,
    )

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlsplit(self.path)
            path = parsed.path
            if path == _JAEGER_PROXY_PREFIX or path.startswith(f"{_JAEGER_PROXY_PREFIX}/"):
                self._proxy_jaeger()
                return
            if path in {"/", "/index.html"}:
                html_body = render_rollout_trace_html(
                    self._payload(self._query_train_step(parsed.query)),
                    live=True,
                    api_url="/api/trace",
                    refresh_interval_s=refresh_interval_s,
                )
                self._send_bytes(html_body.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/api/trace":
                self._send_json(self._payload(self._query_train_step(parsed.query)))
                return
            self.send_error(404)

        def _query_train_step(self, query: str) -> str | int | None:
            values = parse_qs(query).get("train_step")
            if not values:
                return train_step
            return values[-1]

        def _payload(self, selected_train_step: str | int | None) -> dict[str, Any]:
            payload = payload_cache.get(selected_train_step)
            if payload_output_path is not None:
                _write_payload_json(payload, payload_output_path)
            return payload

        def _send_json(self, payload: dict[str, Any]) -> None:
            self._send_bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json")

        def _proxy_jaeger(self) -> None:
            if jaeger_query_url is None:
                self.send_error(502, "Jaeger query URL is not configured")
                return
            try:
                target_url = _jaeger_proxy_target_url(jaeger_query_url, self.path)
                request = Request(
                    target_url,
                    headers={
                        "Accept": self.headers.get("Accept", "*/*"),
                        "User-Agent": self.headers.get("User-Agent", "XTunerTraceViewer"),
                    },
                )
                with urlopen(request, timeout=_PROXY_TIMEOUT_S) as response:
                    self._send_proxy_response(response.status, response.headers.items(), response.read())
            except HTTPError as exc:
                self._send_proxy_response(exc.code, exc.headers.items(), exc.read())
            except (OSError, URLError) as exc:
                self.send_error(502, f"Failed to proxy Jaeger request: {exc}")

        def _send_proxy_response(self, status: int, headers: Any, body: bytes) -> None:
            self.send_response(status)
            for key, value in headers:
                if key.lower() in _HOP_BY_HOP_HEADERS:
                    continue
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_bytes(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = http.server.ThreadingHTTPServer((host, port), Handler)
    server_host, server_port = server.server_address
    display_host = server_host or host
    thread = threading.Thread(target=server.serve_forever, name="XTunerRolloutTraceViewer", daemon=True)
    thread.start()
    return TraceViewerHandle(
        server=server,
        thread=thread,
        host=display_host,
        port=server_port,
        url=f"http://{display_host}:{server_port}",
    )


def _write_payload_json(payload: dict[str, Any], output_path: Path | str) -> None:
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _source_signature(*paths: Path | str | None) -> tuple[tuple[str, int | None, int | None], ...]:
    signatures = []
    for value in paths:
        if value is None:
            continue
        path = Path(value).expanduser()
        try:
            stat = path.stat()
        except OSError:
            signatures.append((os.fspath(path), None, None))
            continue
        signatures.append((os.fspath(path), stat.st_mtime_ns, stat.st_size))
    return tuple(signatures)


def _jaeger_proxy_target_url(jaeger_query_url: str, request_path: str) -> str:
    parsed = urlsplit(request_path)
    path = parsed.path
    if path == _JAEGER_PROXY_PREFIX:
        jaeger_path = "/"
    else:
        jaeger_path = path.removeprefix(_JAEGER_PROXY_PREFIX) or "/"
    target = f"{require_jaeger_query_url(jaeger_query_url)}{jaeger_path}"
    if parsed.query:
        target = f"{target}?{parsed.query}"
    return target


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve or render an XTuner rollout trace viewer backed by Jaeger or JSONL."
    )
    parser.add_argument("--jaeger-query-url", default=JAEGER_DEFAULT_QUERY_URL)
    parser.add_argument("--jaeger-link-url", default=None)
    parser.add_argument("--trace-jsonl", type=Path, default=None)
    parser.add_argument("--live-jsonl", type=Path, default=None)
    parser.add_argument("--service", "--service-name", dest="service", default="xtuner-rollout")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--lookback", type=int, default=JAEGER_DEFAULT_LOOKBACK_S)
    parser.add_argument("--limit", type=int, default=JAEGER_DEFAULT_LIMIT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--payload-output", type=Path, default=None)
    parser.add_argument("--train-step", default="latest", help="Initial train step to render: latest, all, or a step value.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.output is not None:
        if args.trace_jsonl is not None:
            payload = fetch_rollout_view_payload_from_trace_jsonl(
                args.trace_jsonl,
                jaeger_query_url=args.jaeger_query_url,
                jaeger_link_url=args.jaeger_link_url,
                live_jsonl_path=args.live_jsonl,
                service_name=args.service,
                run_id=args.run_id,
                train_step=args.train_step,
            )
        else:
            payload = fetch_rollout_view_payload(
                args.jaeger_query_url,
                jaeger_link_url=args.jaeger_link_url,
                live_jsonl_path=args.live_jsonl,
                service_name=args.service,
                run_id=args.run_id,
                lookback_s=args.lookback,
                limit=args.limit,
                train_step=args.train_step,
            )
        if args.payload_output is not None:
            _write_payload_json(payload, args.payload_output)
        write_rollout_trace_html(payload, args.output)
        print(args.output)
        if args.payload_output is not None:
            print(args.payload_output)
        return

    handle = start_rollout_trace_viewer(
        args.jaeger_query_url,
        jaeger_link_url=args.jaeger_link_url,
        service_name=args.service,
        run_id=args.run_id,
        trace_jsonl_path=args.trace_jsonl,
        live_jsonl_path=args.live_jsonl,
        payload_output_path=args.payload_output,
        host=args.host,
        port=args.port,
        lookback_s=args.lookback,
        limit=args.limit,
        train_step=args.train_step,
    )
    print(f"XTuner Rollout Trace Viewer: {handle.url}", flush=True)
    if args.trace_jsonl is not None:
        print(f"Trace JSONL: {args.trace_jsonl}", flush=True)
    if args.live_jsonl is not None:
        print(f"Live JSONL: {args.live_jsonl}", flush=True)
    jaeger_query_url = normalize_jaeger_query_url(args.jaeger_query_url)
    if jaeger_query_url is not None:
        print(f"Jaeger Trace Viewer: {jaeger_query_url}", flush=True)
        print(f"Jaeger Same-Origin Proxy: {handle.url}{_JAEGER_PROXY_PREFIX}/", flush=True)
    jaeger_link_url = normalize_jaeger_query_url(args.jaeger_link_url)
    if jaeger_link_url is not None:
        print(f"Jaeger Open Links: {jaeger_link_url}", flush=True)
    if args.payload_output is not None:
        print(f"Viewer Payload JSON: {args.payload_output}", flush=True)
    try:
        handle.thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        handle.close()


if __name__ == "__main__":
    main()


__all__ = [
    "TraceViewerHandle",
    "_TraceViewerPayloadCache",
    "build_rollout_view_payload_from_jaeger_traces",
    "fetch_jaeger_traces",
    "fetch_rollout_view_payload",
    "fetch_rollout_view_payload_from_trace_jsonl",
    "render_rollout_trace_html",
    "start_rollout_trace_viewer",
    "write_rollout_trace_html",
]
