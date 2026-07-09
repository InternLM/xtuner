from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from xtuner.tools.trace_viewer.payload import load_jaeger_traces_from_otel_jsonl


JAEGER_DEFAULT_QUERY_URL = "http://127.0.0.1:16686"
JAEGER_DEFAULT_LOOKBACK_S = 60 * 60
JAEGER_DEFAULT_LIMIT = 500


@dataclass(frozen=True)
class JaegerQuerySource:
    query_url: str
    service_name: str
    lookback_s: int = JAEGER_DEFAULT_LOOKBACK_S
    limit: int = JAEGER_DEFAULT_LIMIT
    timeout_s: float = 5.0

    def load(self) -> list[dict[str, Any]]:
        return fetch_jaeger_traces(
            self.query_url,
            service_name=self.service_name,
            lookback_s=self.lookback_s,
            limit=self.limit,
            timeout_s=self.timeout_s,
        )


@dataclass(frozen=True)
class JsonlTraceSource:
    trace_jsonl_path: Path | str

    def load(self) -> list[dict[str, Any]]:
        return load_jaeger_traces_from_otel_jsonl(self.trace_jsonl_path)


def fetch_jaeger_traces(
    jaeger_query_url: str,
    *,
    service_name: str,
    lookback_s: int = JAEGER_DEFAULT_LOOKBACK_S,
    limit: int = JAEGER_DEFAULT_LIMIT,
    timeout_s: float = 5.0,
) -> list[dict[str, Any]]:
    base_url = require_jaeger_query_url(jaeger_query_url)
    query = urlencode(
        {
            "service": service_name,
            "lookback": f"{max(1, lookback_s)}s",
            "limit": max(1, limit),
        }
    )
    request = Request(f"{base_url}/api/traces?{query}", headers={"Accept": "application/json"})
    with urlopen(request, timeout=timeout_s) as response:
        payload = json.loads(response.read().decode("utf-8"))
    traces = payload.get("data") if isinstance(payload, dict) else None
    return traces if isinstance(traces, list) else []


def normalize_jaeger_query_url(jaeger_query_url: str | None) -> str | None:
    if jaeger_query_url is None:
        return None
    stripped = jaeger_query_url.strip()
    return stripped.rstrip("/") if stripped else None


def require_jaeger_query_url(jaeger_query_url: str | None) -> str:
    normalized = normalize_jaeger_query_url(jaeger_query_url)
    if normalized is None:
        raise ValueError("jaeger_query_url is required")
    return normalized


__all__ = [
    "JAEGER_DEFAULT_LIMIT",
    "JAEGER_DEFAULT_LOOKBACK_S",
    "JAEGER_DEFAULT_QUERY_URL",
    "JaegerQuerySource",
    "JsonlTraceSource",
    "fetch_jaeger_traces",
    "normalize_jaeger_query_url",
    "require_jaeger_query_url",
]
