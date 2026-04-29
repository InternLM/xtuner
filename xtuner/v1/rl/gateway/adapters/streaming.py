from __future__ import annotations

import json
from typing import Any

from fastapi.responses import StreamingResponse


def encode_sse_event(data: Any, *, event: str | None = None) -> str:
    if isinstance(data, str):
        payload = data
    else:
        payload = json.dumps(data, ensure_ascii=False)

    lines: list[str] = []
    if event is not None:
        lines.append(f"event: {event}")
    if payload:
        lines.extend(f"data: {line}" for line in payload.splitlines())
    else:
        lines.append("data:")
    return "\n".join(lines) + "\n\n"


def build_sse_response(event_iterator) -> StreamingResponse:
    return StreamingResponse(
        event_iterator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
