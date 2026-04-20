from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any


_CAPTURE_LOCK = threading.RLock()
_NO_API_KEY_CAPTURE_FILE_NAME = "api_key_none.jsonl"


def resolve_capture_output_path(folder: str | Path, api_key: str | None = None) -> Path:
    if not api_key:
        return Path(folder) / _NO_API_KEY_CAPTURE_FILE_NAME
    api_key_hash = sha256(api_key.encode("utf-8")).hexdigest()[:16]
    return Path(folder) / f"api_key_{api_key_hash}.jsonl"


def append_gateway_capture_record(folder: str | Path, record: dict[str, Any], api_key: str | None = None) -> None:
    capture_path = resolve_capture_output_path(folder, api_key=api_key)
    capture_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "gateway_turn",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **record,
    }
    with _CAPTURE_LOCK:
        with capture_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def render_blocks_as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        rendered_parts = [render_blocks_as_text(item) for item in value]
        return "\n".join(part for part in rendered_parts if part)
    if isinstance(value, dict):
        block_type = value.get("type")
        if block_type == "text":
            return str(value.get("text", ""))
        if block_type == "tool_use":
            name = value.get("name", "")
            input_payload = json.dumps(value.get("input", {}), ensure_ascii=False, sort_keys=True)
            return f"<tool_use name={name}>{input_payload}</tool_use>"
        if block_type == "tool_result":
            tool_use_id = value.get("tool_use_id", "")
            content = render_blocks_as_text(value.get("content"))
            return f"<tool_result id={tool_use_id}>{content}</tool_result>"
        if "content" in value:
            return render_blocks_as_text(value["content"])
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)
