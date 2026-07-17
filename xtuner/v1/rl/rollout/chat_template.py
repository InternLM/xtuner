import copy
import json
from typing import Any


_RAW_ARGUMENTS_KEY = "__xtuner_raw_arguments__"
_PROCESS_ONLY_MESSAGE_KEYS = ("finish_reason", "metainfo")


def canonicalize_messages_for_chat_template(messages: list[dict]) -> list[dict]:
    """Render a copied, trace-store-only view of agent messages.

    Some LLM tool-call messages are not directly accepted by Qwen-style
    Jinja chat templates, but trace-store writes and reads both need a stable
    prompt string key. This helper hardcodes the copied messages into a
    template-renderable shape before ``apply_chat_template``. It must not
    mutate model responses returned to the agent/client, nor the generated
    token ids, labels, or raw rollout artifacts used for training.
    """

    messages = copy.deepcopy(messages)
    for message in messages:
        for key in _PROCESS_ONLY_MESSAGE_KEYS:
            message.pop(key, None)
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict) or "arguments" not in function:
                continue
            function["arguments"] = canonicalize_tool_arguments(function["arguments"])
    return messages


def canonicalize_tool_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str):
        return {_RAW_ARGUMENTS_KEY: str(arguments)}

    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        parsed = _loads_partial_json_object(arguments)
        if isinstance(parsed, dict):
            parsed[_RAW_ARGUMENTS_KEY] = arguments
            return parsed
        return {_RAW_ARGUMENTS_KEY: arguments}

    if isinstance(parsed, dict):
        return parsed
    return {_RAW_ARGUMENTS_KEY: arguments}


def _loads_partial_json_object(raw: str) -> dict[str, Any] | None:
    stripped = raw.strip()
    if not stripped.startswith("{"):
        return None

    repaired = _loads_repaired_json_object(stripped)
    if isinstance(repaired, dict):
        return repaired

    return _loads_json_object_prefix(stripped)


def _loads_repaired_json_object(raw: str) -> dict[str, Any] | None:
    repaired = _repair_json_suffix(raw)
    if repaired is None:
        return None
    try:
        parsed = json.loads(repaired)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _repair_json_suffix(raw: str) -> str | None:
    stack: list[str] = []
    in_string = False
    escaped = False

    for char in raw:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in ("}", "]"):
            if not stack or stack[-1] != char:
                return None
            stack.pop()

    repaired = raw
    if escaped:
        repaired = repaired[:-1]
    if in_string:
        repaired += '"'
    repaired += "".join(reversed(stack))
    return repaired


def _loads_json_object_prefix(raw: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    values: dict[str, Any] = {}
    index = 1
    length = len(raw)

    while index < length:
        index = _skip_ws(raw, index)
        if index >= length or raw[index] == "}":
            break
        if raw[index] != '"':
            break

        try:
            key, index = decoder.raw_decode(raw, index)
        except json.JSONDecodeError:
            break
        if not isinstance(key, str):
            break

        index = _skip_ws(raw, index)
        if index >= length or raw[index] != ":":
            break
        index = _skip_ws(raw, index + 1)
        if index >= length:
            break

        try:
            value, next_index = decoder.raw_decode(raw, index)
        except json.JSONDecodeError:
            value, next_index = _parse_partial_value(raw, index)
            if next_index == index:
                break

        values[key] = value
        index = _skip_ws(raw, next_index)
        if index < length and raw[index] == ",":
            index += 1
            continue
        break

    return values or None


def _parse_partial_value(raw: str, index: int) -> tuple[Any, int]:
    if raw[index] == '"':
        repaired = _repair_json_suffix(raw[index:])
        if repaired is not None:
            try:
                return json.loads(repaired), len(raw)
            except json.JSONDecodeError:
                pass
        return raw[index + 1 :], len(raw)

    end = _find_value_boundary(raw, index)
    raw_value = raw[index:end].strip()
    if not raw_value:
        return None, index

    repaired = _repair_json_suffix(raw_value)
    if repaired is not None:
        try:
            return json.loads(repaired), end
        except json.JSONDecodeError:
            pass
    return raw_value, end


def _find_value_boundary(raw: str, index: int) -> int:
    stack: list[str] = []
    in_string = False
    escaped = False

    for pos in range(index, len(raw)):
        char = raw[pos]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in ("}", "]"):
            if stack and stack[-1] == char:
                stack.pop()
            elif not stack:
                return pos
        elif char == "," and not stack:
            return pos

    return len(raw)


def _skip_ws(raw: str, index: int) -> int:
    while index < len(raw) and raw[index].isspace():
        index += 1
    return index
