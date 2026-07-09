from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_SPAN_NAME_PATH_ATTRIBUTE = "xtuner.span_name_path"
_LEGACY_LOGICAL_PATH_ATTRIBUTE = "xtuner.logical_path"

_INITIAL_SAMPLE_STATUSES = {"init"}
_NON_TERMINAL_SAMPLE_STATUSES = {"", "init", "pending", "queued", "running", "scheduled", "started", "unknown"}
_ERROR_SAMPLE_STATUSES = {"aborted", "error", "exception", "failed", "timeout", "timed_out"}
_TERMINAL_STAGE_STATUSES = {
    "completed",
    "failed",
    "aborted",
    "timeout",
    "timed_out",
    "expired",
    "stale",
    "filtered",
}


def build_rollout_view_payload_from_jaeger_traces(
    traces: Iterable[dict[str, Any]],
    *,
    jaeger_query_url: str | None = None,
    jaeger_link_url: str | None = None,
    live_records: Iterable[dict[str, Any]] | None = None,
    service_name: str | None = None,
    run_id: str | None = None,
    train_step: Any = "latest",
) -> dict[str, Any]:
    samples_by_key: dict[tuple[str, Any], dict[str, Any]] = {}
    jaeger_trace_link_base_url = _jaeger_trace_link_base_url(jaeger_query_url, jaeger_link_url)

    for trace_data in traces:
        trace_id = str(trace_data.get("traceID") or trace_data.get("trace_id") or "")
        if not trace_id:
            continue
        process_metadata = _process_metadata(trace_data)
        span_entries = []
        entries_by_span_id: dict[str, dict[str, Any]] = {}
        for span in trace_data.get("spans") or []:
            process = process_metadata.get(str(span.get("processID") or ""), {})
            span_service_name = process.get("service_name")
            if service_name is not None and span_service_name != service_name:
                continue
            tags = _tags_to_dict(span.get("tags") or [])
            span_run_id = tags.get("run.id") or process.get("run_id")
            if run_id is not None and span_run_id != run_id:
                continue
            entry = {
                "span": span,
                "tags": tags,
                "service_name": span_service_name,
                "run_id": span_run_id,
                "span_id": _span_id(span),
                "trace_id": trace_id,
            }
            span_entries.append(entry)
            if entry["span_id"]:
                entries_by_span_id[entry["span_id"]] = entry

        for entry in span_entries:
            span = entry["span"]
            tags = entry["tags"]
            rollout_id, sample_tags = _resolve_rollout_sample(entry, entries_by_span_id)
            if rollout_id is None:
                continue
            span_service_name = entry["service_name"]
            span_run_id = entry["run_id"]
            sample_key = (trace_id, rollout_id)
            sample = samples_by_key.setdefault(
                sample_key,
                {
                    "trace_id": trace_id,
                    "rollout_id": rollout_id,
                    "group_id": sample_tags.get("xtuner.group_id"),
                    "producer_future_step": _producer_future_step(sample_tags),
                    "task_name": sample_tags.get("xtuner.task_name"),
                    "status": sample_tags.get("xtuner.status"),
                    "service_name": span_service_name,
                    "run_id": span_run_id,
                    "jaeger_url": _jaeger_trace_url(jaeger_trace_link_base_url, trace_id),
                    "spans": [],
                },
            )
            _merge_sample_fields(sample, sample_tags)
            if span_run_id is not None:
                sample["run_id"] = span_run_id
            sample["spans"].append(_span_payload(span, tags, service_name=span_service_name, run_id=span_run_id))

    _merge_live_records(samples_by_key, live_records or (), jaeger_trace_link_base_url)

    generated_at_s = time.time()
    samples = []
    for sample in samples_by_key.values():
        sample["spans"].sort(key=lambda item: (item["start_time_us"], item["span_id"]))
        sample["span_count"] = len(sample["spans"])
        _apply_live_state(sample, generated_at_s)
        _apply_sample_display_status(sample)
        _apply_sample_reward_filter(sample)
        sample["stage"] = _sample_stage(sample)
        samples.append(sample)

    samples.sort(key=lambda item: (str(item.get("group_id")), str(item.get("rollout_id")), item["trace_id"]))
    base_payload = {
        "title": "XTuner Rollout Trace Viewer",
        "generated_at_s": generated_at_s,
        "source": "jaeger",
        "jaeger_query_url": _normalize_jaeger_query_url(jaeger_query_url),
        "jaeger_link_url": jaeger_trace_link_base_url,
        "service_name": service_name,
        "run_id": run_id,
        "available_train_steps": _available_train_steps(samples),
        "samples": samples,
    }
    return filter_rollout_view_payload_by_train_step(base_payload, train_step)


def load_jaeger_traces_from_otel_jsonl(trace_jsonl_path: Path | str) -> list[dict[str, Any]]:
    traces_by_id: dict[str, dict[str, Any]] = {}
    process_ids: dict[tuple[str, str, tuple[tuple[str, str], ...]], str] = {}
    path = Path(trace_jsonl_path).expanduser()
    if not path.is_file():
        return []

    for record in _iter_jsonl_records(path):
        if not isinstance(record, dict):
            continue
        jaeger_traces = _jaeger_traces_from_json_record(record)
        if jaeger_traces is not None:
            for trace_data in jaeger_traces:
                _merge_jaeger_trace(traces_by_id, trace_data)
            continue
        for resource_span in record.get("resourceSpans") or []:
            if not isinstance(resource_span, dict):
                continue
            resource_attrs = _otel_attributes_to_dict(
                (resource_span.get("resource") or {}).get("attributes") or []
            )
            service_name = str(resource_attrs.get("service.name") or "unknown")
            process_tags = _dict_to_jaeger_tags(resource_attrs)
            for scope_span in _otel_scope_spans(resource_span):
                for otel_span in scope_span.get("spans") or []:
                    if not isinstance(otel_span, dict):
                        continue
                    trace_id = str(
                        otel_span.get("traceId")
                        or otel_span.get("traceID")
                        or otel_span.get("trace_id")
                        or ""
                    )
                    span_id = str(
                        otel_span.get("spanId")
                        or otel_span.get("spanID")
                        or otel_span.get("span_id")
                        or ""
                    )
                    if not trace_id or not span_id:
                        continue
                    trace_data = traces_by_id.setdefault(
                        trace_id,
                        {
                            "traceID": trace_id,
                            "processes": {},
                            "spans": [],
                        },
                    )
                    process_key = (
                        trace_id,
                        service_name,
                        tuple(sorted((str(key), str(value)) for key, value in resource_attrs.items())),
                    )
                    process_id = process_ids.get(process_key)
                    if process_id is None:
                        process_id = f"p{len(trace_data['processes']) + 1}"
                        process_ids[process_key] = process_id
                    trace_data["processes"][process_id] = {
                        "serviceName": service_name,
                        "tags": process_tags,
                    }
                    trace_data["spans"].append(
                        _otel_span_to_jaeger_span(otel_span, trace_id, span_id, process_id)
                    )
    return list(traces_by_id.values())


def load_live_trace_records(live_jsonl_path: Path | str | None) -> list[dict[str, Any]]:
    if live_jsonl_path is None:
        return []
    path = Path(live_jsonl_path).expanduser()
    if not path.is_file():
        return []
    return list(_iter_jsonl_records(path))


def _iter_jsonl_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


def _jaeger_traces_from_json_record(record: dict[str, Any]) -> list[dict[str, Any]] | None:
    data = record.get("data")
    if isinstance(data, list):
        return [trace_data for trace_data in data if isinstance(trace_data, dict)]
    if record.get("traceID") is not None and isinstance(record.get("spans"), list):
        return [record]
    return None


def _merge_jaeger_trace(traces_by_id: dict[str, dict[str, Any]], trace_data: dict[str, Any]) -> None:
    trace_id = str(trace_data.get("traceID") or trace_data.get("trace_id") or "")
    if not trace_id:
        return
    target = traces_by_id.setdefault(trace_id, {"traceID": trace_id, "processes": {}, "spans": []})
    if isinstance(trace_data.get("processes"), dict):
        target["processes"].update(trace_data["processes"])
    target["spans"].extend([span for span in trace_data.get("spans") or [] if isinstance(span, dict)])


def _otel_scope_spans(resource_span: dict[str, Any]) -> list[dict[str, Any]]:
    scope_spans = resource_span.get("scopeSpans")
    if isinstance(scope_spans, list):
        return [scope_span for scope_span in scope_spans if isinstance(scope_span, dict)]
    legacy_scope_spans = resource_span.get("instrumentationLibrarySpans")
    if isinstance(legacy_scope_spans, list):
        return [scope_span for scope_span in legacy_scope_spans if isinstance(scope_span, dict)]
    return []


def _otel_span_to_jaeger_span(
    otel_span: dict[str, Any],
    trace_id: str,
    span_id: str,
    process_id: str,
) -> dict[str, Any]:
    attributes = _otel_attributes_to_dict(otel_span.get("attributes") or [])
    tags = _dict_to_jaeger_tags(attributes)
    tags.extend(_otel_status_tags(otel_span.get("status") or {}))
    start_ns = _int_from_otel_time(
        otel_span.get("startTimeUnixNano")
        or otel_span.get("start_time_unix_nano")
        or otel_span.get("startTime")
        or 0
    )
    end_ns = _int_from_otel_time(
        otel_span.get("endTimeUnixNano")
        or otel_span.get("end_time_unix_nano")
        or otel_span.get("endTime")
        or start_ns
    )
    parent_span_id = otel_span.get("parentSpanId") or otel_span.get("parent_span_id")
    references = []
    if parent_span_id is not None and str(parent_span_id):
        references.append({"refType": "CHILD_OF", "traceID": trace_id, "spanID": str(parent_span_id)})
    return {
        "traceID": trace_id,
        "spanID": span_id,
        "operationName": str(otel_span.get("name") or otel_span.get("operationName") or "unknown"),
        "processID": process_id,
        "startTime": start_ns // 1_000,
        "duration": max(0, end_ns - start_ns) // 1_000,
        "references": references,
        "tags": tags,
    }


def _otel_status_tags(status: dict[str, Any]) -> list[dict[str, Any]]:
    code = str(status.get("code") or status.get("statusCode") or "STATUS_CODE_UNSET")
    if code in {"STATUS_CODE_ERROR", "ERROR", "2"}:
        normalized = "ERROR"
    elif code in {"STATUS_CODE_OK", "OK", "1"}:
        normalized = "OK"
    else:
        normalized = "UNSET"
    tags = [{"key": "otel.status_code", "type": "string", "value": normalized}]
    message = status.get("message") or status.get("description")
    if message:
        tags.append({"key": "otel.status_description", "type": "string", "value": str(message)})
        if normalized == "ERROR":
            tags.append({"key": "error.message", "type": "string", "value": str(message)})
    return tags


def _otel_attributes_to_dict(attributes: Any) -> dict[str, Any]:
    if isinstance(attributes, dict):
        return dict(attributes)
    result: dict[str, Any] = {}
    for attribute in attributes or []:
        if not isinstance(attribute, dict):
            continue
        key = attribute.get("key")
        if key is None:
            continue
        result[str(key)] = _otel_any_value_to_python(attribute.get("value"))
    return result


def _otel_any_value_to_python(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    if "stringValue" in value:
        return value["stringValue"]
    if "boolValue" in value:
        return bool(value["boolValue"])
    if "intValue" in value:
        return _int_or_original(value["intValue"])
    if "doubleValue" in value:
        return float(value["doubleValue"])
    if "bytesValue" in value:
        return value["bytesValue"]
    if "arrayValue" in value:
        return [_otel_any_value_to_python(item) for item in (value["arrayValue"].get("values") or [])]
    if "kvlistValue" in value:
        return {
            str(item.get("key")): _otel_any_value_to_python(item.get("value"))
            for item in (value["kvlistValue"].get("values") or [])
            if isinstance(item, dict) and item.get("key") is not None
        }
    return value


def _dict_to_jaeger_tags(attributes: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"key": str(key), "type": _jaeger_tag_type(value), "value": value}
        for key, value in attributes.items()
        if value is not None
    ]


def _jaeger_tag_type(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int64"
    if isinstance(value, float):
        return "float64"
    return "string"


def _int_from_otel_time(value: Any) -> int:
    parsed = _int_or_original(value)
    return parsed if isinstance(parsed, int) and not isinstance(parsed, bool) else 0


def _int_or_original(value: Any) -> int | Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return value


def _merge_sample_fields(sample: dict[str, Any], tags: dict[str, Any]) -> None:
    for sample_key, tag_key in (
        ("rollout_id", "xtuner.rollout_id"),
        ("group_id", "xtuner.group_id"),
        ("task_name", "xtuner.task_name"),
    ):
        if tags.get(tag_key) is not None:
            sample[sample_key] = tags[tag_key]
    _merge_sample_status(sample, tags.get("xtuner.status"))
    producer_future_step = _producer_future_step(tags)
    if producer_future_step is not None:
        sample["producer_future_step"] = producer_future_step


def _merge_sample_status(sample: dict[str, Any], status: Any) -> None:
    if status is None:
        return
    current_status = sample.get("status")
    if current_status is None or _sample_status_priority(status) >= _sample_status_priority(current_status):
        sample["status"] = status


def _apply_sample_display_status(sample: dict[str, Any]) -> None:
    status = str(sample.get("status") or "").strip().lower()
    if status not in _INITIAL_SAMPLE_STATUSES:
        return
    if _sample_has_observed_stage(sample):
        sample["status"] = "running"


def _sample_has_observed_stage(sample: dict[str, Any]) -> bool:
    current_stage = sample.get("current_stage")
    if isinstance(current_stage, dict) and current_stage.get("name"):
        return True
    return bool(sample.get("spans"))


def _sample_status_priority(status: Any) -> int:
    normalized = str(status or "").strip().lower()
    if normalized in _ERROR_SAMPLE_STATUSES:
        return 3
    if normalized in _TERMINAL_STAGE_STATUSES:
        return 2
    if normalized in _NON_TERMINAL_SAMPLE_STATUSES:
        return 0
    return 1


def _merge_live_records(
    samples_by_key: dict[tuple[str, Any], dict[str, Any]],
    live_records: Iterable[dict[str, Any]],
    jaeger_trace_link_base_url: str | None,
) -> None:
    for record in live_records:
        if not isinstance(record, dict):
            continue
        trace_id = str(record.get("trace_id") or "")
        attributes = record.get("attributes")
        if not trace_id or not isinstance(attributes, dict):
            continue
        rollout_id = attributes.get("xtuner.rollout_id")
        if rollout_id is None:
            continue
        sample_key = (trace_id, rollout_id)
        sample = samples_by_key.setdefault(
            sample_key,
            {
                "trace_id": trace_id,
                "rollout_id": rollout_id,
                "group_id": attributes.get("xtuner.group_id"),
                "producer_future_step": _producer_future_step(attributes),
                "task_name": attributes.get("xtuner.task_name"),
                "status": attributes.get("xtuner.status"),
                "service_name": None,
                "run_id": None,
                "jaeger_url": _jaeger_trace_url(jaeger_trace_link_base_url, trace_id),
                "spans": [],
            },
        )
        _merge_sample_fields(sample, attributes)
        sample.setdefault("_live_records", []).append(record)


def _apply_live_state(sample: dict[str, Any], generated_at_s: float) -> None:
    live_states = _live_span_states(sample.pop("_live_records", []))
    active_states = [state for state in live_states if state.get("status") == "running"]
    active_states.sort(key=lambda state: (len(state.get("span_name_path") or []), float(state.get("started_at_s") or 0.0)))
    current_state = active_states[-1] if active_states else None
    if current_state is not None:
        started_at_s = float(current_state.get("started_at_s") or generated_at_s)
        span_name = str(current_state.get("span_name") or "")
        sample["current_stage"] = {
            "name": span_name,
            "stage": str(current_state.get("stage") or span_name or "unknown"),
            "status": "running",
            "elapsed_ms": round(max(0.0, generated_at_s - started_at_s) * 1000.0, 3),
            "started_at_s": started_at_s,
        }
    else:
        sample["current_stage"] = None
    sample["live_spans"] = live_states
    sample["display_path"] = _build_display_path(sample, live_states, current_state, generated_at_s)
    sample["chain"] = " -> ".join(node["name"] for node in sample["display_path"])


def _live_span_states(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        span_name = str(record.get("span_name") or "")
        attributes = record.get("attributes") if isinstance(record.get("attributes"), dict) else {}
        span_id = str(record.get("span_id") or "")
        if not span_id:
            span_id = f"live:{span_name}:{index}"
        state = states.setdefault(
            span_id,
            {
                "span_id": span_id,
                "span_name": span_name,
                "stage": _stage_from_span_name_and_attributes(span_name, attributes),
                "span_name_path": _span_name_path_from_value(
                    record.get("span_name_path") or record.get("logical_path")
                ),
                "attributes": attributes,
                "status": "running",
            },
        )
        event = str(record.get("event") or "")
        if event == "start":
            state["started_at_s"] = _float_or_none(record.get("time_s"))
            state["status"] = "running"
        elif event == "end":
            state["ended_at_s"] = _float_or_none(record.get("time_s"))
            state["duration_ms"] = _float_or_none(record.get("duration_ms"))
            state["status"] = str(record.get("status") or "completed")
            if record.get("error_message"):
                state["error_message"] = record["error_message"]
        if record.get("trace_id") is not None:
            state["trace_id"] = record["trace_id"]
    return sorted(states.values(), key=lambda state: (float(state.get("started_at_s") or 0.0), str(state.get("span_id"))))


def _build_display_path(
    sample: dict[str, Any],
    live_states: list[dict[str, Any]],
    current_state: dict[str, Any] | None,
    generated_at_s: float,
) -> list[dict[str, Any]]:
    spans = sample.get("spans") or []
    spans_by_name = {str(span.get("name") or ""): span for span in spans}
    live_by_name = {str(state.get("span_name") or ""): state for state in live_states}
    path = _display_path_names(spans, live_states, current_state)
    nodes = []
    for name in path:
        span = spans_by_name.get(name)
        live_state = live_by_name.get(name)
        if current_state is not None and name == current_state.get("span_name"):
            started_at_s = float(current_state.get("started_at_s") or generated_at_s)
            nodes.append(
                {
                    "name": name,
                    "stage": current_state.get("stage") or name,
                    "source": "live",
                    "status": "running",
                    "elapsed_ms": round(max(0.0, generated_at_s - started_at_s) * 1000.0, 3),
                }
            )
        elif span is not None:
            nodes.append(
                {
                    "name": name,
                    "stage": _span_semantic_stage(span),
                    "source": "span",
                    "status": "done" if str(span.get("status") or "").upper() != "ERROR" else "error",
                    "duration_ms": span.get("duration_ms"),
                }
            )
        elif live_state is not None and live_state.get("status") == "running":
            nodes.append({"name": name, "stage": live_state.get("stage") or name, "source": "live", "status": "active"})
        else:
            nodes.append({"name": name, "source": "logical", "status": "inferred"})
    return nodes


def _display_path_names(
    spans: list[dict[str, Any]],
    live_states: list[dict[str, Any]],
    current_state: dict[str, Any] | None,
) -> list[str]:
    path = _span_name_path_from_value(current_state.get("span_name_path")) if current_state is not None else []
    if not path:
        path = [str(span.get("name") or "") for span in spans if span.get("name")]
    if not path:
        path = [str(state.get("span_name") or "") for state in live_states if state.get("span_name")]
    if not path:
        span_paths = [
            _span_name_path_from_span_attributes(span.get("attributes") or {})
            for span in spans
        ]
        path = max(span_paths, key=len, default=[])
    if not path:
        live_paths = [_span_name_path_from_value(state.get("span_name_path")) for state in live_states]
        path = max(live_paths, key=len, default=[])
    return _unique_path(path)


def _span_name_path_from_span_attributes(attributes: dict[str, Any]) -> list[str]:
    return _span_name_path_from_value(
        attributes.get(_SPAN_NAME_PATH_ATTRIBUTE) or attributes.get(_LEGACY_LOGICAL_PATH_ATTRIBUTE)
    )


def _span_name_path_from_value(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            decoded = [part.strip() for part in value.split("->")]
        value = decoded
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if isinstance(item, str) and item.strip()]
    return []


def _unique_path(path: list[str]) -> list[str]:
    result = []
    for name in path:
        if not result or result[-1] != name:
            result.append(name)
    return result


def _producer_future_step(tags: dict[str, Any]) -> Any:
    value = tags.get("xtuner.producer_future_step")
    return value if value is not None else tags.get("xtuner.train_step")


def _build_step_group_summaries(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    steps: dict[str, dict[str, Any]] = {}
    for sample in samples:
        producer_future_step = sample.get("producer_future_step")
        group_id = sample.get("group_id")
        step_key = _summary_key(producer_future_step)
        group_key = _summary_key(group_id)

        step_summary = steps.setdefault(
            step_key,
            {
                "producer_future_step": producer_future_step,
                "sample_count": 0,
                "groups": {},
            },
        )
        step_summary["sample_count"] += 1
        groups = step_summary["groups"]
        group_summary = groups.setdefault(
            group_key,
            {
                "group_id": group_id,
                "sample_count": 0,
                "statuses": Counter(),
                "stages": Counter(),
                "rollout_ids": [],
            },
        )
        group_summary["sample_count"] += 1
        group_summary["statuses"][str(sample.get("status") or "unknown")] += 1
        if _should_show_group_stage(sample):
            group_summary["stages"][str(sample.get("stage") or "unknown")] += 1
        group_summary["rollout_ids"].append(sample.get("rollout_id"))

    summaries = []
    for step_summary in steps.values():
        groups = []
        for group_summary in step_summary["groups"].values():
            group_summary["statuses"] = dict(sorted(group_summary["statuses"].items()))
            group_summary["stages"] = dict(sorted(group_summary["stages"].items()))
            group_summary["rollout_ids"].sort(key=lambda value: str(value))
            groups.append(group_summary)
        groups.sort(key=lambda item: _sortable_summary_value(item["group_id"]))
        summaries.append(
            {
                "producer_future_step": step_summary["producer_future_step"],
                "group_count": sum(1 for group in groups if group["group_id"] is not None),
                "sample_count": step_summary["sample_count"],
                "groups": groups,
            }
        )
    summaries.sort(key=lambda item: _sortable_summary_value(item["producer_future_step"]))
    return summaries


def _available_train_steps(samples: list[dict[str, Any]]) -> list[Any]:
    values = {sample.get("producer_future_step") for sample in samples if sample.get("producer_future_step") is not None}
    return sorted(values, key=_sortable_summary_value)


def _select_train_step(requested: Any, available_steps: list[Any]) -> Any:
    if not available_steps:
        return "all"
    if requested is None or requested == "":
        requested = "latest"
    requested_text = str(requested).strip()
    if requested_text.lower() == "all":
        return "all"
    if requested_text.lower() == "latest":
        return max(available_steps, key=_sortable_summary_value)
    for step in available_steps:
        if str(step) == requested_text:
            return step
    return max(available_steps, key=_sortable_summary_value)


def _filter_samples_by_train_step(samples: list[dict[str, Any]], selected_train_step: Any) -> list[dict[str, Any]]:
    if selected_train_step == "all":
        return samples
    return [sample for sample in samples if str(sample.get("producer_future_step")) == str(selected_train_step)]


def filter_rollout_view_payload_by_train_step(payload: dict[str, Any], train_step: Any = "latest") -> dict[str, Any]:
    samples = list(payload.get("samples") or [])
    generated_at_s = float(payload.get("generated_at_s") or time.time())
    available_train_steps = list(payload.get("available_train_steps") or _available_train_steps(samples))
    selected_train_step = _select_train_step(train_step, available_train_steps)
    visible_samples = _filter_samples_by_train_step(samples, selected_train_step)

    status_counts: Counter[str] = Counter()
    stage_counts: Counter[str] = Counter()
    group_ids: set[Any] = set()
    for sample in visible_samples:
        status = str(sample.get("status") or "unknown")
        status_counts[status] += 1
        stage_counts[str(sample["stage"])] += 1
        if sample.get("group_id") is not None:
            group_ids.add(sample["group_id"])

    step_group_summaries = _build_step_group_summaries(visible_samples)
    filtered_payload = dict(payload)
    filtered_payload.update(
        {
            "generated_at_s": generated_at_s,
            "selected_train_step": selected_train_step,
            "available_train_steps": available_train_steps,
            "total_sample_count": len(samples),
            "sample_count": len(visible_samples),
            "group_count": len(group_ids),
            "step_count": len(step_group_summaries),
            "step_group_summaries": step_group_summaries,
            "stage_occupancy": _build_stage_occupancy(visible_samples, generated_at_s),
            "stage_duration_summaries": _build_stage_duration_summaries(visible_samples),
            "status_counts": dict(sorted(status_counts.items())),
            "stage_counts": dict(sorted(stage_counts.items())),
            "samples": visible_samples,
        }
    )
    return filtered_payload


def _build_stage_duration_summaries(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    durations_by_stage: dict[str, list[float]] = defaultdict(list)
    raw_durations_by_stage: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    errors_by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        for span in sample.get("spans") or []:
            try:
                duration_s = float(span.get("duration_ms") or 0.0) / 1000.0
            except (TypeError, ValueError):
                continue
            stage = _span_semantic_stage(span)
            raw_span_name = _span_raw_name(span)
            durations_by_stage[stage].append(duration_s)
            raw_durations_by_stage[stage][raw_span_name].append(duration_s)
        error = _extract_sample_error(sample)
        if error is not None:
            errors_by_stage[str(error.get("stage") or sample.get("stage") or "unknown")].append(
                {
                    **error,
                    "rollout_id": sample.get("rollout_id"),
                    "group_id": sample.get("group_id"),
                    "producer_future_step": sample.get("producer_future_step"),
                    "jaeger_url": sample.get("jaeger_url"),
                }
            )

    summaries = []
    for stage, durations in durations_by_stage.items():
        durations.sort()
        top_errors = _summarize_stage_errors(errors_by_stage.get(stage, []))
        summaries.append(
            {
                "stage": stage,
                "span_count": len(durations),
                "avg_duration_s": _round_duration(sum(durations) / len(durations)),
                "p50_duration_s": _round_duration(_nearest_rank_percentile(durations, 0.50)),
                "p95_duration_s": _round_duration(_nearest_rank_percentile(durations, 0.95)),
                "max_duration_s": _round_duration(durations[-1]),
                "error_count": sum(error["sample_count"] for error in top_errors),
                "top_errors": top_errors,
                "raw_spans": _summarize_raw_span_durations(raw_durations_by_stage.get(stage, {})),
            }
        )
    for stage, errors in errors_by_stage.items():
        if stage in durations_by_stage:
            continue
        top_errors = _summarize_stage_errors(errors)
        summaries.append(
            {
                "stage": stage,
                "span_count": 0,
                "avg_duration_s": 0.0,
                "p50_duration_s": 0.0,
                "p95_duration_s": 0.0,
                "max_duration_s": 0.0,
                "error_count": sum(error["sample_count"] for error in top_errors),
                "top_errors": top_errors,
                "raw_spans": [],
            }
        )
    return summaries


def _summarize_raw_span_durations(raw_durations: dict[str, list[float]]) -> list[dict[str, Any]]:
    rows = []
    for span_name, durations in raw_durations.items():
        if not durations:
            continue
        durations = sorted(durations)
        rows.append(
            {
                "span": span_name,
                "span_count": len(durations),
                "avg_duration_s": _round_duration(sum(durations) / len(durations)),
                "max_duration_s": _round_duration(durations[-1]),
            }
        )
    rows.sort(key=lambda item: (-item["span_count"], str(item["span"])))
    return rows


def _summarize_stage_errors(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, Any], dict[str, Any]] = {}
    for error in errors:
        key = (
            str(error.get("error_type") or "error"),
            str(error.get("message") or ""),
            error.get("http_status_code"),
        )
        summary = grouped.setdefault(
            key,
            {
                "error_type": key[0],
                "message": key[1],
                "http_status_code": key[2],
                "sample_count": 0,
                "rollout_ids": [],
                "groups": [],
                "steps": [],
                "jaeger_urls": [],
            },
        )
        summary["sample_count"] += 1
        _append_unique(summary["rollout_ids"], error.get("rollout_id"))
        _append_unique(summary["groups"], error.get("group_id"))
        _append_unique(summary["steps"], error.get("producer_future_step"))
        _append_unique(summary["jaeger_urls"], error.get("jaeger_url"))
    result = list(grouped.values())
    for summary in result:
        summary["rollout_ids"].sort(key=lambda value: str(value))
        summary["groups"].sort(key=lambda value: str(value))
        summary["steps"].sort(key=lambda value: str(value))
    result.sort(key=lambda item: (-item["sample_count"], str(item["error_type"]), str(item["message"])))
    return result


def _build_stage_occupancy(samples: list[dict[str, Any]], generated_at_s: float) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for sample in samples:
        stage, raw_span_name = _stage_bucket_for_sample(sample)
        latest_start_s = _latest_span_start_s(sample)
        age_s = 0.0 if stage in _TERMINAL_STAGE_STATUSES else max(0.0, generated_at_s - latest_start_s)
        bucket = buckets.setdefault(
            stage,
            {
                "stage": stage,
                "sample_count": 0,
                "_group_ids": set(),
                "_raw_spans": {},
                "oldest_age_s": 0.0,
                "oldest_rollout_id": None,
                "oldest_group_id": None,
                "oldest_producer_future_step": None,
            },
        )
        bucket["sample_count"] += 1
        if sample.get("group_id") is not None:
            bucket["_group_ids"].add(sample["group_id"])
        if raw_span_name:
            raw_spans = bucket["_raw_spans"]
            raw_bucket = raw_spans.setdefault(raw_span_name, {"span": raw_span_name, "sample_count": 0, "_group_ids": set()})
            raw_bucket["sample_count"] += 1
            if sample.get("group_id") is not None:
                raw_bucket["_group_ids"].add(sample["group_id"])
        if age_s >= bucket["oldest_age_s"]:
            bucket["oldest_age_s"] = _round_duration(age_s)
            bucket["oldest_rollout_id"] = sample.get("rollout_id")
            bucket["oldest_group_id"] = sample.get("group_id")
            bucket["oldest_producer_future_step"] = sample.get("producer_future_step")
    rows = list(buckets.values())
    for row in rows:
        row["group_count"] = len(row.pop("_group_ids"))
        row["raw_spans"] = _summarize_raw_span_occupancy(row.pop("_raw_spans"))
    return rows


def _summarize_raw_span_occupancy(raw_spans: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for raw_bucket in raw_spans.values():
        rows.append(
            {
                "span": raw_bucket["span"],
                "sample_count": raw_bucket["sample_count"],
                "group_count": len(raw_bucket["_group_ids"]),
            }
        )
    rows.sort(key=lambda item: (-item["sample_count"], str(item["span"])))
    return rows


def _stage_bucket_for_sample(sample: dict[str, Any]) -> tuple[str, str | None]:
    status = str(sample.get("status") or "").strip().lower()
    if status in _TERMINAL_STAGE_STATUSES:
        return status, None
    current_stage = sample.get("current_stage")
    if isinstance(current_stage, dict):
        raw_stage_name = str(current_stage.get("name") or "").strip()
        semantic_stage = str(current_stage.get("stage") or "").strip()
        if semantic_stage:
            return semantic_stage, raw_stage_name or None
        if raw_stage_name:
            return raw_stage_name, raw_stage_name
    spans = sample.get("spans") or []
    if not spans:
        return status or "unknown", None
    latest_span = spans[-1]
    if latest_span.get("rollout_backend"):
        return str(latest_span.get("stage") or "llm_generate"), _span_raw_name(latest_span)
    if status in {"pending", "queued", "scheduled"}:
        return "scheduled", None
    return _span_semantic_stage(latest_span), _span_raw_name(latest_span)


def _latest_span_start_s(sample: dict[str, Any]) -> float:
    current_stage = sample.get("current_stage")
    if isinstance(current_stage, dict) and current_stage.get("started_at_s") is not None:
        try:
            return float(current_stage["started_at_s"])
        except (TypeError, ValueError):
            pass
    spans = sample.get("spans") or []
    if not spans:
        return time.time()
    return max(float(span.get("start_time_us") or 0) / 1_000_000.0 for span in spans)


def _apply_sample_reward_filter(sample: dict[str, Any]) -> None:
    values: dict[str, Any] = {}
    for span in sample.get("spans") or []:
        attrs = span.get("attributes") or {}
        for target, keys in (
            ("reward_score", ("reward.score", "reward_score")),
            ("reward_pass", ("reward.pass",)),
            ("filter_decision", ("filter.decision",)),
            ("filter_reason", ("filter.reason",)),
            ("train_included", ("train.included",)),
            ("oversample_source", ("oversample.source",)),
            ("drop_reason", ("drop.reason",)),
        ):
            for key in keys:
                if key in attrs:
                    values[target] = attrs[key]
    if "reward_score" in values:
        values["reward_score"] = _to_float(values["reward_score"])
    if "reward_pass" in values:
        values["reward_pass"] = _to_bool(values["reward_pass"])
    if "train_included" in values:
        values["train_included"] = _to_bool(values["train_included"])
    sample.update(values)


def _extract_sample_error(sample: dict[str, Any]) -> dict[str, Any] | None:
    fallback_span_name = str(sample.get("stage") or "unknown")
    for span in sample.get("spans") or []:
        attrs = span.get("attributes") or {}
        http_status = attrs.get("http.status_code")
        is_http_error = False
        try:
            is_http_error = http_status is not None and int(http_status) >= 400
        except (TypeError, ValueError):
            is_http_error = False
        has_error = (
            str(span.get("status") or "").upper() == "ERROR"
            or _to_bool(attrs.get("error")) is True
            or any(key.startswith("error.") for key in attrs)
            or any(key.startswith("exception.") for key in attrs)
            or is_http_error
        )
        if not has_error:
            continue
        return {
            "error_type": attrs.get("exception.type")
            or attrs.get("error.type")
            or attrs.get("xtuner.error_type")
            or ("HTTPError" if is_http_error else str(sample.get("status") or "error")),
            "message": attrs.get("xtuner.error_msg")
            or attrs.get("error.message")
            or attrs.get("exception.message")
            or (f"http_status={http_status}" if is_http_error else ""),
            "http_status_code": http_status,
            "span_name": span.get("name") or fallback_span_name,
            "stage": _span_semantic_stage(span),
        }
    status = str(sample.get("status") or "").lower()
    if status in _ERROR_SAMPLE_STATUSES:
        return {
            "error_type": status,
            "message": "",
            "http_status_code": None,
            "span_name": fallback_span_name,
            "stage": fallback_span_name,
        }
    return None


def _nearest_rank_percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    index = max(0, min(len(sorted_values) - 1, math.ceil(percentile * len(sorted_values)) - 1))
    return sorted_values[index]


def _round_duration(value: float) -> float:
    return round(value, 3)


def _span_raw_name(span: dict[str, Any]) -> str:
    return str(span.get("name") or "unknown")


def _span_semantic_stage(span: dict[str, Any]) -> str:
    name = _span_raw_name(span)
    attributes = span.get("attributes")
    return _stage_from_span_name_and_attributes(name, attributes if isinstance(attributes, dict) else None)


def _stage_from_span_name_and_attributes(span_name: str, attributes: dict[str, Any] | None = None) -> str:
    attrs = attributes or {}
    for key in ("xtuner.stage", "stage", "stage.name"):
        stage = str(attrs.get(key) or "").strip()
        if stage:
            return stage
    return span_name or "unknown"


def _sample_stage(sample: dict[str, Any]) -> str:
    status = str(sample.get("status") or "").strip().lower()
    if status and status not in _NON_TERMINAL_SAMPLE_STATUSES:
        return status
    current_stage = sample.get("current_stage")
    if isinstance(current_stage, dict):
        semantic_stage = str(current_stage.get("stage") or "").strip()
        if semantic_stage:
            return semantic_stage
        if current_stage.get("name"):
            return str(current_stage["name"])

    spans = sample.get("spans") or []
    for span in spans:
        attributes = span.get("attributes") or {}
        error_value = str(attributes.get("error") or "").strip().lower()
        if str(span.get("status") or "").upper() == "ERROR" or error_value == "true":
            return "error"
    if spans:
        return _span_semantic_stage(spans[-1])
    return status or "unknown"


def _should_show_group_stage(sample: dict[str, Any]) -> bool:
    status = str(sample.get("status") or "unknown").strip().lower()
    stage = str(sample.get("stage") or "unknown").strip().lower()
    return stage != status


def _summary_key(value: Any) -> str:
    return "<unknown>" if value is None else str(value)


def _sortable_summary_value(value: Any) -> tuple[int, float | str]:
    if value is None:
        return (1, "")
    if isinstance(value, bool):
        return (0, str(value))
    if isinstance(value, (int, float)):
        return (0, float(value))
    text = str(value)
    try:
        return (0, float(text))
    except ValueError:
        return (0, text)


def _span_payload(
    span: dict[str, Any],
    tags: dict[str, Any],
    *,
    service_name: str | None,
    run_id: str | None,
) -> dict[str, Any]:
    name = str(span.get("operationName") or span.get("name") or "unknown")
    attributes = {
        key: value
        for key, value in tags.items()
        if key.startswith("xtuner.")
        or key.startswith("agent.")
        or key.startswith("session.")
        or key.startswith("judger.")
        or key.startswith("http.")
        or key.startswith("error.")
        or key.startswith("exception.")
        or key.startswith("filter.")
        or key.startswith("reward.")
        or key.startswith("oversample.")
        or key.startswith("drop.")
        or key.startswith("train.")
        or key.startswith("stage.")
        or key in {"error", "rollout.backend", "prompt.tokens", "completion.tokens", "reward_score"}
    }
    return {
        "name": name,
        "stage": _stage_from_span_name_and_attributes(name, attributes),
        "span_id": str(span.get("spanID") or span.get("span_id") or ""),
        "parent_span_id": _parent_span_id(span),
        "start_time_us": int(span.get("startTime") or 0),
        "duration_ms": float(span.get("duration") or 0) / 1000.0,
        "status": tags.get("otel.status_code") or tags.get("status.code") or "UNSET",
        "service_name": service_name,
        "run_id": run_id,
        "rollout_backend": tags.get("rollout.backend"),
        "attributes": attributes,
    }


def _span_id(span: dict[str, Any]) -> str:
    return str(span.get("spanID") or span.get("span_id") or "")


def _resolve_rollout_sample(
    entry: dict[str, Any],
    entries_by_span_id: dict[str, dict[str, Any]],
) -> tuple[Any | None, dict[str, Any]]:
    tags = entry["tags"]

    ancestor_sample: tuple[Any, dict[str, Any]] | None = None
    visited: set[str] = set()
    parent_span_id = _parent_span_id(entry["span"])
    while parent_span_id and parent_span_id not in visited:
        visited.add(parent_span_id)
        parent = entries_by_span_id.get(parent_span_id)
        if parent is None:
            break
        parent_tags = parent["tags"]
        parent_rollout_id = parent_tags.get("xtuner.rollout_id")
        if parent_rollout_id is not None:
            ancestor_sample = (parent_rollout_id, parent_tags)
        parent_span_id = _parent_span_id(parent["span"])

    if ancestor_sample is not None:
        return ancestor_sample

    rollout_id = tags.get("xtuner.rollout_id")
    if rollout_id is not None:
        return rollout_id, tags
    return None, {}


def _process_metadata(trace_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for process_id, process in (trace_data.get("processes") or {}).items():
        if not isinstance(process, dict):
            continue
        tags = _tags_to_dict(process.get("tags") or [])
        metadata[str(process_id)] = {
            "service_name": str(process["serviceName"]) if process.get("serviceName") is not None else None,
            "run_id": tags.get("run.id"),
        }
    return metadata


def _tags_to_dict(tags: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for tag in tags:
        key = tag.get("key")
        if key is None:
            continue
        result[str(key)] = tag.get("value")
    return result


def _parent_span_id(span: dict[str, Any]) -> str | None:
    for reference in span.get("references") or []:
        if reference.get("refType") == "CHILD_OF" and reference.get("spanID") is not None:
            return str(reference["spanID"])
    return None


def _normalize_jaeger_query_url(jaeger_query_url: str | None) -> str | None:
    if jaeger_query_url is None:
        return None
    stripped = jaeger_query_url.strip()
    return stripped.rstrip("/") if stripped else None


def _jaeger_trace_url(jaeger_query_url: str | None, trace_id: str) -> str | None:
    base = _normalize_jaeger_query_url(jaeger_query_url)
    if base is None:
        return None
    return f"{base}/trace/{trace_id}"


def _jaeger_trace_link_base_url(jaeger_query_url: str | None, jaeger_link_url: str | None) -> str | None:
    return _normalize_jaeger_query_url(jaeger_link_url) or _normalize_jaeger_query_url(jaeger_query_url)


def _append_unique(values: list[Any], value: Any) -> None:
    if value is not None and value not in values:
        values.append(value)


def _to_float(value: Any) -> float | Any:
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if parsed.is_integer():
        return int(parsed)
    return None


__all__ = [
    "build_rollout_view_payload_from_jaeger_traces",
    "filter_rollout_view_payload_by_train_step",
    "load_jaeger_traces_from_otel_jsonl",
    "load_live_trace_records",
]
