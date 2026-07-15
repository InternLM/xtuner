"""Build the XTuner trace-viewer payload from Jaeger-shaped traces.

This module owns XTuner semantics: rollout grouping, stage names, train-step
filtering, duration summaries, error summaries, and display paths. It does not
serve HTTP or render HTML.
"""

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
    service_name: str | None = None,
    run_id: str | None = None,
    train_step: Any = "latest",
) -> dict[str, Any]:
    samples_by_key: dict[tuple[str, Any], dict[str, Any]] = {}
    jaeger_trace_link_base_url = _normalize_jaeger_query_url(jaeger_link_url) or _normalize_jaeger_query_url(
        jaeger_query_url
    )

    for trace_data in traces:
        if not isinstance(trace_data, dict):
            raise ValueError(f"Jaeger trace must be an object, got {type(trace_data).__name__}")
        trace_id = str(trace_data.get("traceID") or trace_data.get("trace_id") or "")
        if not trace_id:
            raise ValueError("Jaeger trace is missing traceID")
        process_metadata = _process_metadata(trace_data)
        span_entries = []
        entries_by_span_id: dict[str, dict[str, Any]] = {}
        spans = trace_data.get("spans") or []
        if not isinstance(spans, list):
            raise ValueError(f"Jaeger trace {trace_id} has non-list spans")
        if spans and not process_metadata:
            raise ValueError(f"Jaeger trace {trace_id} has spans but no processes")
        for span in spans:
            if not isinstance(span, dict):
                raise ValueError(f"Jaeger trace {trace_id} span must be an object")
            span_id = str(span.get("spanID") or span.get("span_id") or "")
            if not span_id:
                raise ValueError(f"Jaeger trace {trace_id} span is missing spanID")
            process_id = str(span.get("processID") or "")
            if process_metadata:
                if not process_id:
                    raise ValueError(f"Jaeger trace {trace_id} span {span_id} is missing processID")
                process = process_metadata.get(process_id)
                if process is None:
                    raise ValueError(f"Jaeger trace {trace_id} span {span_id} references unknown processID {process_id}")
            else:
                process = {}
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
                "span_id": span_id,
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
                    "status": _sample_status_from_span(span, tags),
                    "service_name": span_service_name,
                    "run_id": span_run_id,
                    "jaeger_url": f"{jaeger_trace_link_base_url}/trace/{trace_id}"
                    if jaeger_trace_link_base_url is not None
                    else None,
                    "spans": [],
                },
            )
            for sample_key, tag_key in (
                ("rollout_id", "xtuner.rollout_id"),
                ("group_id", "xtuner.group_id"),
                ("task_name", "xtuner.task_name"),
            ):
                if sample_tags.get(tag_key) is not None:
                    sample[sample_key] = sample_tags[tag_key]
            status = _sample_status_from_span(span, tags)
            if status is not None:
                current_status = sample.get("status")
                if current_status is None or _sample_status_priority(status) >= _sample_status_priority(current_status):
                    sample["status"] = status
            producer_future_step = _producer_future_step(sample_tags)
            if producer_future_step is not None:
                sample["producer_future_step"] = producer_future_step
            if span_run_id is not None:
                sample["run_id"] = span_run_id
            sample["spans"].append(_span_payload(span, tags, service_name=span_service_name, run_id=span_run_id))

    generated_at_s = time.time()
    samples = []
    for sample in samples_by_key.values():
        sample["spans"].sort(key=lambda item: (item["start_time_us"], item["span_id"]))
        sample["span_count"] = len(sample["spans"])
        if sample.get("status") is None:
            if _has_finished_sample_root_span(sample["spans"]):
                sample["status"] = "completed"
            elif sample.get("spans"):
                sample["status"] = "running"
        sample["display_path"] = _build_display_path(sample)
        sample["chain"] = " -> ".join(node["name"] for node in sample["display_path"])
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
        raise FileNotFoundError(f"trace_jsonl path does not exist: {path}")

    for line_no, record in _iter_jsonl_records(path):
        context = f"{path}:{line_no}"
        jaeger_traces = _jaeger_traces_from_json_record(record, context=context)
        if jaeger_traces is not None:
            for trace_data in jaeger_traces:
                _merge_jaeger_trace(traces_by_id, trace_data, context=context)
            continue
        if "resourceSpans" not in record:
            raise ValueError(f"{context}: trace record must contain Jaeger data or OTLP resourceSpans")
        resource_spans = record.get("resourceSpans")
        if not isinstance(resource_spans, list):
            raise ValueError(f"{context}: OTLP resourceSpans must be a list")
        for resource_index, resource_span in enumerate(resource_spans):
            resource_attrs = _otel_attributes_to_dict((resource_span.get("resource") or {}).get("attributes") or [])
            service_name = str(resource_attrs.get("service.name") or "unknown")
            process_tags = _dict_to_jaeger_tags(resource_attrs)
            scope_spans = resource_span.get("scopeSpans")
            if scope_spans is None:
                scope_spans = resource_span.get("instrumentationLibrarySpans") or []
            for scope_index, scope_span in enumerate(scope_spans):
                for span_index, otel_span in enumerate(scope_span.get("spans") or []):
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
                        raise ValueError(
                            f"{context}: resourceSpans[{resource_index}].scopeSpans[{scope_index}].spans[{span_index}] "
                            "is missing traceId or spanId"
                        )
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


def _iter_jsonl_records(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc.msg}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_no}: trace JSONL record must be an object")
            yield line_no, payload


def _jaeger_traces_from_json_record(record: dict[str, Any], *, context: str) -> list[dict[str, Any]] | None:
    data = record.get("data")
    if data is not None:
        if not isinstance(data, list):
            raise ValueError(f"{context}: Jaeger data must be a list")
        return data
    if record.get("traceID") is not None and isinstance(record.get("spans"), list):
        return [record]
    if record.get("traceID") is not None:
        raise ValueError(f"{context}: Jaeger trace spans must be a list")
    return None


def _merge_jaeger_trace(traces_by_id: dict[str, dict[str, Any]], trace_data: dict[str, Any], *, context: str) -> None:
    trace_id = str(trace_data.get("traceID") or trace_data.get("trace_id") or "")
    if not trace_id:
        raise ValueError(f"{context}: Jaeger trace is missing traceID")
    target = traces_by_id.setdefault(trace_id, {"traceID": trace_id, "processes": {}, "spans": []})
    processes = trace_data.get("processes") or {}
    if not isinstance(processes, dict):
        raise ValueError(f"{context}: Jaeger trace {trace_id} processes must be an object")
    target["processes"].update(processes)
    spans = trace_data.get("spans") or []
    if not isinstance(spans, list):
        raise ValueError(f"{context}: Jaeger trace {trace_id} spans must be a list")
    target["spans"].extend(spans)


def _otel_span_to_jaeger_span(
    otel_span: dict[str, Any],
    trace_id: str,
    span_id: str,
    process_id: str,
) -> dict[str, Any]:
    attributes = _otel_attributes_to_dict(otel_span.get("attributes") or [])
    tags = _dict_to_jaeger_tags(attributes)
    tags.extend(_otel_status_tags(otel_span.get("status") or {}))
    start_time = (
        otel_span.get("startTimeUnixNano")
        or otel_span.get("start_time_unix_nano")
        or otel_span.get("startTime")
        or 0
    )
    start_ns = int(str(start_time))
    end_time = (
        otel_span.get("endTimeUnixNano")
        or otel_span.get("end_time_unix_nano")
        or otel_span.get("endTime")
        or start_ns
    )
    end_ns = int(str(end_time))
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


def _otel_status_tags(status: Any) -> list[dict[str, Any]]:
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
        key = attribute.get("key")
        if key is not None:
            result[str(key)] = _otel_any_value_to_python(attribute.get("value"))
    return result


def _otel_any_value_to_python(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, dict):
        return value
    if "stringValue" in value:
        return value["stringValue"]
    if "boolValue" in value:
        return bool(value["boolValue"])
    if "intValue" in value:
        return int(str(value["intValue"]))
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
    tags = []
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, bool):
            tag_type = "bool"
        elif isinstance(value, int):
            tag_type = "int64"
        elif isinstance(value, float):
            tag_type = "float64"
        else:
            tag_type = "string"
        tags.append({"key": str(key), "type": tag_type, "value": value})
    return tags


def _sample_status_priority(status: Any) -> int:
    normalized = str(status or "").strip().lower()
    if normalized in _ERROR_SAMPLE_STATUSES:
        return 3
    if normalized in _TERMINAL_STAGE_STATUSES:
        return 2
    if normalized in _NON_TERMINAL_SAMPLE_STATUSES:
        return 0
    return 1


def _sample_status_from_span(span: dict[str, Any], tags: dict[str, Any]) -> Any:
    status = tags.get("xtuner.status")
    if status is None:
        return None
    normalized = str(status).strip().lower()
    if normalized in _ERROR_SAMPLE_STATUSES:
        return status
    return None


def _has_finished_sample_root_span(spans: list[dict[str, Any]]) -> bool:
    return any(_is_sample_root_span(span) for span in spans)


def _is_sample_root_span(span: dict[str, Any]) -> bool:
    attributes = span.get("attributes") or {}
    span_path = _span_name_path(attributes)
    if span_path:
        return len(span_path) == 1
    return span.get("parent_span_id") is None


def _build_display_path(sample: dict[str, Any]) -> list[dict[str, Any]]:
    spans = sample.get("spans") or []
    spans_by_name = {str(span.get("name") or ""): span for span in spans}
    if str(sample.get("status") or "").strip().lower() == "running" and not _has_finished_sample_root_span(spans):
        path = _running_display_path_names(spans)
    else:
        path = _display_path_names(spans)
    nodes = []
    for name in path:
        span = spans_by_name.get(name)
        if span is not None:
            nodes.append(
                {
                    "name": name,
                    "stage": _span_semantic_stage(span),
                    "source": "span",
                    "status": "done" if str(span.get("status") or "").upper() != "ERROR" else "error",
                    "duration_ms": span.get("duration_ms"),
                }
            )
        else:
            nodes.append({"name": name, "source": "logical", "status": "inferred"})
    return nodes


def _running_display_path_names(spans: list[dict[str, Any]]) -> list[str]:
    roots = []
    for span in spans:
        attributes = span.get("attributes") or {}
        span_path = _span_name_path(attributes)
        if span_path:
            roots.append(span_path[0])
        elif span.get("parent_span_id") is None and span.get("name"):
            roots.append(str(span["name"]))
    unique_roots = []
    for name in roots:
        if name not in unique_roots:
            unique_roots.append(name)
    return unique_roots


def _display_path_names(spans: list[dict[str, Any]]) -> list[str]:
    span_paths = []
    for span in spans:
        attributes = span.get("attributes") or {}
        span_paths.append(_span_name_path(attributes))
    path = []
    for span_path in span_paths:
        if not span_path:
            continue
        common_prefix_len = 0
        while (
            common_prefix_len < len(path)
            and common_prefix_len < len(span_path)
            and path[common_prefix_len] == span_path[common_prefix_len]
        ):
            common_prefix_len += 1
        path.extend(span_path[common_prefix_len:])
    if not path:
        path = [str(span.get("name") or "") for span in spans if span.get("name")]
    unique_path = []
    for name in path:
        if not unique_path or unique_path[-1] != name:
            unique_path.append(name)
    return unique_path


def _span_name_path(attributes: dict[str, Any]) -> list[str]:
    value = attributes.get(_SPAN_NAME_PATH_ATTRIBUTE) or attributes.get(_LEGACY_LOGICAL_PATH_ATTRIBUTE)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = [part.strip() for part in value.split("->")]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if isinstance(item, str) and item.strip()]
    return []


def _producer_future_step(tags: dict[str, Any]) -> Any:
    value = tags.get("xtuner.producer_future_step")
    return value if value is not None else tags.get("xtuner.train_step")


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


def filter_rollout_view_payload_by_train_step(payload: dict[str, Any], train_step: Any = "latest") -> dict[str, Any]:
    samples = list(payload.get("samples") or [])
    generated_at_s = float(payload.get("generated_at_s") or time.time())
    available_train_steps = list(payload.get("available_train_steps") or _available_train_steps(samples))
    requested_train_step = _requested_train_step(train_step)
    selected_train_step = _select_train_step(train_step, available_train_steps)
    if selected_train_step == "all":
        visible_samples = samples
    else:
        visible_samples = [
            sample for sample in samples if str(sample.get("producer_future_step")) == str(selected_train_step)
        ]

    status_counts: Counter[str] = Counter()
    group_ids: set[Any] = set()
    visible_steps: set[Any] = set()
    for sample in visible_samples:
        status = str(sample.get("status") or "unknown")
        status_counts[status] += 1
        if sample.get("group_id") is not None:
            group_ids.add(sample["group_id"])
        if sample.get("producer_future_step") is not None:
            visible_steps.add(sample["producer_future_step"])

    filtered_payload = dict(payload)
    filtered_payload.update(
        {
            "generated_at_s": generated_at_s,
            "requested_train_step": requested_train_step,
            "selected_train_step": selected_train_step,
            "available_train_steps": available_train_steps,
            "sample_count": len(visible_samples),
            "group_count": len(group_ids),
            "step_count": len(visible_steps),
            "stage_duration_summaries": _build_stage_duration_summaries(visible_samples),
            "status_counts": dict(sorted(status_counts.items())),
            "samples": visible_samples,
        }
    )
    return filtered_payload


def _requested_train_step(train_step: Any) -> Any:
    if train_step is None:
        return "latest"
    if isinstance(train_step, str) and not train_step.strip():
        return "latest"
    return train_step


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
        try:
            values["reward_score"] = float(values["reward_score"])
        except (TypeError, ValueError):
            pass
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
    if status in _ERROR_SAMPLE_STATUSES:
        return status

    spans = sample.get("spans") or []
    for span in spans:
        attributes = span.get("attributes") or {}
        error_value = str(attributes.get("error") or "").strip().lower()
        if str(span.get("status") or "").upper() == "ERROR" or error_value == "true":
            return "error"
    if spans:
        return _span_semantic_stage(spans[-1])
    return status or "unknown"


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
    span_id = str(span.get("spanID") or span.get("span_id") or "")
    name_value = span.get("operationName") or span.get("name")
    if not name_value:
        raise ValueError(f"Jaeger span {span_id} is missing operationName")
    if span.get("startTime") is None:
        raise ValueError(f"Jaeger span {span_id} is missing startTime")
    if span.get("duration") is None:
        raise ValueError(f"Jaeger span {span_id} is missing duration")
    name = str(name_value)
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
        "span_id": span_id,
        "parent_span_id": _parent_span_id(span),
        "start_time_us": int(span["startTime"]),
        "duration_ms": float(span["duration"]) / 1000.0,
        "status": tags.get("otel.status_code") or tags.get("status.code") or "UNSET",
        "service_name": service_name,
        "run_id": run_id,
        "rollout_backend": tags.get("rollout.backend"),
        "attributes": attributes,
    }


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
    processes = trace_data.get("processes") or {}
    if not isinstance(processes, dict):
        trace_id = str(trace_data.get("traceID") or trace_data.get("trace_id") or "")
        raise ValueError(f"Jaeger trace {trace_id} processes must be an object")
    for process_id, process in processes.items():
        if not isinstance(process, dict):
            raise ValueError(f"Jaeger process {process_id} must be an object")
        tags = _tags_to_dict(process.get("tags") or [])
        if process.get("serviceName") is None:
            raise ValueError(f"Jaeger process {process_id} is missing serviceName")
        metadata[str(process_id)] = {
            "service_name": str(process["serviceName"]),
            "run_id": tags.get("run.id"),
        }
    return metadata


def _tags_to_dict(tags: Any) -> dict[str, Any]:
    if not isinstance(tags, list):
        raise ValueError(f"Jaeger tags must be a list, got {type(tags).__name__}")
    result: dict[str, Any] = {}
    for index, tag in enumerate(tags):
        if not isinstance(tag, dict):
            raise ValueError(f"Jaeger tags[{index}] must be an object")
        key = tag.get("key")
        if key is None:
            raise ValueError(f"Jaeger tags[{index}] is missing key")
        result[str(key)] = tag.get("value")
    return result


def _parent_span_id(span: dict[str, Any]) -> str | None:
    references = span.get("references") or []
    if not isinstance(references, list):
        raise ValueError("Jaeger span references must be a list")
    for index, reference in enumerate(references):
        if not isinstance(reference, dict):
            raise ValueError(f"Jaeger span references[{index}] must be an object")
        if reference.get("refType") == "CHILD_OF" and reference.get("spanID") is not None:
            return str(reference["spanID"])
    return None


def _normalize_jaeger_query_url(jaeger_query_url: str | None) -> str | None:
    if jaeger_query_url is None:
        return None
    stripped = jaeger_query_url.strip()
    return stripped.rstrip("/") if stripped else None


def _append_unique(values: list[Any], value: Any) -> None:
    if value is not None and value not in values:
        values.append(value)


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


__all__ = [
    "build_rollout_view_payload_from_jaeger_traces",
    "filter_rollout_view_payload_by_train_step",
    "load_jaeger_traces_from_otel_jsonl",
]
