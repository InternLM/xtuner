from __future__ import annotations

import dataclasses
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from xtuner.v1.rl.trace import (
    TRACE_JSONL_BASENAME,
    TRACE_VIEWER_SCOPE_ALL,
    TRACE_VIEWER_SCOPE_LATEST_PRODUCE_BATCH,
    TraceEvent,
    TraceViewerScope,
)


TRACE_STAGE_LABELS = {
    "xtuner.producer.sample_group": "sampler",
    "xtuner.producer.generate_group": "producer.generate",
    "xtuner.producer.put_generated_group": "producer.put",
    "xtuner.agent_loop.generate_group": "agent_loop.generate_group",
    "xtuner.agent_loop.generate_sample": "agent_loop.generate_sample",
    "xtuner.rollout_controller.generate": "rollout.generate",
    "xtuner.rollout_worker.generate": "rollout_worker.generate",
    "xtuner.rollout_engine.generate": "engine.generate",
    "xtuner.judger.judge": "judger",
}


@dataclasses.dataclass(frozen=True)
class TraceViewerRow:
    trace_id: str
    task_name: str | None
    uid: int | str | None
    status: str | None
    latest_stage: str
    latest_timestamp_s: float
    event_count: int
    open_span: str | None = None
    open_age_s: float | None = None


@dataclasses.dataclass(frozen=True)
class OpenSpanSummary:
    span: str
    open_count: int
    oldest_age_s: float
    p50_age_s: float
    p95_age_s: float
    oldest_trace_id: str
    oldest_task_name: str | None
    oldest_uid: int | str | None


def display_trace_stage(span: str | None) -> str:
    if not span:
        return "unknown"
    if span in TRACE_STAGE_LABELS:
        return TRACE_STAGE_LABELS[span]
    if span.startswith("xtuner.") and span.endswith(".request"):
        return span.removeprefix("xtuner.")
    return span.removeprefix("xtuner.")


def load_trace_jsonl(path: str | Path) -> list[TraceEvent]:
    path = Path(path)
    if path.is_file():
        files = [path]
    else:
        files = sorted(path.glob(f"{TRACE_JSONL_BASENAME}_*.jsonl"))
    events: list[TraceEvent] = []
    for file in files:
        with file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                events.append(TraceEvent.from_dict(json.loads(line)))
    return events


def events_to_timelines(events: Iterable[TraceEvent]) -> dict[str, list[TraceEvent]]:
    timelines: dict[str, list[TraceEvent]] = defaultdict(list)
    for event in events:
        timelines[event.trace_id].append(event)
    for trace_id in timelines:
        timelines[trace_id].sort(key=lambda event: event.timestamp_s)
    return dict(timelines)


def filter_trace_events_by_scope(
    events: Iterable[TraceEvent],
    scope: TraceViewerScope = TRACE_VIEWER_SCOPE_LATEST_PRODUCE_BATCH,
) -> list[TraceEvent]:
    event_list = list(events)
    if scope == TRACE_VIEWER_SCOPE_ALL:
        return event_list
    if scope != TRACE_VIEWER_SCOPE_LATEST_PRODUCE_BATCH:
        raise ValueError(f"Unsupported trace viewer scope: {scope!r}")

    latest_batch_id = _latest_produce_batch_id(event_list)
    if latest_batch_id is not None:
        return [event for event in event_list if event.produce_batch_id == latest_batch_id]

    latest_key = _latest_produce_batch_key(event_list)
    if latest_key is None:
        return event_list
    return [event for event in event_list if _produce_batch_key(event) == latest_key]


def build_viewer_rows(
    timelines: dict[str, list[TraceEvent]] | Iterable[TraceEvent],
    *,
    now_s: float | None = None,
) -> list[TraceViewerRow]:
    if not isinstance(timelines, dict):
        timelines = events_to_timelines(timelines)
    now_s = time.time() if now_s is None else now_s
    rows: list[TraceViewerRow] = []
    for trace_id, events in timelines.items():
        if not events:
            continue
        sorted_events = sorted(events, key=lambda event: event.timestamp_s)
        latest = sorted_events[-1]
        open_spans = _get_open_spans(sorted_events)
        newest_open = max(open_spans, key=lambda span: span[1].timestamp_s) if open_spans else None
        open_span = newest_open[0] if newest_open is not None else None
        open_age_s = now_s - newest_open[1].timestamp_s if newest_open is not None else None
        rows.append(
            TraceViewerRow(
                trace_id=trace_id,
                task_name=latest.task_name,
                uid=latest.uid,
                status=latest.status,
                latest_stage=latest.stage,
                latest_timestamp_s=latest.timestamp_s,
                event_count=len(sorted_events),
                open_span=open_span,
                open_age_s=open_age_s,
            )
        )
    return sorted(rows, key=lambda row: ((row.open_age_s or 0.0), row.latest_timestamp_s), reverse=True)


def build_open_span_summaries(
    timelines: dict[str, list[TraceEvent]] | Iterable[TraceEvent],
    *,
    now_s: float | None = None,
) -> list[OpenSpanSummary]:
    if not isinstance(timelines, dict):
        timelines = events_to_timelines(timelines)
    now_s = time.time() if now_s is None else now_s
    grouped: dict[str, list[tuple[float, str, TraceEvent]]] = defaultdict(list)
    for trace_id, events in timelines.items():
        for span, start_event in _get_open_spans(sorted(events, key=lambda event: event.timestamp_s)):
            grouped[span].append((now_s - start_event.timestamp_s, trace_id, start_event))

    summaries: list[OpenSpanSummary] = []
    for span, entries in grouped.items():
        entries.sort(key=lambda item: item[0], reverse=True)
        ages = sorted(age for age, _, _ in entries)
        oldest_age, oldest_trace_id, oldest_event = entries[0]
        summaries.append(
            OpenSpanSummary(
                span=span,
                open_count=len(entries),
                oldest_age_s=oldest_age,
                p50_age_s=_percentile(ages, 0.50),
                p95_age_s=_percentile(ages, 0.95),
                oldest_trace_id=oldest_trace_id,
                oldest_task_name=oldest_event.task_name,
                oldest_uid=oldest_event.uid,
            )
        )
    return sorted(summaries, key=lambda summary: (summary.oldest_age_s, summary.open_count), reverse=True)


def _latest_produce_batch_key(events: list[TraceEvent]) -> tuple[int, int, int] | None:
    keys: list[tuple[int, int, int]] = []
    for event in events:
        key = _produce_batch_key(event)
        if key is not None:
            keys.append(key)
    return max(keys) if keys else None


def _latest_produce_batch_id(events: list[TraceEvent]) -> str | None:
    latest_event: TraceEvent | None = None
    latest_sort_key: tuple[int, int, int, float] | None = None
    for event in events:
        if event.produce_batch_id is None:
            continue
        batch_key = _produce_batch_key(event)
        if batch_key is None:
            sort_key = (-1, -1, -1, event.timestamp_s)
        else:
            sort_key = (*batch_key, event.timestamp_s)
        if latest_sort_key is None or sort_key > latest_sort_key:
            latest_event = event
            latest_sort_key = sort_key
    if latest_event is None:
        return None
    return latest_event.produce_batch_id


def _produce_batch_key(event: TraceEvent) -> tuple[int, int, int] | None:
    if event.train_step is None:
        return None
    return (
        event.train_step,
        -1 if event.model_step is None else event.model_step,
        -1 if event.producer_future_step is None else event.producer_future_step,
    )


def _get_open_spans(events: list[TraceEvent]) -> list[tuple[str, TraceEvent]]:
    stacks: dict[str, list[TraceEvent]] = defaultdict(list)
    for event in events:
        span, suffix = _split_span_stage(event.stage)
        if span is None or suffix is None:
            continue
        if suffix == "start":
            stacks[span].append(event)
        elif suffix in {"end", "error"} and stacks.get(span):
            stacks[span].pop()
    open_spans: list[tuple[str, TraceEvent]] = []
    for span, stack in stacks.items():
        open_spans.extend((span, event) for event in stack)
    return open_spans


def _split_span_stage(stage: str) -> tuple[str | None, str | None]:
    for suffix in (".start", ".end", ".error"):
        if stage.endswith(suffix):
            return stage[: -len(suffix)], suffix[1:]
    return None, None


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction
