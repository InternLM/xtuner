from __future__ import annotations

import argparse
import dataclasses
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from xtuner.v1.rl.trace import (
    TRACE_VIEWER_SCOPE_LATEST_PRODUCE_BATCH,
    TraceEvent,
    TraceViewerScope,
)
from xtuner.tools.producer_trace_analysis import (
    _percentile,
    _split_span_stage,
    display_trace_stage,
    filter_trace_events_by_scope,
    load_trace_jsonl,
)


_PALETTE = [
    "#2563eb",
    "#059669",
    "#d97706",
    "#7c3aed",
    "#dc2626",
    "#0891b2",
    "#9333ea",
    "#4d7c0f",
    "#be123c",
    "#0f766e",
    "#b45309",
    "#475569",
]


@dataclass
class TraceSpanRecord:
    trace_id: str
    span: str
    display_stage: str
    start_s: float
    end_s: float
    duration_s: float
    outcome: str
    depth: int = 0
    task_name: str | None = None
    uid: int | str | None = None
    status: str | None = None
    train_step: int | None = None
    worker_rank: int | None = None
    error_msg: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an offline Producer Trace hotspot timeline HTML.")
    parser.add_argument("trace_dir", type=Path, help="Directory containing producer_trace_*.jsonl files.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output HTML file. Defaults to <trace_dir>/producer_trace_hotspots.html.",
    )
    parser.add_argument(
        "--scope",
        choices=("latest-produce-batch", "all"),
        default=TRACE_VIEWER_SCOPE_LATEST_PRODUCE_BATCH,
        help="Trace scope shown by the hotspot viewer. Defaults to the latest produce batch.",
    )
    return parser.parse_args()


def build_trace_span_records(
    events: Iterable[TraceEvent],
    *,
    include_open: bool = True,
    now_s: float | None = None,
) -> list[TraceSpanRecord]:
    events_by_trace: dict[str, list[TraceEvent]] = defaultdict(list)
    latest_event_s = 0.0
    for event in events:
        events_by_trace[event.trace_id].append(event)
        latest_event_s = max(latest_event_s, event.timestamp_s)
    now_s = latest_event_s if now_s is None else now_s

    records: list[TraceSpanRecord] = []
    for trace_id, trace_events in events_by_trace.items():
        stacks: dict[str, list[TraceEvent]] = defaultdict(list)
        for event in sorted(trace_events, key=lambda item: item.timestamp_s):
            span, suffix = _split_span_stage(event.stage)
            if span is None or suffix is None:
                continue
            if suffix == "start":
                stacks[span].append(event)
                continue
            if suffix in {"end", "error"} and stacks.get(span):
                start_event = stacks[span].pop()
                records.append(_build_span_record(start_event, event, outcome=suffix))
                continue
            if suffix in {"end", "error"} and event.elapsed_s is not None:
                records.append(_build_elapsed_only_span_record(event, span=span, outcome=suffix))

        if include_open:
            for span_starts in stacks.values():
                for start_event in span_starts:
                    span, _ = _split_span_stage(start_event.stage)
                    if span is None:
                        continue
                    records.append(_build_open_span_record(start_event, span=span, now_s=now_s))

    _assign_depths(records)
    return sorted(records, key=lambda record: (record.trace_id, record.start_s, record.depth, record.end_s))


def build_timeline_stage_records(records: Iterable[TraceSpanRecord]) -> list[TraceSpanRecord]:
    return sorted(records, key=lambda record: (record.trace_id, record.start_s, record.depth, record.end_s))


def build_hotspot_payload(
    trace_dir: Path,
    *,
    scope: TraceViewerScope = TRACE_VIEWER_SCOPE_LATEST_PRODUCE_BATCH,
) -> dict[str, Any]:
    return build_hotspot_payload_from_events(load_trace_jsonl(trace_dir), trace_source=str(trace_dir), scope=scope)


def build_hotspot_payload_from_events(
    events: Iterable[TraceEvent],
    *,
    trace_source: str,
    scope: TraceViewerScope = TRACE_VIEWER_SCOPE_LATEST_PRODUCE_BATCH,
) -> dict[str, Any]:
    events = filter_trace_events_by_scope(events, scope)
    raw_records = build_trace_span_records(events)
    records = build_timeline_stage_records(raw_records)
    if records:
        global_start_s = min(record.start_s for record in records)
        global_end_s = max(record.end_s for record in records)
    else:
        global_start_s = 0.0
        global_end_s = 1.0
    global_window_s = max(0.001, global_end_s - global_start_s)
    stage_colors = _build_stage_colors(records)

    rows = []
    for trace_id, trace_records in _group_records(records).items():
        trace_start = min(record.start_s for record in trace_records)
        trace_end = max(record.end_s for record in trace_records)
        trace_window_s = max(0.001, trace_end - trace_start)
        max_depth = max(record.depth for record in trace_records) if trace_records else 0
        rows.append(
            {
                "trace_id": trace_id,
                "task_name": trace_records[-1].task_name,
                "uid": trace_records[-1].uid,
                "duration_s": trace_end - trace_start,
                "start_s": trace_start,
                "end_s": trace_end,
                "row_height_px": 56 + (max_depth + 1) * 22,
                "spans": [
                    _span_record_to_payload(record, trace_start, trace_window_s, stage_colors)
                    for record in trace_records
                ],
            }
        )
    rows.sort(key=lambda row: row["duration_s"], reverse=True)

    return {
        "title": "Producer Trace Hotspots",
        "trace_source": trace_source,
        "scope": scope,
        "task_count": len(rows),
        "span_count": len(records),
        "raw_span_count": len(raw_records),
        "timeline_start_s": global_start_s,
        "timeline_end_s": global_end_s,
        "timeline_duration_s": global_window_s,
        "max_task_duration_s": max((row["duration_s"] for row in rows), default=0.0),
        "scale_mode": "task_relative",
        "stage_colors": stage_colors,
        "stage_stats": _build_stage_stats(records),
        "rows": rows,
    }


def write_hotspot_html(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_hotspot_html(payload), encoding="utf-8")


def render_hotspot_html(payload: dict[str, Any]) -> str:
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return _HTML_TEMPLATE.replace("__TRACE_DATA__", data_json)


def main() -> None:
    args = parse_args()
    trace_dir = args.trace_dir
    output_path = args.output or trace_dir / "producer_trace_hotspots.html"
    payload = build_hotspot_payload(trace_dir, scope=args.scope)
    write_hotspot_html(payload, output_path)
    print(output_path)


def _build_span_record(start_event: TraceEvent, end_event: TraceEvent, *, outcome: str) -> TraceSpanRecord:
    span, _ = _split_span_stage(start_event.stage)
    assert span is not None
    duration_s = max(0.0, end_event.timestamp_s - start_event.timestamp_s)
    return TraceSpanRecord(
        trace_id=start_event.trace_id,
        span=span,
        display_stage=display_trace_stage(span),
        start_s=start_event.timestamp_s,
        end_s=end_event.timestamp_s,
        duration_s=duration_s,
        outcome=outcome,
        task_name=end_event.task_name or start_event.task_name,
        uid=end_event.uid if end_event.uid is not None else start_event.uid,
        status=end_event.status or start_event.status,
        train_step=end_event.train_step if end_event.train_step is not None else start_event.train_step,
        worker_rank=end_event.worker_rank if end_event.worker_rank is not None else start_event.worker_rank,
        error_msg=end_event.error_msg,
    )


def _build_elapsed_only_span_record(end_event: TraceEvent, *, span: str, outcome: str) -> TraceSpanRecord:
    elapsed_s = max(0.0, end_event.elapsed_s or 0.0)
    return TraceSpanRecord(
        trace_id=end_event.trace_id,
        span=span,
        display_stage=display_trace_stage(span),
        start_s=end_event.timestamp_s - elapsed_s,
        end_s=end_event.timestamp_s,
        duration_s=elapsed_s,
        outcome=outcome,
        task_name=end_event.task_name,
        uid=end_event.uid,
        status=end_event.status,
        train_step=end_event.train_step,
        worker_rank=end_event.worker_rank,
        error_msg=end_event.error_msg,
    )


def _build_open_span_record(start_event: TraceEvent, *, span: str, now_s: float) -> TraceSpanRecord:
    duration_s = max(0.0, now_s - start_event.timestamp_s)
    return TraceSpanRecord(
        trace_id=start_event.trace_id,
        span=span,
        display_stage=display_trace_stage(span),
        start_s=start_event.timestamp_s,
        end_s=now_s,
        duration_s=duration_s,
        outcome="open",
        task_name=start_event.task_name,
        uid=start_event.uid,
        status=start_event.status,
        train_step=start_event.train_step,
        worker_rank=start_event.worker_rank,
        error_msg=start_event.error_msg,
    )


def _assign_depths(records: list[TraceSpanRecord]) -> None:
    records_by_trace = _group_records(records)
    for trace_records in records_by_trace.values():
        active_end_times: list[float] = []
        for record in sorted(trace_records, key=lambda item: (item.start_s, -item.end_s)):
            active_end_times = [end_s for end_s in active_end_times if end_s > record.start_s]
            record.depth = len(active_end_times)
            active_end_times.append(record.end_s)


def _group_records(records: Iterable[TraceSpanRecord]) -> dict[str, list[TraceSpanRecord]]:
    grouped: dict[str, list[TraceSpanRecord]] = defaultdict(list)
    for record in records:
        grouped[record.trace_id].append(record)
    for trace_id in grouped:
        grouped[trace_id].sort(key=lambda record: (record.start_s, record.depth, record.end_s))
    return dict(grouped)


def _span_record_to_payload(
    record: TraceSpanRecord,
    timeline_start_s: float,
    timeline_duration_s: float,
    stage_colors: dict[str, str],
) -> dict[str, Any]:
    left_pct = (record.start_s - timeline_start_s) / timeline_duration_s * 100.0
    width_pct = max(0.2, record.duration_s / timeline_duration_s * 100.0)
    return {
        **dataclasses.asdict(record),
        "left_pct": left_pct,
        "width_pct": min(width_pct, max(0.2, 100.0 - left_pct)),
        "top_px": record.depth * 22,
        "color": stage_colors[record.display_stage],
    }


def _build_stage_colors(records: Iterable[TraceSpanRecord]) -> dict[str, str]:
    stages = sorted({record.display_stage for record in records})
    return {stage: _PALETTE[index % len(_PALETTE)] for index, stage in enumerate(stages)}


def _build_stage_stats(records: Iterable[TraceSpanRecord]) -> list[dict[str, Any]]:
    grouped: dict[str, list[TraceSpanRecord]] = defaultdict(list)
    for record in records:
        grouped[record.display_stage].append(record)

    stats = []
    for stage, stage_records in grouped.items():
        durations = sorted(record.duration_s for record in stage_records)
        stats.append(
            {
                "stage": stage,
                "count": len(stage_records),
                "open_count": sum(1 for record in stage_records if record.outcome == "open"),
                "error_count": sum(1 for record in stage_records if record.outcome == "error"),
                "total_s": sum(durations),
                "avg_s": sum(durations) / len(durations),
                "p50_s": _percentile(durations, 0.50),
                "p95_s": _percentile(durations, 0.95),
                "max_s": durations[-1],
            }
        )
    return sorted(stats, key=lambda item: (item["p95_s"], item["total_s"]), reverse=True)


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Producer Trace Hotspots</title>
  <style>
    :root {
      --bg: #f7f8fa;
      --panel: #ffffff;
      --line: #d9dde4;
      --text: #1f2937;
      --muted: #637083;
      --danger: #b42318;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 13px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      display: flex;
      justify-content: space-between;
      gap: 24px;
      padding: 18px 24px 12px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    h1, h2 { margin: 0; font-weight: 650; letter-spacing: 0; }
    h1 { font-size: 20px; }
    h2 { font-size: 14px; margin-bottom: 8px; }
    .subtle { color: var(--muted); }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .metrics {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      padding: 12px 24px 0;
    }
    .metric, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .metric {
      min-width: 150px;
      padding: 10px 12px;
    }
    .metric strong {
      display: block;
      font-size: 20px;
      line-height: 1.2;
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(360px, 420px) minmax(560px, 1fr);
      gap: 16px;
      padding: 16px 24px 24px;
    }
    .panel-body { padding: 12px; }
    table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
    }
    th, td {
      padding: 7px 8px;
      border-bottom: 1px solid #edf0f4;
      text-align: left;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    th {
      color: var(--muted);
      background: #fafbfc;
      font-weight: 600;
      position: sticky;
      top: 0;
      z-index: 1;
    }
    .table-wrap {
      max-height: 360px;
      overflow: auto;
    }
    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 7px;
      margin-top: 12px;
    }
    .legend-item {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      max-width: 240px;
    }
    .swatch {
      width: 12px;
      height: 12px;
      border-radius: 3px;
      flex: 0 0 auto;
    }
    .timeline-panel {
      min-width: 0;
    }
    .timeline-scroll {
      overflow: auto;
      max-height: 760px;
      border-top: 1px solid #edf0f4;
    }
    .trace-row {
      display: grid;
      grid-template-columns: 190px minmax(0, 1fr);
      border-bottom: 1px solid #edf0f4;
      min-width: 920px;
    }
    .trace-label {
      padding: 8px;
      border-right: 1px solid #edf0f4;
      background: #fbfcfd;
      overflow: hidden;
    }
    .trace-id {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-weight: 600;
    }
    .trace-lane {
      position: relative;
      margin: 8px;
      min-height: 24px;
      background: repeating-linear-gradient(
        to right,
        #eef2f6 0,
        #eef2f6 1px,
        transparent 1px,
        transparent 10%
      );
    }
    .span-block {
      position: absolute;
      height: 18px;
      border-radius: 4px;
      color: #fff;
      padding: 1px 5px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 11px;
      line-height: 16px;
      box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.12);
    }
    .span-block.open {
      outline: 2px dashed var(--danger);
      outline-offset: 1px;
    }
    .span-block.error {
      box-shadow: inset 0 0 0 2px var(--danger);
    }
    .empty {
      color: var(--muted);
      padding: 18px 4px;
      text-align: center;
    }
    @media (max-width: 980px) {
      header { flex-direction: column; }
      .layout { grid-template-columns: 1fr; }
      .trace-row { grid-template-columns: 150px minmax(0, 1fr); }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Producer Trace Hotspots</h1>
      <div class="subtle mono" id="trace-source"></div>
    </div>
    <div class="subtle" id="timeline-range"></div>
  </header>

  <section class="metrics">
    <div class="metric"><span class="subtle">Tasks</span><strong id="metric-tasks">0</strong></div>
    <div class="metric"><span class="subtle">Spans</span><strong id="metric-spans">0</strong></div>
    <div class="metric"><span class="subtle">Max Task Duration</span><strong id="metric-window">0s</strong></div>
  </section>

  <main class="layout">
    <section class="panel">
      <div class="panel-body">
        <h2>Stage Hotspots</h2>
        <div class="table-wrap">
          <table>
            <thead><tr><th>Stage</th><th style="width:64px">Count</th><th style="width:70px">Avg</th><th style="width:70px">P95</th><th style="width:70px">Max</th></tr></thead>
            <tbody id="stage-stats"></tbody>
          </table>
        </div>
        <div class="legend" id="legend"></div>
      </div>
    </section>

    <section class="panel timeline-panel">
      <div class="panel-body">
        <h2>Task Timeline</h2>
        <div class="subtle">Each row is one task. The x-axis is normalized to that task's own duration.</div>
      </div>
      <div class="timeline-scroll" id="timeline"></div>
    </section>
  </main>

  <script id="trace-data" type="application/json">__TRACE_DATA__</script>
  <script>
    const data = JSON.parse(document.getElementById("trace-data").textContent);
    const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (ch) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    }[ch]));
    const fmtSeconds = (seconds) => {
      if (seconds === null || seconds === undefined) return "";
      if (seconds < 60) return `${seconds.toFixed(1)}s`;
      if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`;
      return `${(seconds / 3600).toFixed(1)}h`;
    };

    function renderMetrics() {
      document.getElementById("trace-source").textContent = data.trace_source || "";
      document.getElementById("timeline-range").textContent =
        `Task-relative timeline across ${fmtSeconds(data.timeline_duration_s || 0)} trace window`;
      document.getElementById("metric-tasks").textContent = data.task_count || 0;
      document.getElementById("metric-spans").textContent = data.span_count || 0;
      document.getElementById("metric-window").textContent = fmtSeconds(data.max_task_duration_s || 0);
    }

    function renderStageStats() {
      const body = document.getElementById("stage-stats");
      const stats = data.stage_stats || [];
      if (!stats.length) {
        body.innerHTML = `<tr><td colspan="5" class="empty">No span data</td></tr>`;
        return;
      }
      body.innerHTML = stats.map((item) => `
        <tr>
          <td title="${esc(item.stage)}">${esc(item.stage)}</td>
          <td>${item.count}</td>
          <td>${fmtSeconds(item.avg_s)}</td>
          <td>${fmtSeconds(item.p95_s)}</td>
          <td>${fmtSeconds(item.max_s)}</td>
        </tr>
      `).join("");
    }

    function renderLegend() {
      const legend = document.getElementById("legend");
      const entries = Object.entries(data.stage_colors || {});
      legend.innerHTML = entries.map(([stage, color]) => `
        <span class="legend-item" title="${esc(stage)}"><span class="swatch" style="background:${color}"></span><span>${esc(stage)}</span></span>
      `).join("");
    }

    function renderTimeline() {
      const timeline = document.getElementById("timeline");
      const rows = data.rows || [];
      if (!rows.length) {
        timeline.innerHTML = `<div class="empty">No task spans</div>`;
        return;
      }
      timeline.innerHTML = rows.map((row) => `
        <div class="trace-row" style="height:${row.row_height_px}px">
          <div class="trace-label">
            <div class="trace-id mono" title="${esc(row.trace_id)}">${esc(row.trace_id)}</div>
            <div class="subtle">duration ${fmtSeconds(row.duration_s)}</div>
          </div>
          <div class="trace-lane">
            ${row.spans.map((span) => `
              <div
                class="span-block ${esc(span.outcome)}"
                title="${esc(span.display_stage)} ${fmtSeconds(span.duration_s)} ${esc(span.span)}"
                style="left:${span.left_pct}%;width:${span.width_pct}%;top:${span.top_px}px;background:${span.color}">
                ${esc(span.display_stage)}
              </div>
            `).join("")}
          </div>
        </div>
      `).join("");
    }

    renderMetrics();
    renderStageStats();
    renderLegend();
    renderTimeline();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
