from __future__ import annotations

import argparse
import dataclasses
import http.server
import json
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from xtuner.v1.rl.trace import (
    TRACE_JSONL_BASENAME,
    TRACE_VIEWER_SCOPE_LATEST_PRODUCE_BATCH,
    TraceEvent,
    TraceViewerScope,
)
from xtuner.tools.producer_trace_analysis import (
    build_open_span_summaries,
    build_viewer_rows,
    display_trace_stage,
    events_to_timelines,
    filter_trace_events_by_scope,
    load_trace_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect producer trace JSONL shards.")
    parser.add_argument("trace_dir", type=Path, help="Directory containing producer_trace_*.jsonl files.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output HTML file. Defaults to <trace_dir>/producer_trace_viewer.html.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Serve a live viewer that polls the trace directory instead of writing a static HTML snapshot.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host used by --serve.")
    parser.add_argument("--port", type=int, default=0, help="Port used by --serve. Defaults to an available port.")
    parser.add_argument(
        "--refresh-interval",
        type=float,
        default=1.0,
        help="Live viewer refresh interval in seconds.",
    )
    parser.add_argument(
        "--scope",
        choices=("latest-produce-batch", "all"),
        default=TRACE_VIEWER_SCOPE_LATEST_PRODUCE_BATCH,
        help="Trace scope shown by the viewer. Defaults to the latest produce batch.",
    )
    return parser.parse_args()


@dataclasses.dataclass
class ProducerTraceViewerHandle:
    server: http.server.ThreadingHTTPServer
    thread: threading.Thread
    url: str
    closed: bool = False

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


class TraceJsonlIndex:
    def __init__(self, trace_dir: Path) -> None:
        self.trace_dir = trace_dir.absolute()
        self._offsets: dict[Path, int] = {}
        self._events: list[TraceEvent] = []
        self._events_by_batch_id: defaultdict[str, list[TraceEvent]] = defaultdict(list)
        self._latest_batch_id: str | None = None
        self._latest_batch_sort_key: tuple[int, int, int, float] | None = None
        self._lock = threading.RLock()

    def build_payload(
        self,
        *,
        scope: TraceViewerScope = TRACE_VIEWER_SCOPE_LATEST_PRODUCE_BATCH,
    ) -> dict[str, Any]:
        with self._lock:
            self.refresh()
            events = self._events_for_scope(scope)
        return build_viewer_payload_from_events(events, trace_source=str(self.trace_dir), scope=scope)

    def refresh(self) -> None:
        with self._lock:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            for file in sorted(self.trace_dir.glob(f"{TRACE_JSONL_BASENAME}_*.jsonl")):
                self._tail_file(file)

    def _tail_file(self, file: Path) -> None:
        offset = self._offsets.get(file, 0)
        try:
            size = file.stat().st_size
        except OSError:
            return
        if size < offset:
            offset = 0
        try:
            with file.open("r", encoding="utf-8") as f:
                f.seek(offset)
                while True:
                    line_start = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    if not line.endswith("\n"):
                        offset = line_start
                        break
                    offset = f.tell()
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self._add_event(TraceEvent.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                        continue
        except OSError:
            return
        self._offsets[file] = offset

    def _add_event(self, event: TraceEvent) -> None:
        self._events.append(event)
        if event.produce_batch_id is None:
            return
        self._events_by_batch_id[event.produce_batch_id].append(event)
        sort_key = self._batch_sort_key(event)
        if self._latest_batch_sort_key is None or sort_key > self._latest_batch_sort_key:
            self._latest_batch_id = event.produce_batch_id
            self._latest_batch_sort_key = sort_key

    def _events_for_scope(self, scope: TraceViewerScope) -> list[TraceEvent]:
        if scope == "all":
            return list(self._events)
        if self._latest_batch_id is not None:
            return list(self._events_by_batch_id[self._latest_batch_id])
        return filter_trace_events_by_scope(self._events, scope)

    @staticmethod
    def _batch_sort_key(event: TraceEvent) -> tuple[int, int, int, float]:
        if event.train_step is None:
            return (-1, -1, -1, event.timestamp_s)
        model_step = -1 if event.model_step is None else event.model_step
        producer_future_step = -1 if event.producer_future_step is None else event.producer_future_step
        return (event.train_step, model_step, producer_future_step, event.timestamp_s)


def build_viewer_payload(
    trace_dir: Path,
    *,
    scope: TraceViewerScope = TRACE_VIEWER_SCOPE_LATEST_PRODUCE_BATCH,
) -> dict[str, Any]:
    return build_viewer_payload_from_events(load_trace_jsonl(trace_dir), trace_source=str(trace_dir), scope=scope)


def build_viewer_payload_from_events(
    events: list[TraceEvent],
    *,
    trace_source: str,
    scope: TraceViewerScope = TRACE_VIEWER_SCOPE_LATEST_PRODUCE_BATCH,
) -> dict[str, Any]:
    raw_event_count = len(events)
    events = filter_trace_events_by_scope(events, scope)
    timelines = events_to_timelines(events)
    rows = build_viewer_rows(timelines)
    summaries = build_open_span_summaries(timelines)
    latest_stage_counts = Counter(row.latest_stage for row in rows)

    return {
        "generated_at_s": time.time(),
        "trace_dir": trace_source,
        "scope": scope,
        "event_count": len(events),
        "raw_event_count": raw_event_count,
        "trace_count": len(timelines),
        "open_trace_count": sum(1 for row in rows if row.open_span is not None),
        "task_summary": build_task_summary(rows),
        "latest_stage_counts": dict(latest_stage_counts.most_common()),
        "open_span_summaries": [dataclasses.asdict(summary) for summary in summaries],
        "rows": [dataclasses.asdict(row) for row in rows],
        "timelines": {
            trace_id: [event.to_dict() for event in trace_events] for trace_id, trace_events in timelines.items()
        },
    }


def build_task_summary(rows: list[Any]) -> dict[str, Any]:
    running_tasks = sum(1 for row in rows if row.open_span is not None)
    current_stage_counts = Counter(display_trace_stage(row.open_span) for row in rows if row.open_span is not None)
    return {
        "total_tasks": len(rows),
        "running_tasks": running_tasks,
        "completed_tasks": max(0, len(rows) - running_tasks),
        "current_stage_counts": dict(current_stage_counts.most_common()),
    }


def write_viewer_html(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_viewer_html(payload), encoding="utf-8")


def render_viewer_html(
    payload: dict[str, Any],
    *,
    live: bool = False,
    api_url: str = "/api/trace",
    refresh_interval_s: float = 1.0,
) -> str:
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    api_url_json = json.dumps(api_url, ensure_ascii=False).replace("</", "<\\/")
    refresh_interval_ms = max(100, int(refresh_interval_s * 1000))
    return (
        _HTML_TEMPLATE.replace("__TRACE_DATA__", data_json)
        .replace("__LIVE_MODE__", "true" if live else "false")
        .replace("__TRACE_API_URL__", api_url_json)
        .replace("__REFRESH_INTERVAL_MS__", str(refresh_interval_ms))
    )


def serve_trace_viewer(
    trace_dir: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    refresh_interval_s: float = 1.0,
    scope: TraceViewerScope = TRACE_VIEWER_SCOPE_LATEST_PRODUCE_BATCH,
) -> None:
    handle = start_trace_viewer(
        trace_dir,
        host=host,
        port=port,
        refresh_interval_s=refresh_interval_s,
        scope=scope,
    )
    print(f"Serving Producer Trace Viewer on {handle.url}", flush=True)
    print(f"Trace dir: {trace_dir.absolute()}", flush=True)
    try:
        handle.thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        handle.close()


def start_trace_viewer(
    trace_dir: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    refresh_interval_s: float = 1.0,
    scope: TraceViewerScope = TRACE_VIEWER_SCOPE_LATEST_PRODUCE_BATCH,
) -> ProducerTraceViewerHandle:
    trace_dir = trace_dir.absolute()
    trace_index = TraceJsonlIndex(trace_dir)

    class TraceViewerHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path in {"/", "/index.html"}:
                payload = trace_index.build_payload(scope=scope)
                html = render_viewer_html(
                    payload,
                    live=True,
                    api_url="/api/trace",
                    refresh_interval_s=refresh_interval_s,
                )
                self._send_bytes(html.encode("utf-8"), "text/html; charset=utf-8")
                return

            if path == "/api/trace":
                payload = trace_index.build_payload(scope=scope)
                body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                self._send_bytes(body, "application/json; charset=utf-8")
                return

            self.send_error(404)

        def _send_bytes(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = http.server.ThreadingHTTPServer((host, port), TraceViewerHandler)
    server_host, server_port = server.server_address
    display_host = "127.0.0.1" if server_host in {"", "0.0.0.0"} else server_host
    thread = threading.Thread(target=server.serve_forever, name="ProducerTraceViewer", daemon=True)
    thread.start()
    return ProducerTraceViewerHandle(
        server=server,
        thread=thread,
        url=f"http://{display_host}:{server_port}",
    )


def main() -> None:
    args = parse_args()
    trace_dir = args.trace_dir
    if args.serve:
        serve_trace_viewer(
            trace_dir,
            host=args.host,
            port=args.port,
            refresh_interval_s=args.refresh_interval,
            scope=args.scope,
        )
        return

    output_path = args.output or trace_dir / "producer_trace_viewer.html"
    payload = build_viewer_payload(trace_dir, scope=args.scope)
    write_viewer_html(payload, output_path)
    print(output_path)


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Producer Trace Viewer</title>
  <style>
    :root {
      --bg: #f7f8fa;
      --panel: #ffffff;
      --line: #d9dde4;
      --text: #1f2937;
      --muted: #637083;
      --accent: #1b6ac9;
      --danger: #b42318;
      --ok: #217a3d;
      --warn-bg: #fff3e6;
      --danger-bg: #fff0ef;
      --ok-bg: #ebf8ef;
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
      align-items: flex-end;
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
    .status-line {
      text-align: right;
      max-width: 360px;
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(560px, 1.7fr) minmax(360px, 1fr);
      gap: 16px;
      padding: 16px 24px 24px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      min-width: 0;
    }
    .panel-body { padding: 12px; }
    .metrics {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      padding: 12px 24px 0;
      align-items: stretch;
    }
    .metric {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      min-height: 64px;
      flex: 0 0 150px;
    }
    .metric strong { display: block; font-size: 20px; line-height: 1.2; }
    .stage-metric {
      flex: 1 1 420px;
      min-width: 320px;
    }
    .stage-list {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 7px;
    }
    .stage-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      max-width: 240px;
      min-height: 24px;
      border: 1px solid #cfd8e3;
      border-radius: 999px;
      padding: 2px 8px;
      background: #f8fafc;
      color: #344054;
    }
    .stage-pill strong {
      display: inline;
      font-size: 13px;
      line-height: 1;
    }
    .stage-pill span {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .toolbar {
      display: flex;
      gap: 8px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      align-items: center;
    }
    input, select {
      height: 30px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 8px;
      background: #fff;
      color: var(--text);
      min-width: 0;
    }
    input { flex: 1; }
    table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
    }
    th, td {
      padding: 7px 8px;
      border-bottom: 1px solid #edf0f4;
      text-align: left;
      vertical-align: top;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .wrap-cell {
      white-space: normal;
      overflow-wrap: anywhere;
      text-overflow: clip;
    }
    th {
      color: var(--muted);
      font-weight: 600;
      background: #fafbfc;
      position: sticky;
      top: 0;
      z-index: 1;
    }
    tbody tr { cursor: pointer; }
    tbody tr:hover { background: #f3f6fa; }
    tbody tr.selected { background: #eaf2ff; }
    .table-wrap { max-height: 420px; overflow: auto; }
    .tag {
      display: inline-flex;
      align-items: center;
      max-width: 100%;
      height: 22px;
      padding: 0 7px;
      border-radius: 999px;
      background: #edf2f7;
      color: #344054;
      font-size: 12px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .tag.open { background: var(--danger-bg); color: var(--danger); }
    .tag.done { background: var(--ok-bg); color: var(--ok); }
    .split {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-bottom: 16px;
    }
    .timeline {
      display: grid;
      gap: 8px;
      max-height: 520px;
      overflow: auto;
      padding-right: 4px;
    }
    .event {
      display: grid;
      grid-template-columns: 112px minmax(0, 1fr) 88px;
      gap: 10px;
      align-items: start;
      border-bottom: 1px solid #edf0f4;
      padding: 7px 0;
    }
    .event-stage { font-weight: 600; overflow-wrap: anywhere; }
    .timeline-heading {
      display: grid;
      gap: 2px;
      margin-bottom: 8px;
    }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .empty {
      color: var(--muted);
      padding: 18px 4px;
      text-align: center;
    }
    @media (max-width: 980px) {
      .layout, .split { grid-template-columns: 1fr; }
      .metric, .stage-metric { flex-basis: 100%; min-width: 0; }
      header { align-items: flex-start; flex-direction: column; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Producer Trace Viewer</h1>
      <div class="subtle mono" id="trace-source"></div>
    </div>
    <div class="subtle status-line">
      <div id="generated-at"></div>
      <div id="live-status"></div>
    </div>
  </header>

  <section class="metrics">
    <div class="metric"><span class="subtle">Total tasks</span><strong id="metric-total-tasks">0</strong></div>
    <div class="metric"><span class="subtle">Running tasks</span><strong id="metric-running-tasks">0</strong></div>
    <div class="metric"><span class="subtle">Completed tasks</span><strong id="metric-completed-tasks">0</strong></div>
    <div class="metric stage-metric">
      <span class="subtle">Current Trace Function Stages</span>
      <div class="stage-list" id="current-stage-counts"></div>
    </div>
  </section>

  <main class="layout">
    <section>
      <div class="split">
        <div class="panel">
          <div class="panel-body">
            <h2>Suspect Open Spans</h2>
            <div class="table-wrap">
              <table>
                <thead><tr><th>Span</th><th style="width:70px">Open</th><th style="width:92px">Oldest</th><th style="width:150px">Oldest Trace</th></tr></thead>
                <tbody id="open-spans"></tbody>
              </table>
            </div>
          </div>
        </div>
        <div class="panel">
          <div class="panel-body">
            <h2>Latest Stage Distribution</h2>
            <div class="table-wrap">
              <table>
                <thead><tr><th>Stage</th><th style="width:80px">Tasks</th></tr></thead>
                <tbody id="stage-counts"></tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      <div class="panel">
        <div class="toolbar">
          <input id="search" placeholder="Filter trace id, task name, latest stage, open span">
          <select id="open-filter">
            <option value="all">All tasks</option>
            <option value="open">Open spans only</option>
            <option value="closed">No open span</option>
          </select>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th style="width:150px">Trace</th>
                <th style="width:82px">Task</th>
                <th>Latest Stage</th>
                <th style="width:170px">Open Span</th>
                <th style="width:82px">Open Age</th>
              </tr>
            </thead>
            <tbody id="task-rows"></tbody>
          </table>
        </div>
      </div>
    </section>

    <section class="panel">
      <div class="panel-body">
        <div class="timeline-heading">
          <h2>Task Timeline</h2>
          <div class="subtle mono" id="timeline-title"></div>
        </div>
        <div class="timeline" id="timeline"></div>
      </div>
    </section>
  </main>

  <script id="trace-data" type="application/json">__TRACE_DATA__</script>
  <script>
    const LIVE_MODE = __LIVE_MODE__;
    const TRACE_API_URL = __TRACE_API_URL__;
    const REFRESH_INTERVAL_MS = __REFRESH_INTERVAL_MS__;
    let data = JSON.parse(document.getElementById("trace-data").textContent);
    let rows = data.rows || [];
    let timelines = data.timelines || {};
    let selectedTraceId = rows[0]?.trace_id || null;

    const fmtSeconds = (seconds) => {
      if (seconds === null || seconds === undefined) return "";
      if (seconds < 60) return `${seconds.toFixed(1)}s`;
      if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`;
      return `${(seconds / 3600).toFixed(1)}h`;
    };
    const fmtTime = (timestamp) => new Date(timestamp * 1000).toLocaleString();
    const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (ch) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    }[ch]));

    function renderMetrics() {
      document.getElementById("trace-source").textContent = data.trace_dir || "";
      document.getElementById("generated-at").textContent = `Generated ${fmtTime(data.generated_at_s)}`;
      const summary = data.task_summary || {};
      const totalTasks = summary.total_tasks ?? data.trace_count ?? rows.length ?? 0;
      const runningTasks = summary.running_tasks ?? data.open_trace_count ?? 0;
      const completedTasks = summary.completed_tasks ?? Math.max(0, totalTasks - runningTasks);
      document.getElementById("metric-total-tasks").textContent = totalTasks;
      document.getElementById("metric-running-tasks").textContent = runningTasks;
      document.getElementById("metric-completed-tasks").textContent = completedTasks;

      const stageCounts = Object.entries(summary.current_stage_counts || {});
      const stageCountsEl = document.getElementById("current-stage-counts");
      if (!stageCounts.length) {
        stageCountsEl.innerHTML = `<span class="subtle">No running tasks</span>`;
        return;
      }
      stageCountsEl.innerHTML = stageCounts.map(([stage, count]) => `
        <span class="stage-pill" title="${esc(stage)}"><strong>${count}</strong><span>${esc(stage)}</span></span>
      `).join("");
    }

    function renderOpenSpans() {
      const body = document.getElementById("open-spans");
      const summaries = data.open_span_summaries || [];
      if (!summaries.length) {
        body.innerHTML = `<tr><td colspan="4" class="empty">No open spans</td></tr>`;
        return;
      }
      body.innerHTML = summaries.map((summary) => `
        <tr data-trace-id="${esc(summary.oldest_trace_id)}">
          <td class="wrap-cell" title="${esc(summary.span)}">${esc(summary.span)}</td>
          <td>${summary.open_count}</td>
          <td>${fmtSeconds(summary.oldest_age_s)}</td>
          <td class="mono wrap-cell" title="${esc(summary.oldest_trace_id)}">${esc(summary.oldest_trace_id)}</td>
        </tr>
      `).join("");
      [...body.querySelectorAll("tr")].forEach((tr) => {
        tr.addEventListener("click", () => selectTrace(tr.dataset.traceId));
      });
    }

    function renderStageCounts() {
      const body = document.getElementById("stage-counts");
      const entries = Object.entries(data.latest_stage_counts || {});
      if (!entries.length) {
        body.innerHTML = `<tr><td colspan="2" class="empty">No tasks</td></tr>`;
        return;
      }
      body.innerHTML = entries.map(([stage, count]) => `
        <tr><td title="${esc(stage)}">${esc(stage)}</td><td>${count}</td></tr>
      `).join("");
    }

    function renderTaskRows() {
      const query = document.getElementById("search").value.trim().toLowerCase();
      const openFilter = document.getElementById("open-filter").value;
      const visibleRows = rows.filter((row) => {
        const isOpen = row.open_span !== null && row.open_span !== undefined;
        if (openFilter === "open" && !isOpen) return false;
        if (openFilter === "closed" && isOpen) return false;
        if (!query) return true;
        return [row.trace_id, row.task_name, row.latest_stage, row.open_span, row.status]
          .some((value) => String(value ?? "").toLowerCase().includes(query));
      });
      const body = document.getElementById("task-rows");
      if (!visibleRows.length) {
        body.innerHTML = `<tr><td colspan="5" class="empty">No matching tasks</td></tr>`;
        return;
      }
      body.innerHTML = visibleRows.map((row) => `
        <tr class="${row.trace_id === selectedTraceId ? "selected" : ""}" data-trace-id="${esc(row.trace_id)}">
          <td class="mono" title="${esc(row.trace_id)}">${esc(row.trace_id)}</td>
          <td title="${esc(row.task_name)}">${esc(row.task_name)}</td>
          <td class="wrap-cell" title="${esc(row.latest_stage)}">${esc(row.latest_stage)}</td>
          <td>${row.open_span ? `<span class="tag open" title="${esc(row.open_span)}">${esc(row.open_span)}</span>` : `<span class="tag done">closed</span>`}</td>
          <td>${fmtSeconds(row.open_age_s)}</td>
        </tr>
      `).join("");
      [...body.querySelectorAll("tr")].forEach((tr) => {
        tr.addEventListener("click", () => selectTrace(tr.dataset.traceId));
      });
    }

    function selectTrace(traceId) {
      selectedTraceId = traceId;
      renderTaskRows();
      renderTimeline();
    }

    function renderTimeline() {
      const timeline = timelines[selectedTraceId] || [];
      const title = document.getElementById("timeline-title");
      const body = document.getElementById("timeline");
      title.textContent = selectedTraceId || "";
      if (!timeline.length) {
        body.innerHTML = `<div class="empty">Select a task to inspect its events</div>`;
        return;
      }
      body.innerHTML = timeline.map((event) => `
        <div class="event">
          <div class="subtle">${fmtTime(event.timestamp_s)}</div>
          <div>
            <div class="event-stage">${esc(event.stage)}</div>
            <div class="subtle">status=${esc(event.status)} session=${esc(event.session_uid)} train_step=${esc(event.train_step)} model_step=${esc(event.model_step)}</div>
            ${event.error_msg ? `<div class="tag open" title="${esc(event.error_msg)}">${esc(event.error_msg)}</div>` : ""}
          </div>
          <div>${event.elapsed_s === null || event.elapsed_s === undefined ? "" : fmtSeconds(event.elapsed_s)}</div>
        </div>
      `).join("");
    }

    document.getElementById("search").addEventListener("input", renderTaskRows);
    document.getElementById("open-filter").addEventListener("change", renderTaskRows);

    function renderAll() {
      renderMetrics();
      renderOpenSpans();
      renderStageCounts();
      renderTaskRows();
      renderTimeline();
    }

    function setTraceData(nextData) {
      data = nextData || {};
      rows = data.rows || [];
      timelines = data.timelines || {};
      if (!selectedTraceId || !timelines[selectedTraceId]) {
        selectedTraceId = rows[0]?.trace_id || null;
      }
      renderAll();
    }

    async function refreshData() {
      const status = document.getElementById("live-status");
      try {
        const response = await fetch(TRACE_API_URL, { cache: "no-store" });
        if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
        const nextData = await response.json();
        setTraceData(nextData);
        status.textContent = `Live refresh ${fmtTime(nextData.generated_at_s)}`;
      } catch (error) {
        status.textContent = `Live refresh failed: ${error.message}`;
      }
    }

    renderAll();
    if (LIVE_MODE) {
      document.getElementById("live-status").textContent =
        `Live refresh every ${(REFRESH_INTERVAL_MS / 1000).toFixed(1)}s`;
      setInterval(refreshData, REFRESH_INTERVAL_MS);
    }
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
