from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def render_rollout_trace_html(
    payload: dict[str, Any],
    *,
    live: bool = False,
    api_url: str = "/api/trace",
    refresh_interval_s: float = 2.0,
) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return (
        _HTML_TEMPLATE.replace("__TRACE_DATA__", data)
        .replace("__LIVE_MODE__", json.dumps(live))
        .replace("__TRACE_API_URL__", json.dumps(api_url))
        .replace("__REFRESH_INTERVAL_MS__", str(int(refresh_interval_s * 1000)))
    )


def write_rollout_trace_html(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_rollout_trace_html(payload), encoding="utf-8")


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>XTuner Rollout Trace Viewer</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fb;
      --panel: #ffffff;
      --line: #d6dde8;
      --text: #171f2f;
      --muted: #667085;
      --accent: #0f766e;
      --bad: #b42318;
      --warn: #a15c07;
      --ok: #166534;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    header {
      padding: 18px 24px 14px;
      border-bottom: 1px solid var(--line);
      background: #fff;
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: end;
    }
    h1 { margin: 0; font-size: 22px; font-weight: 720; letter-spacing: 0; }
    h2 { margin: 0 0 10px; font-size: 15px; }
    main { padding: 16px 24px 28px; display: grid; gap: 14px; }
    .muted { color: var(--muted); font-size: 13px; }
    .metric-grid, .panel-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; }
    .metric, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-width: 0;
    }
    .metric strong { display: block; margin-top: 5px; font-size: 22px; overflow-wrap: anywhere; }
    .toolbar {
      display: grid;
      grid-template-columns: minmax(280px, 1fr) 150px;
      gap: 10px;
      margin-bottom: 10px;
    }
    .step-toolbar {
      display: grid;
      grid-template-columns: minmax(160px, 220px);
      gap: 10px;
    }
    input, select, button {
      height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 10px;
      background: #fff;
      color: var(--text);
      font: inherit;
    }
    button { cursor: pointer; }
    table { width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
    th, td { border-bottom: 1px solid var(--line); padding: 9px 10px; text-align: left; vertical-align: top; font-size: 13px; }
    th { background: #eef3f7; font-size: 12px; color: #344054; text-transform: uppercase; }
    tr:last-child td { border-bottom: 0; }
    details { margin-top: 8px; }
    summary { cursor: pointer; color: var(--accent); }
	    .tag { display: inline-block; border-radius: 999px; padding: 2px 8px; font-size: 12px; border: 1px solid var(--line); background: #f8fafc; }
	    .tag.completed, .tag.ok, .tag.kept { border-color: #86efac; background: #f0fdf4; color: var(--ok); }
		    .tag.failed, .tag.error, .tag.exception, .tag.timeout, .tag.timed_out, .tag.aborted { border-color: #fecaca; background: #fef2f2; color: var(--bad); }
		    .tag.running, .tag.materialize, .tag.session_server, .tag.backend, .tag.judger, .tag.env_init, .tag.context_build, .tag.agent_run, .tag.llm_generate, .tag.tool_call { border-color: #bae6fd; background: #f0f9ff; color: #075985; }
		    .tag.filtered, .tag.stale, .tag.missing, .tag.extra, .tag.mixed { border-color: #fed7aa; background: #fff7ed; color: var(--warn); }
		    .stage-label { font-weight: 720; }
		    .raw-spans { margin-top: 5px; display: grid; gap: 3px; color: var(--muted); font-size: 12px; line-height: 1.35; }
		    .raw-span-item { overflow-wrap: anywhere; }
		    .raw-span-details { margin-top: 3px; }
		    .path-summary { display: grid; gap: 6px; min-width: 360px; }
	    .path-current { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
	    .path-chain { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; line-height: 1.8; }
	    .path-node {
	      display: inline-flex;
	      align-items: center;
	      gap: 4px;
	      max-width: 260px;
	      border: 1px solid var(--line);
	      border-radius: 6px;
	      padding: 1px 6px;
	      background: #f8fafc;
	      color: #344054;
	      white-space: nowrap;
	    }
	    .path-node-name { overflow: hidden; text-overflow: ellipsis; }
	    .path-node.running { border-color: #7dd3fc; background: #e0f2fe; color: #075985; font-weight: 650; }
	    .path-node.active { border-color: #bae6fd; background: #f0f9ff; color: #075985; }
	    .path-node.done { border-color: #bbf7d0; background: #f0fdf4; color: #166534; }
	    .path-node.error { border-color: #fecaca; background: #fef2f2; color: var(--bad); }
	    .path-node.inferred { border-style: dashed; color: var(--muted); }
	    .path-arrow { color: #98a2b3; }
	    .path-meta { color: var(--muted); font-size: 11px; }
	    .span-list { margin: 8px 0 0; padding-left: 18px; color: #344054; }
    .scroll { overflow-x: auto; }
    a { color: var(--accent); text-decoration: none; }
    a:hover { text-decoration: underline; }
    @media (max-width: 820px) {
      header { display: block; }
      main { padding: 12px; }
      .metric-grid, .panel-grid, .toolbar { grid-template-columns: 1fr; }
      th, td { font-size: 12px; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1 id="title">XTuner Rollout Trace Viewer</h1>
      <div class="muted" id="source"></div>
    </div>
    <div class="muted" id="mode"></div>
  </header>
  <main>
    <section class="panel step-toolbar" id="viewerControls">
      <select id="step"><option>All steps</option></select>
    </section>

    <section class="metric-grid" id="metrics"></section>

    <section class="panel" id="stageOccupancy">
      <h2>Stage Occupancy</h2>
      <div class="scroll">
        <table>
          <thead><tr><th>Stage</th><th>Samples</th><th>Groups</th></tr></thead>
          <tbody id="stageOccupancyRows"></tbody>
        </table>
      </div>
    </section>

    <section class="panel" id="stageDurations">
      <h2>Stage Durations</h2>
      <div class="scroll">
        <table>
          <thead><tr><th>Stage</th><th>Count</th><th>Avg s</th><th>P50 s</th><th>P95 s</th><th>Max s</th><th>Errors</th><th>Top Error</th></tr></thead>
          <tbody id="stageDurationRows"></tbody>
        </table>
      </div>
    </section>

    <section class="panel" id="samples">
      <h2>Samples</h2>
      <section class="toolbar">
        <input id="search" type="search" placeholder="rollout:1001 step:5 stack:session error:timeout" title="Search prefixes: rollout:, step:, task:, trace:, status:, stage:, span:, stack:, error:, session:. Plain text searches all sample fields.">
        <select id="status"><option>All status</option></select>
      </section>
      <div class="scroll">
        <table>
          <thead>
            <tr>
	              <th>Sample</th>
	              <th>Status</th>
	              <th>Group</th>
	              <th>Step</th>
	              <th>Reward</th>
	              <th>Path / Current</th>
              <th>Jaeger</th>
            </tr>
          </thead>
          <tbody id="sampleRows"></tbody>
        </table>
      </div>
    </section>

  </main>
  <script>
    const initialData = __TRACE_DATA__;
    const liveMode = __LIVE_MODE__;
    const traceApiUrl = __TRACE_API_URL__;
    const refreshIntervalMs = __REFRESH_INTERVAL_MS__;
    let data = initialData;
    let stepRequestId = 0;
    let pendingStepRequest = null;
    const stepPayloadCache = new Map();
    const filters = {search: "", step: String(initialData.selected_train_step ?? "all"), status: "all"};
    const els = {
      title: document.getElementById("title"),
      source: document.getElementById("source"),
      mode: document.getElementById("mode"),
      metrics: document.getElementById("metrics"),
      stageOccupancyRows: document.getElementById("stageOccupancyRows"),
      stageDurationRows: document.getElementById("stageDurationRows"),
      sampleRows: document.getElementById("sampleRows"),
      search: document.getElementById("search"),
      step: document.getElementById("step"),
      status: document.getElementById("status"),
    };
    function esc(value) {
      return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#39;");
    }
    function tagClass(value) {
      return `tag ${String(value || "unknown").toLowerCase().replaceAll(".", "_")}`;
    }
    function compareValues(left, right) {
      if (left == null && right == null) return 0;
      if (left == null) return 1;
      if (right == null) return -1;
      const leftNumber = Number(left);
      const rightNumber = Number(right);
      if (!Number.isNaN(leftNumber) && !Number.isNaN(rightNumber)) return leftNumber - rightNumber;
      return String(left).localeCompare(String(right));
    }
    function sortValues(values) {
      return Array.from(new Set(values.filter((value) => value != null))).sort(compareValues);
    }
    function cacheStepPayload(payload) {
      if (!payload) return;
      if (payload.selected_train_step !== undefined && payload.selected_train_step !== null) {
        stepPayloadCache.set(String(payload.selected_train_step), payload);
      }
    }
    function selectOptionsSignature(optionValues) {
      return optionValues.join("\\u0001");
    }
    function setSelectOptions(select, values, allLabel, labelForValue, currentValue) {
      const optionValues = ["all", ...values.map((value) => String(value))];
      const signature = selectOptionsSignature(optionValues);
      const nextValue = optionValues.includes(String(currentValue)) ? String(currentValue) : "all";
      if (select.dataset.optionsSignature !== signature) {
        select.innerHTML = optionValues.map((value) => {
          const label = value === "all" ? allLabel : labelForValue(value);
          return `<option value="${esc(value)}">${esc(label)}</option>`;
        }).join("");
        select.dataset.optionsSignature = signature;
      }
      if (select.value !== nextValue) select.value = nextValue;
      return nextValue;
    }
    function parseSearchTerms(query) {
      return query.trim().toLowerCase().split(/\s+/).filter(Boolean).map((term) => {
        const separator = term.indexOf(":");
        if (separator > 0) return {key: term.slice(0, separator), value: term.slice(separator + 1)};
        return {key: "", value: term};
      }).filter((term) => term.value);
    }
		    function sampleSearchFields(sample) {
		      const spans = sample.spans || [];
		      const displayPathText = (sample.display_path || []).map((node) => `${node.name ?? ""} ${node.stage ?? ""} ${node.status ?? ""} ${node.source ?? ""}`).join(" ");
		      const currentStageText = sample.current_stage ? `${sample.current_stage.name ?? ""} ${sample.current_stage.stage ?? ""} ${sample.current_stage.status ?? ""}` : "";
		      const spanText = spans.map((span) => `${span.name ?? ""} ${span.stage ?? ""} ${span.parent_span_id ?? ""} ${JSON.stringify(span.attributes || {})}`).join(" ");
      const errorText = spans.map((span) => {
        const attrs = span.attributes || {};
        return Object.entries(attrs).filter(([key]) => key.startsWith("error.") || key.startsWith("exception.") || key === "error").map(([key, value]) => `${key} ${value}`).join(" ");
      }).join(" ");
      const sessionText = spans.map((span) => {
        const attrs = span.attributes || {};
        const attrText = Object.entries(attrs).filter(([key]) => key.startsWith("session.")).map(([key, value]) => `${key} ${value}`).join(" ");
        return `${span.name ?? ""} ${attrText}`;
      }).join(" ");
      const fields = {
        rollout: sample.rollout_id,
        rollout_id: sample.rollout_id,
        step: sample.producer_future_step,
        producer_future_step: sample.producer_future_step,
        task: sample.task_name,
        trace: sample.trace_id,
        trace_id: sample.trace_id,
        status: sample.status,
        stage: sample.stage,
	        span: `${sample.chain ?? ""} ${displayPathText} ${spanText}`,
	        stack: `${sample.chain ?? ""} ${displayPathText} ${currentStageText} ${spanText}`,
	        current: currentStageText,
        error: errorText,
        reward: `${sample.reward_score ?? ""} ${sample.reward_pass ?? ""}`,
        filter: `${sample.filter_decision ?? ""} ${sample.filter_reason ?? ""} ${sample.train_included ?? ""}`,
        drop: `${sample.drop_reason ?? ""}`,
        session: sessionText,
      };
      fields.all = Object.values(fields).map((value) => String(value ?? "")).join(" ").toLowerCase();
      return fields;
    }
    function matchesSearch(sample, terms) {
      const fields = sampleSearchFields(sample);
      return terms.every((term) => {
        if (!term.key) return fields.all.includes(term.value);
        const fieldValue = fields[term.key];
        if (fieldValue === undefined) return fields.all.includes(term.value);
        return String(fieldValue ?? "").toLowerCase().includes(term.value);
      });
    }
    function filteredSamples() {
	      const searchTerms = parseSearchTerms(filters.search);
	      return (data.samples || []).filter((sample) => {
	        if (filters.step !== "all" && String(sample.producer_future_step) !== filters.step) return false;
	        if (filters.status !== "all" && String(sample.status) !== filters.status) return false;
	        return !searchTerms.length || matchesSearch(sample, searchTerms);
	      });
	    }
	    function formatDurationS(value) {
	      const number = Number(value ?? 0);
	      if (!Number.isFinite(number)) return "-";
	      return number.toFixed(3).replace(/\.000$/, "").replace(/(\.\d*[1-9])0+$/, "$1");
	    }
	    function formatDurationMs(value) {
	      const number = Number(value ?? 0);
	      if (!Number.isFinite(number)) return "-";
	      if (number >= 1000) return `${formatDurationS(number / 1000)}s`;
	      return `${number.toFixed(1).replace(/\.0$/, "")}ms`;
	    }
    function formatCounts(counts) {
      return Object.entries(counts || {}).sort(([left], [right]) => left.localeCompare(right)).map(([key, value]) => `${key}: ${value}`).join(" | ") || "-";
    }
    function countBy(samples, field) {
      return samples.reduce((counts, sample) => {
        const key = String(sample[field] ?? "unknown");
        counts[key] = (counts[key] || 0) + 1;
        return counts;
      }, {});
    }
    function renderOptions() {
      const statuses = Object.keys(data.status_counts || {}).sort();
      const steps = sortValues(data.available_train_steps || (data.samples || []).map((sample) => sample.producer_future_step));
      filters.step = setSelectOptions(
        els.step,
        steps,
        "All steps",
        (value) => `step ${value}`,
        data.selected_train_step ?? filters.step,
      );
      els.step.disabled = !liveMode;
      filters.status = setSelectOptions(els.status, statuses, "All status", (value) => value, filters.status);
    }
    function renderMetrics() {
      const visibleSamples = filteredSamples();
      const metrics = [
        ["Samples", data.sample_count || 0],
	        ["Groups", data.group_count || 0],
	        ["Steps", data.step_count || 0],
	        ["Statuses", formatCounts(data.status_counts || {})],
	        ["Visible", visibleSamples.length],
	      ];
      els.metrics.innerHTML = metrics.map(([label, value]) => `<div class="metric"><div class="muted">${esc(label)}</div><strong>${esc(value)}</strong></div>`).join("");
    }
	    function renderStageOccupancy() {
	      const rows = data.stage_occupancy || [];
	      els.stageOccupancyRows.innerHTML = rows.length ? rows.map((row) => `<tr>
	        <td>${stageCellHtml(row, "sample_count", "samples")}</td>
	        <td>${esc(row.sample_count)}</td>
	        <td>${esc(row.group_count ?? 0)}</td>
	      </tr>`).join("") : `<tr><td colspan="3" class="muted">No stage occupancy data</td></tr>`;
	    }
	    function rawSpanBreakdownHtml(rawSpans, countKey, countLabel) {
	      const rows = rawSpans || [];
	      if (!rows.length) return "";
	      const itemHtml = (item) => {
	        const groupText = item.group_count !== undefined ? `, ${item.group_count} groups` : "";
	        return `<div class="raw-span-item">${esc(item.span || "unknown")} <span>${esc(item[countKey] ?? 0)} ${esc(countLabel)}${esc(groupText)}</span></div>`;
	      };
	      const visible = rows.slice(0, 3).map(itemHtml).join("");
	      const hidden = rows.slice(3).map(itemHtml).join("");
	      const hiddenHtml = hidden ? `<details class="raw-span-details"><summary>+${rows.length - 3} more spans</summary>${hidden}</details>` : "";
	      return `<div class="raw-spans">${visible}${hiddenHtml}</div>`;
	    }
	    function stageCellHtml(row, countKey, countLabel) {
	      return `<div><span class="${tagClass(row.stage)} stage-label">${esc(row.stage)}</span>${rawSpanBreakdownHtml(row.raw_spans, countKey, countLabel)}</div>`;
	    }
    function topErrorText(summary) {
      const error = (summary.top_errors || [])[0];
      if (!error) return "-";
      const type = error.error_type || "error";
      const message = error.message ? `: ${error.message}` : "";
      return `${type}${message} (${error.sample_count || 0})`;
    }
    function topErrorHtml(summary) {
      const errors = summary.top_errors || [];
      if (!errors.length) return "-";
      const items = errors.map((error) => {
        const jaeger = (error.jaeger_urls || [])[0];
        const jaegerText = jaeger ? `<a href="${esc(jaeger)}" target="_blank" rel="noreferrer">Open</a>` : "-";
        const extra = (error.jaeger_urls || []).length > 1 ? ` +${(error.jaeger_urls || []).length - 1}` : "";
        return `<li>${esc(error.error_type || "error")} ${esc(error.message || "")}
          <span class="muted">http=${esc(error.http_status_code ?? "-")} samples=${esc((error.rollout_ids || []).join(", ") || "-")} steps=${esc((error.steps || []).join(", ") || "-")} groups=${esc((error.groups || []).join(", ") || "-")} ${jaegerText}${esc(extra)}</span></li>`;
      }).join("");
      return `<details><summary>${esc(topErrorText(summary))}</summary><ol class="span-list">${items}</ol></details>`;
    }
	    function renderStageDurations() {
	      const rows = data.stage_duration_summaries || [];
	      els.stageDurationRows.innerHTML = rows.length ? rows.map((row) => `<tr>
	        <td>${stageCellHtml(row, "span_count", "spans")}</td>
	        <td>${esc(row.span_count)}</td>
	        <td>${esc(formatDurationS(row.avg_duration_s))}</td>
	        <td>${esc(formatDurationS(row.p50_duration_s))}</td>
	        <td>${esc(formatDurationS(row.p95_duration_s))}</td>
	        <td>${esc(formatDurationS(row.max_duration_s))}</td>
	        <td>${esc(row.error_count || 0)}</td>
	        <td>${topErrorHtml(row)}</td>
	      </tr>`).join("") : `<tr><td colspan="8" class="muted">No duration data</td></tr>`;
	    }
	    function fallbackDisplayPath(sample) {
	      return (sample.spans || []).map((span) => ({
	        name: span.name || "unknown",
	        source: "span",
	        status: String(span.status || "").toUpperCase() === "ERROR" ? "error" : "done",
	        duration_ms: span.duration_ms,
	      }));
	    }
	    function pathNodeMeta(node) {
	      if (node.elapsed_ms !== undefined && node.elapsed_ms !== null) return formatDurationMs(node.elapsed_ms);
	      if (node.duration_ms !== undefined && node.duration_ms !== null) return formatDurationMs(node.duration_ms);
	      if (node.source) return node.source;
	      return "";
	    }
	    function displayPathHtml(sample) {
	      const nodes = (sample.display_path && sample.display_path.length) ? sample.display_path : fallbackDisplayPath(sample);
	      const current = sample.current_stage;
	      const currentHtml = current ? `<div class="path-current"><span class="${tagClass(current.status || "running")}">${esc(current.status || "running")}</span><strong>${esc(current.name || "-")}</strong><span class="muted">${esc(formatDurationMs(current.elapsed_ms))}</span></div>` : "";
	      const chainHtml = nodes.length ? nodes.map((node, index) => {
	        const status = node.status || "inferred";
	        const meta = pathNodeMeta(node);
	        const arrow = index ? `<span class="path-arrow">-&gt;</span>` : "";
	        return `${arrow}<span class="path-node ${esc(status)} ${esc(node.source || "")}" title="${esc(node.source || "")}"><span class="path-node-name">${esc(node.name || "-")}</span>${meta ? `<span class="path-meta">${esc(meta)}</span>` : ""}</span>`;
	      }).join("") : `<span class="muted">-</span>`;
	      return `<div class="path-summary">${currentHtml}<div class="path-chain">${chainHtml}</div></div>`;
	    }
	    function renderSamples() {
	      const rows = filteredSamples();
	      if (!rows.length) {
	        els.sampleRows.innerHTML = `<tr><td colspan="7" class="muted">No samples match the current filters</td></tr>`;
	        return;
	      }
	      els.sampleRows.innerHTML = rows.map((sample) => {
	        const jaeger = sample.jaeger_url ? `<a href="${esc(sample.jaeger_url)}" target="_blank" rel="noreferrer">Open</a>` : `<span class="muted">-</span>`;
	        return `<tr>
	          <td><strong>${esc(sample.task_name || "-")} #${esc(sample.rollout_id)}</strong><div class="muted">${esc(sample.trace_id)}</div></td>
	          <td><span class="${tagClass(sample.status)}">${esc(sample.status || "unknown")}</span></td>
	          <td>${esc(sample.group_id ?? "-")}</td>
          <td>${esc(sample.producer_future_step ?? "-")}</td>
          <td>${esc(sample.reward_score ?? "-")}</td>
	          <td>${displayPathHtml(sample)}</td>
	          <td>${jaeger}</td>
	        </tr>`;
	      }).join("");
	    }
    function renderFilteredViews() {
      renderMetrics();
      renderSamples();
    }
    function render() {
      els.title.textContent = data.title || "XTuner Rollout Trace Viewer";
      const sourceParts = [data.source || "jaeger"];
      if (data.service_name) sourceParts.push(`service ${data.service_name}`);
      if (data.run_id) sourceParts.push(`run ${data.run_id}`);
      if (data.generated_at_s) sourceParts.push(`generated ${new Date(data.generated_at_s * 1000).toLocaleString()}`);
      els.source.textContent = sourceParts.join(" | ");
      els.mode.textContent = liveMode ? "Live viewer" : "Static viewer";
      renderOptions();
      renderMetrics();
      renderStageOccupancy();
      renderStageDurations();
      renderSamples();
    }
    function applyPayload(payload, fallbackStep) {
      data = payload;
      cacheStepPayload(data);
      filters.step = String(data.selected_train_step ?? fallbackStep ?? filters.step);
      render();
    }
    async function fetchStep(step) {
      if (!liveMode) return;
      const requestId = ++stepRequestId;
      const selectedStep = String(step || filters.step || "latest");
      filters.step = selectedStep;
      const url = new URL(traceApiUrl, window.location.href);
      url.searchParams.set("train_step", selectedStep);
      const request = (async () => {
        try {
          const response = await fetch(url.toString(), {cache: "no-store"});
          if (requestId !== stepRequestId) return;
          if (response.ok) applyPayload(await response.json(), selectedStep);
        } catch (error) {
          if (requestId === stepRequestId) console.warn("Failed to refresh XTuner trace viewer", error);
        }
      })();
      pendingStepRequest = request;
      try {
        await request;
      } finally {
        if (pendingStepRequest === request) pendingStepRequest = null;
      }
    }
    async function loadStep(step) {
      if (!liveMode) return;
      const selectedStep = String(step || filters.step || "latest");
      filters.step = selectedStep;
      const cachedPayload = stepPayloadCache.get(selectedStep);
      if (cachedPayload) {
        stepRequestId += 1;
        applyPayload(cachedPayload, selectedStep);
        return;
      }
      await fetchStep(selectedStep);
    }
    async function refresh() {
      if (!liveMode || pendingStepRequest) return;
      await fetchStep(filters.step || data.selected_train_step || "latest");
    }
    els.search.addEventListener("input", (event) => { filters.search = event.target.value; renderFilteredViews(); });
    els.step.addEventListener("change", (event) => { filters.step = event.target.value; loadStep(filters.step); });
    els.status.addEventListener("change", (event) => { filters.status = event.target.value; renderFilteredViews(); });
    cacheStepPayload(data);
    render();
    if (liveMode) window.setInterval(refresh, refreshIntervalMs);
  </script>
</body>
</html>
"""


__all__ = ["render_rollout_trace_html", "write_rollout_trace_html"]
