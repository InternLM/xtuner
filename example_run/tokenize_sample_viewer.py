#!/usr/bin/env python3
"""Read-only browser for XTuner tokenize debug samples.

The page only receives natural-language fields. Token IDs, integer labels,
token pieces, and other tensor payloads are intentionally removed by the
server before a sample is returned to the browser.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


DEFAULT_INPUT = Path("tokenize_debug_samples")


@dataclass(frozen=True)
class SampleRef:
    key: str
    label: str
    path: Path | None = None
    offset: int | None = None
    length: int | None = None


class SampleStore:
    """Index a debug-sample directory, one JSON file, or a JSONL file."""

    def __init__(self, input_path: Path) -> None:
        self.input_path = input_path.expanduser().resolve()
        if not self.input_path.exists():
            raise FileNotFoundError(f"input does not exist: {self.input_path}")

        self._refs: list[SampleRef] = []
        self._by_key: dict[str, SampleRef] = {}
        if self.input_path.is_dir():
            self._index_directory()
            self.kind = "directory"
        elif self.input_path.suffix.lower() == ".jsonl":
            self._index_jsonl()
            self.kind = "jsonl"
        elif self.input_path.suffix.lower() == ".json":
            self._index_json()
            self.kind = "json"
        else:
            raise ValueError("input must be a sample directory, .json, or .jsonl file")

        if not self._refs:
            raise ValueError(f"no samples found in {self.input_path}")

    @staticmethod
    def _key_from_filename(path: Path) -> str:
        match = re.fullmatch(r"sample_(\d+)", path.stem)
        return str(int(match.group(1))) if match else path.stem

    def _add_ref(self, ref: SampleRef) -> None:
        key = ref.key
        if key in self._by_key:
            suffix = 2
            while f"{key}-{suffix}" in self._by_key:
                suffix += 1
            ref = SampleRef(
                key=f"{key}-{suffix}",
                label=f"{ref.label} ({suffix})",
                path=ref.path,
                offset=ref.offset,
                length=ref.length,
            )
        self._refs.append(ref)
        self._by_key[ref.key] = ref

    def _index_directory(self) -> None:
        files: list[Path] = []
        manifest_path = self.input_path / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for filename in manifest.get("files", []):
                if not isinstance(filename, str) or Path(filename).name != filename:
                    continue
                path = (self.input_path / filename).resolve()
                if path.parent == self.input_path and path.is_file():
                    files.append(path)
        if not files:
            files = sorted(self.input_path.glob("sample_*.json"))

        for path in files:
            key = self._key_from_filename(path)
            self._add_ref(SampleRef(key=key, label=f"sample {key}", path=path))

    def _index_json(self) -> None:
        record = json.loads(self.input_path.read_text(encoding="utf-8"))
        index = record.get("source", {}).get("sample_index", 0)
        key = str(index)
        self._add_ref(SampleRef(key=key, label=f"sample {key}", path=self.input_path))

    def _index_jsonl(self) -> None:
        with self.input_path.open("rb") as file:
            line_number = 0
            while True:
                offset = file.tell()
                line = file.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                self._add_ref(
                    SampleRef(
                        key=str(line_number),
                        label=f"line {line_number}",
                        offset=offset,
                        length=len(line),
                    )
                )
                line_number += 1

    def list_samples(self) -> list[dict[str, str]]:
        return [{"key": ref.key, "label": ref.label} for ref in self._refs]

    def read(self, key: str) -> dict[str, Any]:
        try:
            ref = self._by_key[key]
        except KeyError as error:
            raise KeyError(f"unknown sample key: {key}") from error

        if ref.path is not None:
            return json.loads(ref.path.read_text(encoding="utf-8"))

        assert ref.offset is not None and ref.length is not None
        with self.input_path.open("rb") as file:
            file.seek(ref.offset)
            payload = file.read(ref.length)
        return json.loads(payload)

    def public_info(self) -> dict[str, Any]:
        return {
            "input": str(self.input_path),
            "kind": self.kind,
            "sample_count": len(self._refs),
        }


def _natural_record(record: dict[str, Any]) -> dict[str, Any]:
    """Remove all token-ID-shaped data before sending a record to the UI."""

    rendered = record.get("rendered") or {}
    rendered_spans = []
    for span in rendered.get("loss_character_spans") or []:
        rendered_spans.append(
            {
                "start": span.get("start"),
                "end": span.get("end"),
            }
        )

    loss = record.get("loss") or {}
    loss_spans = []
    for position, span in enumerate(loss.get("token_spans") or [], start=1):
        loss_spans.append(
            {
                "number": position,
                "start": span.get("start"),
                "end": span.get("end"),
                "decoded_text": span.get("decoded_text", ""),
            }
        )

    tokenized = record.get("tokenized") or {}
    raw_data = record.get("raw_data") or {}
    return {
        "source": record.get("source") or {},
        "raw_data": {
            "messages": raw_data.get("messages") or [],
            "tools": raw_data.get("tools") or [],
        },
        "rendered": {
            "text": rendered.get("text", ""),
            "character_count": rendered.get("character_count"),
            "loss_character_count": rendered.get("loss_character_count"),
            "loss_character_spans": rendered_spans,
        },
        "tokenized": {
            "token_count": tokenized.get("token_count"),
            "decoded_text": tokenized.get("decoded_text", ""),
        },
        "loss": {
            "token_count": loss.get("token_count"),
            "masked_token_count": loss.get("masked_token_count"),
            "token_spans": loss_spans,
        },
        "error": record.get("error"),
    }


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>XTuner Sample Viewer</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f8;
      --panel: #ffffff;
      --line: #dfe3e6;
      --text: #1d252c;
      --muted: #65717b;
      --loss: #dff4e6;
      --loss-line: #2f855a;
      --masked: #f1f3f4;
      --assistant: #eef7ff;
      --tool: #fff7e8;
      --system: #f2efff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.55 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 5;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      padding: 12px 16px;
      background: rgba(255, 255, 255, .96);
      border-bottom: 1px solid var(--line);
    }
    header strong { margin-right: 6px; }
    select, button {
      min-height: 34px;
      border: 1px solid #c8ced3;
      border-radius: 6px;
      background: white;
      color: var(--text);
      padding: 5px 10px;
    }
    select { min-width: 190px; }
    button { cursor: pointer; }
    button:hover { background: #f3f5f6; }
    main { max-width: 1500px; margin: 0 auto; padding: 16px; }
    .path {
      flex: 1;
      min-width: 260px;
      overflow: hidden;
      color: var(--muted);
      font: 12px/1.3 ui-monospace, SFMono-Regular, Consolas, monospace;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .notice, .error {
      margin-bottom: 14px;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: var(--panel);
    }
    .error { color: #9b2c2c; border-color: #feb2b2; background: #fff5f5; }
    .stats {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(135px, 1fr));
      gap: 8px;
      margin-bottom: 14px;
    }
    .stat, section {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }
    .stat { padding: 9px 11px; }
    .stat small { display: block; color: var(--muted); }
    .stat b { font-size: 17px; }
    section { margin-bottom: 14px; overflow: hidden; }
    section > h2 {
      margin: 0;
      padding: 10px 13px;
      border-bottom: 1px solid var(--line);
      font-size: 15px;
    }
    .section-body { padding: 12px; }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      margin-bottom: 10px;
    }
    .legend { color: var(--muted); }
    .legend .swatch {
      display: inline-block;
      width: 12px;
      height: 12px;
      margin: 0 4px 0 10px;
      vertical-align: -2px;
      border: 1px solid var(--line);
    }
    .swatch.loss { background: var(--loss); border-color: var(--loss-line); }
    .swatch.masked { background: var(--masked); }
    pre, .rendered-text {
      margin: 0;
      overflow-wrap: anywhere;
      white-space: pre-wrap;
      word-break: break-word;
      font: 12px/1.5 ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
    }
    .rendered-text {
      max-height: 70vh;
      overflow: auto;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfcfc;
    }
    .rendered-text .masked { background: var(--masked); color: #667078; }
    .rendered-text .loss {
      background: var(--loss);
      box-shadow: inset 3px 0 0 var(--loss-line);
      color: #153d28;
    }
    .message {
      margin-bottom: 10px;
      border: 1px solid var(--line);
      border-left-width: 4px;
      border-radius: 6px;
      overflow: hidden;
    }
    .message.user { border-left-color: #3182ce; }
    .message.assistant { border-left-color: #38a169; background: var(--assistant); }
    .message.tool { border-left-color: #dd6b20; background: var(--tool); }
    .message.system { border-left-color: #805ad5; background: var(--system); }
    .message-head {
      display: flex;
      gap: 8px;
      align-items: center;
      padding: 6px 9px;
      border-bottom: 1px solid var(--line);
      font-weight: 600;
    }
    .badge {
      border: 1px solid #c8ced3;
      border-radius: 999px;
      padding: 0 7px;
      background: white;
      color: var(--muted);
      font-size: 11px;
      font-weight: 500;
    }
    .badge.loss-on { border-color: #68d391; color: #276749; }
    .badge.loss-off { border-color: #fc8181; color: #9b2c2c; }
    .message-body { padding: 9px; }
    details { margin-top: 8px; }
    summary { cursor: pointer; color: #40515e; }
    .tool-call, .label-span {
      margin-top: 8px;
      padding: 8px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: rgba(255,255,255,.75);
    }
    .tool-call strong, .label-span strong { display: block; margin-bottom: 5px; }
    .labels { display: grid; gap: 8px; }
    .label-span { border-left: 4px solid var(--loss-line); background: var(--loss); }
    .empty { padding: 18px; color: var(--muted); text-align: center; }
    [hidden] { display: none !important; }
  </style>
</head>
<body>
  <header>
    <strong>XTuner Sample Viewer</strong>
    <button id="prev" type="button">上一个</button>
    <select id="sample-select" aria-label="选择轨迹"></select>
    <button id="next" type="button">下一个</button>
    <span id="source-path" class="path"></span>
  </header>

  <main>
    <div id="status" class="notice">正在读取样本列表…</div>
    <div id="error" class="error" hidden></div>

    <div id="content" hidden>
      <div id="stats" class="stats"></div>

      <section>
        <h2>原始自然消息轨迹</h2>
        <div id="messages" class="section-body"></div>
      </section>

      <section>
        <h2>拼接文本与 Loss Mask</h2>
        <div class="section-body">
          <div class="toolbar">
            <button id="show-all" type="button">完整文本</button>
            <button id="show-loss" type="button">仅看 Loss</button>
            <span class="legend">
              <span class="swatch masked"></span>mask / label=-100
              <span class="swatch loss"></span>参与 loss
            </span>
          </div>
          <div id="rendered" class="rendered-text"></div>
        </div>
      </section>

      <section>
        <h2>Label 自然文本</h2>
        <div class="section-body">
          <div class="notice">
            这里展示 labels 中非 -100 token 解码后的自然文本，不展示任何 token ID。
          </div>
          <div id="labels" class="labels"></div>
        </div>
      </section>

      <section>
        <h2>工具定义</h2>
        <div class="section-body">
          <details>
            <summary id="tools-summary">展开工具定义</summary>
            <pre id="tools"></pre>
          </details>
        </div>
      </section>
    </div>
  </main>

  <script>
    const state = { samples: [], sample: null, current: -1, mode: "all" };
    const byId = (id) => document.getElementById(id);

    function formatNumber(value) {
      return Number.isFinite(value) ? value.toLocaleString("en-US") : "—";
    }

    function setError(message) {
      byId("error").textContent = message;
      byId("error").hidden = !message;
    }

    function addTextBlock(parent, text) {
      if (text === null || text === undefined || text === "") return;
      const pre = document.createElement("pre");
      pre.textContent = typeof text === "string" ? text : JSON.stringify(text, null, 2);
      parent.appendChild(pre);
    }

    function renderStats(sample) {
      const messages = sample.raw_data.messages || [];
      const values = [
        ["sample index", sample.source.sample_index],
        ["messages", messages.length],
        ["tokens", sample.tokenized.token_count],
        ["loss tokens", sample.loss.token_count],
        ["masked tokens", sample.loss.masked_token_count],
        ["rendered chars", sample.rendered.character_count],
        ["loss chars", sample.rendered.loss_character_count],
        ["loss spans", sample.loss.token_spans.length],
      ];
      const root = byId("stats");
      root.replaceChildren();
      for (const [name, value] of values) {
        const item = document.createElement("div");
        item.className = "stat";
        const label = document.createElement("small");
        label.textContent = name;
        const number = document.createElement("b");
        number.textContent = formatNumber(value);
        item.append(label, number);
        root.appendChild(item);
      }
    }

    function renderMessages(sample) {
      const root = byId("messages");
      root.replaceChildren();
      const messages = sample.raw_data.messages || [];
      messages.forEach((message, index) => {
        const role = message.role || "unknown";
        const card = document.createElement("article");
        card.className = `message ${role}`;

        const head = document.createElement("div");
        head.className = "message-head";
        const title = document.createElement("span");
        title.textContent = `${index}. ${role}`;
        head.appendChild(title);

        if (role === "assistant") {
          const badge = document.createElement("span");
          const enabled = message.loss !== false;
          badge.className = `badge ${enabled ? "loss-on" : "loss-off"}`;
          badge.textContent = enabled ? "loss=true" : "loss=false";
          head.appendChild(badge);
        }
        if (message.name) {
          const badge = document.createElement("span");
          badge.className = "badge";
          badge.textContent = message.name;
          head.appendChild(badge);
        }

        const body = document.createElement("div");
        body.className = "message-body";
        if (message.reasoning_content) {
          const detail = document.createElement("details");
          const summary = document.createElement("summary");
          summary.textContent = "reasoning_content";
          detail.appendChild(summary);
          addTextBlock(detail, message.reasoning_content);
          body.appendChild(detail);
        }
        addTextBlock(body, message.content);

        (message.tool_calls || []).forEach((call, callIndex) => {
          const fn = call.function || call;
          const block = document.createElement("div");
          block.className = "tool-call";
          const name = document.createElement("strong");
          name.textContent = `工具调用 ${callIndex + 1}: ${fn.name || "unknown"}`;
          block.appendChild(name);
          addTextBlock(block, fn.arguments || {});
          body.appendChild(block);
        });

        if (!body.childNodes.length) {
          const empty = document.createElement("div");
          empty.className = "empty";
          empty.textContent = "空内容";
          body.appendChild(empty);
        }
        card.append(head, body);
        root.appendChild(card);
      });
    }

    function renderMaskedText(sample) {
      const root = byId("rendered");
      root.replaceChildren();
      const text = sample.rendered.text || "";
      const spans = sample.rendered.loss_character_spans || [];
      const fragment = document.createDocumentFragment();

      if (state.mode === "loss") {
        spans.forEach((span, index) => {
          const block = document.createElement("span");
          block.className = "loss";
          block.textContent = `[loss span ${index + 1}]\n${text.slice(span.start, span.end)}\n\n`;
          fragment.appendChild(block);
        });
      } else {
        let cursor = 0;
        for (const span of spans) {
          if (span.start > cursor) {
            const masked = document.createElement("span");
            masked.className = "masked";
            masked.textContent = text.slice(cursor, span.start);
            fragment.appendChild(masked);
          }
          const enabled = document.createElement("span");
          enabled.className = "loss";
          enabled.textContent = text.slice(span.start, span.end);
          fragment.appendChild(enabled);
          cursor = span.end;
        }
        if (cursor < text.length) {
          const masked = document.createElement("span");
          masked.className = "masked";
          masked.textContent = text.slice(cursor);
          fragment.appendChild(masked);
        }
      }
      root.appendChild(fragment);
    }

    function renderLabels(sample) {
      const root = byId("labels");
      root.replaceChildren();
      const spans = sample.loss.token_spans || [];
      if (!spans.length) {
        const empty = document.createElement("div");
        empty.className = "empty";
        empty.textContent = "没有参与 loss 的 label 文本";
        root.appendChild(empty);
        return;
      }
      spans.forEach((span) => {
        const block = document.createElement("div");
        block.className = "label-span";
        const title = document.createElement("strong");
        title.textContent = `Label span ${span.number} · token [${span.start}, ${span.end})`;
        block.appendChild(title);
        addTextBlock(block, span.decoded_text);
        root.appendChild(block);
      });
    }

    function renderTools(sample) {
      const tools = sample.raw_data.tools || [];
      byId("tools-summary").textContent = `展开工具定义（${tools.length} 个）`;
      byId("tools").textContent = JSON.stringify(tools, null, 2);
    }

    function renderSample(sample) {
      state.sample = sample;
      renderStats(sample);
      renderMessages(sample);
      renderMaskedText(sample);
      renderLabels(sample);
      renderTools(sample);
      byId("content").hidden = false;
      byId("status").hidden = true;
      if (sample.error) setError(`${sample.error.type || "Error"}: ${sample.error.message || ""}`);
    }

    async function loadSample(position) {
      if (position < 0 || position >= state.samples.length) return;
      state.current = position;
      byId("sample-select").selectedIndex = position;
      byId("prev").disabled = position === 0;
      byId("next").disabled = position === state.samples.length - 1;
      const ref = state.samples[position];
      byId("status").hidden = false;
      byId("status").textContent = `正在读取 ${ref.label}…`;
      setError("");
      try {
        const response = await fetch(`/api/sample/${encodeURIComponent(ref.key)}`);
        if (!response.ok) throw new Error(await response.text());
        const sample = await response.json();
        renderSample(sample);
        history.replaceState(null, "", `#${encodeURIComponent(ref.key)}`);
      } catch (error) {
        byId("status").hidden = true;
        setError(String(error));
      }
    }

    async function init() {
      try {
        const response = await fetch("/api/samples");
        if (!response.ok) throw new Error(await response.text());
        const payload = await response.json();
        state.samples = payload.samples;
        byId("source-path").textContent = `${payload.store.input} · ${payload.store.sample_count} samples`;
        byId("source-path").title = payload.store.input;
        const select = byId("sample-select");
        state.samples.forEach((sample) => {
          const option = document.createElement("option");
          option.value = sample.key;
          option.textContent = sample.label;
          select.appendChild(option);
        });
        const hashKey = decodeURIComponent(location.hash.slice(1));
        const requested = state.samples.findIndex((sample) => sample.key === hashKey);
        await loadSample(requested >= 0 ? requested : 0);
      } catch (error) {
        byId("status").hidden = true;
        setError(String(error));
      }
    }

    byId("sample-select").addEventListener("change", (event) => loadSample(event.target.selectedIndex));
    byId("prev").addEventListener("click", () => loadSample(state.current - 1));
    byId("next").addEventListener("click", () => loadSample(state.current + 1));
    byId("show-all").addEventListener("click", () => {
      state.mode = "all";
      if (state.sample) renderMaskedText(state.sample);
    });
    byId("show-loss").addEventListener("click", () => {
      state.mode = "loss";
      if (state.sample) renderMaskedText(state.sample);
    });
    init();
  </script>
</body>
</html>
"""


class ViewerHandler(BaseHTTPRequestHandler):
    store: SampleStore

    def log_message(self, format_string: str, *args: Any) -> None:
        sys.stderr.write(f"[viewer] {self.address_string()} {format_string % args}\n")

    def _send_bytes(
        self,
        payload: bytes,
        *,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, data: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._send_bytes(payload, content_type="application/json; charset=utf-8", status=status)

    def _send_text(self, text: str, status: HTTPStatus) -> None:
        self._send_bytes(
            text.encode("utf-8"),
            content_type="text/plain; charset=utf-8",
            status=status,
        )

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_bytes(HTML.encode("utf-8"), content_type="text/html; charset=utf-8")
            return

        if parsed.path == "/api/samples":
            self._send_json(
                {
                    "store": self.store.public_info(),
                    "samples": self.store.list_samples(),
                }
            )
            return

        prefix = "/api/sample/"
        if parsed.path.startswith(prefix):
            key = unquote(parsed.path[len(prefix) :])
            try:
                record = self.store.read(key)
            except KeyError as error:
                self._send_text(str(error), HTTPStatus.NOT_FOUND)
                return
            except (OSError, ValueError, json.JSONDecodeError) as error:
                self._send_text(str(error), HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json(_natural_record(record))
            return

        if parsed.path == "/favicon.ico":
            self._send_bytes(b"", content_type=mimetypes.types_map.get(".ico", "image/x-icon"))
            return
        self._send_text("not found", HTTPStatus.NOT_FOUND)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a natural-text-only viewer for XTuner tokenize debug samples."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="sample_*.json directory, one sample JSON, or a JSONL file",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        store = SampleStore(args.input)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    handler = type("ConfiguredViewerHandler", (ViewerHandler,), {"store": store})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Reading: {store.input_path}")
    print(f"Samples: {len(store.list_samples())}")
    print(f"Open: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
