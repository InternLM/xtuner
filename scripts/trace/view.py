#!/usr/bin/env python
"""View per-sample trace events emitted by xtuner.v1.ray.environment.trace.

The trace layer writes two jsonl channels under ``$WORK_DIR/trace/``:

* ``fates.*.jsonl`` — one terminal line per sample (``COMPLETED`` / ``SKIPPED``
  + ``failed_stage`` + ``reason``)
* ``spans.*.jsonl`` — stage-level events (``acquire`` / ``infer`` / ``validate``
  / ``run_single_total``) with ``duration_ms``, ``ok``, ``err``

Each file is produced by a single InstallAgentEnvironment actor.  The viewer
scans the whole directory and correlates events by ``uid``.

Common recipes::

    # Single sample timeline
    python scripts/trace/view.py --uid 129803423424164365087606594476061761437

    # Stage latency + fate distribution (big picture)
    python scripts/trace/view.py --stats

    # Show all samples that failed with a heartbeat-related reason
    python scripts/trace/view.py --grep-fail heartbeat

    # Top 20 slowest spans across the run
    python scripts/trace/view.py --slow 20

    # Live stream (tail -f across all jsonl in dir)
    python scripts/trace/view.py --tail
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterator


# ─────────────────────────────────────────────────────────────────
# Mirror of ``xtuner.v1.ray.environment.install_agent_env
# ._classify_mark_failed_reason`` so ``--stats`` reclassifies old fates
# that were emitted before the classifier had a pattern for their
# reason (e.g. post-hook 404s that all landed in the fallback
# ``mark_failed`` bucket on rc22).  Keep these two in sync.
# ─────────────────────────────────────────────────────────────────
_RE_RETURN_CODE = re.compile(r"return_code=(-?\d+)")


def _reclassify(reason: str | None) -> str:
    r = reason or ""
    if "DaemonStuckError" in r:
        return "daemon_stuck"
    if "could not acquire" in r or "sandbox after" in r:
        return "acquire_failed"
    if "post-hook" in r:
        if "404 Not Found" in r:
            return "posthook_download_404"
        if "Source file does not exist" in r:
            return "posthook_file_missing"
        return "posthook_failed"
    if "pre-hook" in r:
        return "prehook_failed"
    m = _RE_RETURN_CODE.search(r)
    if m:
        rc = int(m.group(1))
        if rc == -1:
            return "entry_pid_lost"
        if rc == -2:
            return "entry_daemon_gone"
        if rc == -3:
            return "entry_timeout"
        if rc == -9 or rc == 137:
            return "oom_killed"
        if rc > 0:
            return f"entry_rc_{rc}"
    if "TimeoutError" in r:
        return "timeout"
    if "OutOfMemoryError" in r or "OutOfMemory" in r:
        return "oom"
    return "mark_failed"


def _effective_failed_stage(rec: dict) -> str | None:
    """Return the fate's ``failed_stage``, reclassifying stale
    ``mark_failed`` rows through :func:`_reclassify` so historical fates
    get the same granularity as newly emitted ones."""
    stage = rec.get("failed_stage")
    if rec.get("final") == "SKIPPED" and stage == "mark_failed":
        return _reclassify(rec.get("reason"))
    return stage


def _find_files(trace_dir: Path, kind: str) -> list[Path]:
    return sorted(trace_dir.glob(f"{kind}.*.jsonl"))


def _load_records(trace_dir: Path, kind: str) -> Iterator[dict]:
    for path in _find_files(trace_dir, kind):
        with open(path, "r", encoding="utf-8") as fp:
            for lineno, raw in enumerate(fp, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    yield json.loads(raw)
                except json.JSONDecodeError:
                    sys.stderr.write(f"skip malformed line {path}:{lineno}\n")


def _percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    idx = min(int(len(values) * q / 100), len(values) - 1)
    return sorted(values)[idx]


def cmd_uid(trace_dir: Path, uid: str) -> int:
    spans = [r for r in _load_records(trace_dir, "spans") if r.get("uid") == uid]
    fates = [r for r in _load_records(trace_dir, "fates") if r.get("uid") == uid]
    if not spans and not fates:
        print(f"no records found for uid={uid}")
        return 1
    spans.sort(key=lambda r: r.get("ts", 0))
    t0 = spans[0]["ts"] if spans else fates[0]["ts"]
    task_id = next((r.get("task_id") for r in spans if r.get("task_id")), None)
    fate = fates[-1] if fates else None
    header_fate = fate["final"] if fate else "?"
    print(f"uid={uid}  task={task_id}  final={header_fate}")
    print(f"{'stage':<20} {'ts(rel)':>9} {'duration':>10} {'status':>7}  note")
    for s in spans:
        rel = s.get("ts", 0) - t0
        dur = s.get("duration_ms", 0) / 1000.0
        status = "OK" if s.get("ok") else "FAIL"
        note = (s.get("err") or "")[:72]
        print(f"{s.get('stage', '?'):<20} {rel:>8.1f}s {dur:>9.1f}s {status:>7}  {note}")
    if fate:
        print(
            f"fate: failed_stage={_effective_failed_stage(fate)}  "
            f"reason={fate.get('reason')}"
        )
    return 0


def cmd_stats(trace_dir: Path) -> int:
    spans = list(_load_records(trace_dir, "spans"))
    fates = list(_load_records(trace_dir, "fates"))
    by_stage: dict[str, list[int]] = defaultdict(list)
    fail_by_stage: dict[str, int] = defaultdict(int)
    for s in spans:
        stage = s.get("stage")
        if stage is None:
            continue
        by_stage[stage].append(s.get("duration_ms", 0))
        if not s.get("ok", True):
            fail_by_stage[stage] += 1
    print(f"spans (n={len(spans)})")
    print(f"  {'stage':<24} {'p50':>8} {'p95':>8} {'p99':>8}  {'fail%':>7}")
    for stage in sorted(by_stage):
        durs = by_stage[stage]
        p50 = _percentile(durs, 50) / 1000.0
        p95 = _percentile(durs, 95) / 1000.0
        p99 = _percentile(durs, 99) / 1000.0
        fail_pct = fail_by_stage[stage] * 100.0 / len(durs) if durs else 0.0
        print(f"  {stage:<24} {p50:>7.1f}s {p95:>7.1f}s {p99:>7.1f}s  {fail_pct:>6.1f}%")
    print()
    by_fate: dict[str, int] = defaultdict(int)
    for r in fates:
        final = r.get("final", "?")
        if final == "SKIPPED":
            final = f"SKIPPED/{_effective_failed_stage(r) or '?'}"
        by_fate[final] += 1
    total = len(fates) or 1
    print(f"fates (n={len(fates)})")
    for key, count in sorted(by_fate.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {key:<28} {count:>6}  ({count * 100.0 / total:>5.1f}%)")

    # ── sandbox_image × outcome ──────────────────────────────────────
    by_image: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in fates:
        img = r.get("sandbox_image") or "(none)"
        outcome = r.get("final", "?")
        if outcome == "SKIPPED":
            outcome = _effective_failed_stage(r) or "?"
        by_image[img]["_total"] += 1
        by_image[img][outcome] += 1
    if any(img != "(none)" for img in by_image):
        print()
        print("by sandbox_image:")
        # Find all outcomes across images for a consistent column set
        outcomes = sorted({k for per in by_image.values() for k in per if k != "_total"})
        header = "  " + f"{'image':<26} {'n':>6}  " + "  ".join(f"{o[:18]:<18}" for o in outcomes)
        print(header)
        for img in sorted(by_image, key=lambda k: -by_image[k]["_total"]):
            if img == "(none)":
                continue
            per = by_image[img]
            row = f"  {img[:26]:<26} {per['_total']:>6}  " + "  ".join(
                f"{per.get(o, 0):<18}" for o in outcomes
            )
            print(row)

    # ── entry_rc distribution (SKIPPED only) ─────────────────────────
    rc_counts: dict[int, int] = defaultdict(int)
    for r in fates:
        rc = r.get("entry_rc")
        if rc is not None and r.get("final") == "SKIPPED":
            rc_counts[rc] += 1
    if rc_counts:
        print()
        print("entry_rc distribution (SKIPPED only):")
        print(f"  {'rc':>4}  {'n':>6}  meaning")
        for rc in sorted(rc_counts, key=lambda k: -rc_counts[k]):
            meaning = {
                -3: "infer_timeout (hit SandboxStage.timeout)",
                -2: "infer_daemon_gone (pgrep lost daemon)",
                -1: "infer_pid_lost (wrapper shell died)",
                -9: "infer_oom",
                137: "infer_oom",
                0: "entry rc=0 but ok=False (abnormal)",
            }.get(rc, f"entry script exit rc={rc}")
            print(f"  {rc:>4}  {rc_counts[rc]:>6}  {meaning}")
    return 0


def cmd_grep_fail(trace_dir: Path, pattern: str) -> int:
    rx = re.compile(pattern)
    hits: list[dict] = []
    for r in _load_records(trace_dir, "fates"):
        stage = _effective_failed_stage(r) or ""
        combined = " ".join(
            str(val or "") for val in (r.get("final"), stage, r.get("reason"))
        )
        if rx.search(combined):
            hits.append(r)
    for r in hits:
        print(
            f"{r.get('uid')}  task={r.get('task_id')}  "
            f"final={r.get('final')}  failed_stage={_effective_failed_stage(r)}  "
            f"reason={r.get('reason')}"
        )
    print(f"# {len(hits)} matches")
    return 0


def cmd_slow(trace_dir: Path, n: int) -> int:
    spans = list(_load_records(trace_dir, "spans"))
    spans.sort(key=lambda r: r.get("duration_ms", 0), reverse=True)
    for s in spans[:n]:
        status = "OK" if s.get("ok") else "FAIL"
        dur = s.get("duration_ms", 0) / 1000.0
        print(
            f"{dur:>8.1f}s  {status:>6}  stage={s.get('stage')}  "
            f"uid={s.get('uid')}  task={s.get('task_id')}"
        )
    return 0


def cmd_llm_stats(trace_dir: Path) -> int:
    """Aggregate ``llm_calls.*.jsonl`` into p50 / p95 / p99 for each
    timing field, plus token-count distributions."""
    records: list[dict] = []
    for path in sorted(trace_dir.glob("llm_calls.*.jsonl")):
        with open(path, "r", encoding="utf-8") as fp:
            for raw in fp:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    records.append(json.loads(raw))
                except json.JSONDecodeError:
                    continue
    if not records:
        print("no llm_calls.*.jsonl records found")
        return 1

    def _pcts(values: list[float]) -> tuple[float, float, float, float]:
        vs = sorted(values)
        n = len(vs)
        return (vs[n // 2], vs[min(int(n * 0.95), n - 1)], vs[min(int(n * 0.99), n - 1)], vs[-1])

    fields = [
        ("total_ms", "total", 1000.0, "s"),
        ("rollout_ms", "rollout", 1000.0, "s"),
        ("tokenize_ms", "tokenize", 1000.0, "s"),
        ("post_ms", "post", 1000.0, "s"),
        ("prompt_tokens", "prompt_tok", 1.0, ""),
        ("completion_tokens", "completion_tok", 1.0, ""),
    ]
    print(f"llm_calls (n={len(records)})")
    print(f"  {'field':<14} {'p50':>10} {'p95':>10} {'p99':>10} {'max':>10}")
    for key, label, scale, unit in fields:
        vs = [r[key] for r in records if r.get(key) is not None]
        if not vs:
            continue
        p50, p95, p99, mx = _pcts(vs)
        print(
            f"  {label:<14} "
            f"{p50 / scale:>9.2f}{unit} "
            f"{p95 / scale:>9.2f}{unit} "
            f"{p99 / scale:>9.2f}{unit} "
            f"{mx / scale:>9.2f}{unit}"
        )
    # Bucketize total duration for a quick long-tail view.
    buckets = [(0, 5), (5, 30), (30, 60), (60, 120), (120, 300),
               (300, 600), (600, 1200), (1200, 1800), (1800, 10000)]
    print()
    print("total duration buckets (seconds):")
    for lo, hi in buckets:
        cnt = sum(1 for r in records if lo * 1000 <= r.get("total_ms", 0) < hi * 1000)
        if cnt:
            pct = cnt * 100.0 / len(records)
            print(f"  [{lo:>4}, {hi:>5}):  {cnt:>7}  ({pct:>5.1f}%)")
    return 0


def cmd_tail(trace_dir: Path) -> int:
    open_files: dict[Path, "io_TextFile"] = {}
    try:
        while True:
            for path in _find_files(trace_dir, "spans") + _find_files(trace_dir, "fates"):
                if path not in open_files:
                    fp = open(path, "r", encoding="utf-8")
                    fp.seek(0, 2)
                    open_files[path] = fp
            produced = False
            for path, fp in open_files.items():
                for raw in fp:
                    raw = raw.rstrip()
                    if raw:
                        sys.stdout.write(f"[{path.stem}] {raw}\n")
                        produced = True
            sys.stdout.flush()
            if not produced:
                time.sleep(0.5)
    except KeyboardInterrupt:
        return 0


def main() -> int:
    default_dir = Path(os.environ.get("WORK_DIR", ".")) / "trace"
    parser = argparse.ArgumentParser(
        description="View xtuner install-env trace events.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--trace-dir", type=Path, default=default_dir)
    parser.add_argument("--uid", help="show gantt for one sample")
    parser.add_argument("--stats", action="store_true", help="stage p50/p95/p99 + fate distribution")
    parser.add_argument("--grep-fail", metavar="REGEX", help="list fates matching regex (final/failed_stage/reason)")
    parser.add_argument("--slow", type=int, metavar="N", help="top N slowest spans")
    parser.add_argument("--tail", action="store_true", help="live stream jsonl")
    parser.add_argument("--llm-stats", action="store_true", help="p50/p95/p99 of /v1/chat/completions requests")
    args = parser.parse_args()

    if not args.trace_dir.exists():
        parser.error(f"trace dir {args.trace_dir} does not exist")

    if args.uid:
        return cmd_uid(args.trace_dir, args.uid)
    if args.stats:
        return cmd_stats(args.trace_dir)
    if args.grep_fail:
        return cmd_grep_fail(args.trace_dir, args.grep_fail)
    if args.slow is not None:
        return cmd_slow(args.trace_dir, args.slow)
    if args.tail:
        return cmd_tail(args.trace_dir)
    if args.llm_stats:
        return cmd_llm_stats(args.trace_dir)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
