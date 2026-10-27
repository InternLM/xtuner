#!/usr/bin/env python3
"""Align per-update metrics from two training logs and report their differences.

Stdlib only; ``--plot`` additionally needs matplotlib.

Each side is either a text log parsed with a user-supplied regex (named groups:
``step`` required, ``rank`` optional, every other group is a metric) or a JSONL
tracker whose records are selected by a marker key. The step/lr phase mapping is
**never inferred**: pass explicit ``--a-offset`` / ``--b-offset`` derived from the
trainer source. Thresholds are the predeclared acceptance criteria, not defaults
chosen after seeing the result.

Example:
    python compare_step_metrics.py \
      --a-log am.log --a-regex 'step=(?P<step>\\d+).*loss=(?P<loss>[\\d.eE+-]+)' \
      --b-log xt.log --b-regex 'Step (?P<step>\\d+).*loss=(?P<loss>[\\d.eE+-]+)' \
      --a-offset 1 --metrics loss --out-prefix out/first_step_pair
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Side:
    """One framework's parsed records, keyed by logical update index."""

    name: str
    records: dict[int, dict[str, float]] = field(default_factory=dict)
    duplicates: int = 0
    matched_lines: int = 0


def logical_lines(path: Path) -> list[str]:
    """Split on newlines after normalizing carriage-return progress updates."""
    return path.read_text(errors="replace").replace("\r\n", "\n").replace("\r", "\n").split("\n")


def dotted_get(record: dict, key: str):
    value = record
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def parse_text_side(args, prefix: str, name: str) -> Side:
    path = getattr(args, f"{prefix}_log")
    pattern = getattr(args, f"{prefix}_regex")
    if not pattern:
        raise ValueError(f"--{prefix}-regex is required for text logs")
    regex = re.compile(pattern)
    if "step" not in regex.groupindex:
        raise ValueError(f"--{prefix}-regex must contain a named group 'step'")
    rank_wanted = getattr(args, f"{prefix}_rank")
    if "rank" in regex.groupindex and rank_wanted is None:
        raise ValueError(f"--{prefix}-regex captures 'rank'; pass --{prefix}-rank to select one explicitly")
    offset = getattr(args, f"{prefix}_offset")
    side = Side(name)
    for line in logical_lines(Path(path)):
        match = regex.search(line)
        if not match:
            continue
        groups = match.groupdict()
        if rank_wanted is not None and int(groups["rank"]) != rank_wanted:
            continue
        side.matched_lines += 1
        values = {}
        for key, raw in groups.items():
            if key in ("step", "rank") or raw is None:
                continue
            try:
                values[key] = float(raw)
            except ValueError:
                pass  # non-numeric named groups are not metrics
        update = int(groups["step"]) + offset
        if update in side.records:
            side.duplicates += 1
            if args.on_duplicate == "error":
                raise ValueError(f"{name}: duplicate logical update {update}; use --on-duplicate first/last")
            if args.on_duplicate == "first":
                continue
        side.records[update] = values
    return side


def parse_jsonl_side(args, prefix: str, name: str) -> Side:
    path = getattr(args, f"{prefix}_log")
    step_key = getattr(args, f"{prefix}_step_key")
    marker = getattr(args, f"{prefix}_marker")
    mapping = dict(item.split("=", 1) for item in getattr(args, f"{prefix}_map"))
    offset = getattr(args, f"{prefix}_offset")
    side = Side(name)
    for line in logical_lines(Path(path)):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        if marker and dotted_get(record, marker) is None:
            continue  # select completed training rows, not eval-only or progress rows
        raw_step = dotted_get(record, step_key)
        if raw_step is None:
            continue
        side.matched_lines += 1
        values = {}
        for metric, key in mapping.items():
            raw = dotted_get(record, key)
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                values[metric] = float(raw)
        update = int(raw_step) + offset
        if update in side.records:
            side.duplicates += 1
            if args.on_duplicate == "error":
                raise ValueError(f"{name}: duplicate logical update {update}; use --on-duplicate first/last")
            if args.on_duplicate == "first":
                continue
        side.records[update] = values
    return side


def compare(a: Side, b: Side, metrics: list[str]) -> tuple[list[dict], dict]:
    """Pair logical updates present on both sides; diff is always A - B."""
    common = sorted(set(a.records) & set(b.records))
    rows = []
    for update in common:
        row = {"update": update}
        for metric in metrics:
            if metric in a.records[update] and metric in b.records[update]:
                row[f"a_{metric}"] = a.records[update][metric]
                row[f"b_{metric}"] = b.records[update][metric]
                row[f"diff_{metric}"] = a.records[update][metric] - b.records[update][metric]
        rows.append(row)
    summary = {
        "paired_updates": len(rows),
        "a_only_updates": sorted(set(a.records) - set(b.records)),
        "b_only_updates": sorted(set(b.records) - set(a.records)),
        "a": {"records": len(a.records), "matched_lines": a.matched_lines, "duplicates": a.duplicates},
        "b": {"records": len(b.records), "matched_lines": b.matched_lines, "duplicates": b.duplicates},
        "metrics": {},
    }
    for metric in metrics:
        diffs = [(r["update"], r[f"diff_{metric}"]) for r in rows if f"diff_{metric}" in r]
        if not diffs:
            summary["metrics"][metric] = {"paired": 0}
            continue
        worst = max(diffs, key=lambda item: abs(item[1]))
        summary["metrics"][metric] = {
            "paired": len(diffs),
            "mean_diff": sum(d for _, d in diffs) / len(diffs),
            "mean_abs_diff": sum(abs(d) for _, d in diffs) / len(diffs),
            "max_abs_diff": abs(worst[1]),
            "max_abs_diff_update": worst[0],
        }
    return rows, summary


def apply_criteria(rows: list[dict], summary: dict, metrics: list[str], atol: float, rtol: float) -> bool:
    """|A-B| <= atol + rtol*|B| per update; bounds come from the declared contract."""
    passed = True
    for metric in metrics:
        failures = [
            (r["update"], r[f"diff_{metric}"], r[f"b_{metric}"])
            for r in rows
            if f"diff_{metric}" in r and abs(r[f"diff_{metric}"]) > atol + rtol * abs(r[f"b_{metric}"])
        ]
        info = summary["metrics"].setdefault(metric, {})
        info["criterion"] = {"atol": atol, "rtol": rtol, "failed_updates": len(failures)}
        if failures:
            passed = False
            info["criterion"]["first_failure"] = {
                "update": failures[0][0],
                "diff": failures[0][1],
                "b": failures[0][2],
            }
    return passed


def write_outputs(rows: list[dict], summary: dict, prefix: Path) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fields = ["update"] + [k for k in rows[0] if k != "update"] if rows else ["update"]
    with prefix.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")


def plot(rows: list[dict], metrics: list[str], path: Path) -> None:
    try:
        import matplotlib
    except ImportError as exc:
        raise SystemExit("--plot needs matplotlib; the CSV/JSON outputs were already written without it") from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    updates = [r["update"] for r in rows]
    fig, axes = plt.subplots(len(metrics), 2, figsize=(12, 3.2 * len(metrics)), squeeze=False)
    for i, metric in enumerate(metrics):
        for key, label in ((f"a_{metric}", "A"), (f"b_{metric}", "B")):
            pairs = [(r["update"], r[key]) for r in rows if key in r]
            if pairs:
                axes[i][0].plot([p[0] for p in pairs], [p[1] for p in pairs], label=label)
        axes[i][0].set_title(metric)
        axes[i][0].set_xlabel("logical update")
        axes[i][0].legend()
        diffs = [r.get(f"diff_{metric}") for r in rows]
        axes[i][1].plot(updates, [d if d is not None else float("nan") for d in diffs])
        axes[i][1].axhline(0.0, linewidth=0.8, color="gray")
        axes[i][1].set_title(f"{metric}: A - B")
        axes[i][1].set_xlabel("logical update")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for prefix, name in (("a", "A"), ("b", "B")):
        group = parser.add_argument_group(f"side {name}")
        group.add_argument(f"--{prefix}-log", required=True)
        group.add_argument(f"--{prefix}-format", choices=["text", "jsonl"], default="text")
        group.add_argument(f"--{prefix}-regex", help="text: named groups 'step', optional 'rank', metrics")
        group.add_argument(f"--{prefix}-rank", type=int, help="text: required when the regex captures 'rank'")
        group.add_argument(f"--{prefix}-offset", type=int, default=0, help="logical update = logged step + offset")
        group.add_argument(f"--{prefix}-step-key", default="step", help="jsonl: dotted step key")
        group.add_argument(f"--{prefix}-marker", default=None, help="jsonl: dotted key that completed rows must contain")
        group.add_argument(
            f"--{prefix}-map",
            action="append",
            default=[],
            metavar="METRIC=DOTTED.KEY",
            help="jsonl: metric mapping; repeat per metric",
        )
    parser.add_argument("--metrics", required=True, help="comma-separated metric names to compare")
    parser.add_argument("--on-duplicate", choices=["first", "last", "error"], default="last")
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--plot", default=None, help="optional PNG path; requires matplotlib")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    sides = []
    for prefix, name in (("a", "A"), ("b", "B")):
        parse = parse_jsonl_side if getattr(args, f"{prefix}_format") == "jsonl" else parse_text_side
        side = parse(args, prefix, name)
        if not side.records:
            raise ValueError(f"side {name}: no records parsed; check the regex/keys against real log lines")
        sides.append(side)
    a, b = sides
    rows, summary = compare(a, b, metrics)
    summary["diff_convention"] = "A - B"
    summary["metrics_requested"] = metrics
    if not rows:
        raise ValueError("no shared logical updates; check offsets and step ranges")
    passed = None
    if args.atol is not None or args.rtol:
        passed = apply_criteria(rows, summary, metrics, args.atol or 0.0, args.rtol)
        summary["passed"] = passed
    prefix = Path(args.out_prefix)
    write_outputs(rows, summary, prefix)
    if args.plot:
        plot(rows, metrics, Path(args.plot))
    print(json.dumps(summary, indent=2))
    print(f"wrote {prefix.with_suffix('.csv')} and {prefix.with_suffix('.json')}")
    if passed is None:
        print("no thresholds given: reported differences only, no pass/fail verdict")
    return 0 if passed is not False else 1


if __name__ == "__main__":
    sys.exit(main())
