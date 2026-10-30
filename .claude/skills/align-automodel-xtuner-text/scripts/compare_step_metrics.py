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
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


def _integer(raw: object, context: str) -> int:
    """Parse integer indices without silently truncating fractional JSON numbers."""
    if isinstance(raw, bool):
        raise ValueError(f"{context}: expected an integer, got {raw!r}")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float) and math.isfinite(raw) and raw.is_integer():
        return int(raw)
    if isinstance(raw, str) and re.fullmatch(r"[+-]?\d+", raw.strip()):
        return int(raw)
    raise ValueError(f"{context}: expected an integer, got {raw!r}")


def _expected_updates(raw: str) -> tuple[int, int]:
    if not re.fullmatch(r"[+-]?\d+:[+-]?\d+", raw):
        raise ValueError("--expected-updates must be START:END with integer, inclusive logical update indices")
    start, end = map(int, raw.split(":"))
    if start > end:
        raise ValueError("--expected-updates requires START <= END")
    return start, end


def _validate_metrics(metrics: list[str]) -> None:
    if not metrics or any(not metric.strip() for metric in metrics):
        raise ValueError("--metrics must contain non-empty, comma-separated metric names")
    if len(metrics) != len(set(metrics)):
        raise ValueError("--metrics must not contain duplicate names")


def _finite(value: float, context: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{context}: non-finite value or arithmetic overflow")
    return value


def _validate_tolerances(atol: float | None, rtol: float | None) -> None:
    for name, value in (("atol", atol), ("rtol", rtol)):
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(f"--{name} must be finite and non-negative")


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
    if rank_wanted is not None and "rank" not in regex.groupindex:
        raise ValueError(f"--{prefix}-rank requires a named 'rank' group in --{prefix}-regex")
    offset = getattr(args, f"{prefix}_offset")
    side = Side(name)
    for line_number, line in enumerate(logical_lines(Path(path)), 1):
        match = regex.search(line)
        if not match:
            continue
        groups = match.groupdict()
        context = f"{name}: {path}:{line_number}"
        if rank_wanted is not None and _integer(groups["rank"], f"{context} rank") != rank_wanted:
            continue
        update = _integer(groups["step"], f"{context} step") + offset
        side.matched_lines += 1
        values = {}
        for key, raw in groups.items():
            if key in ("step", "rank") or raw is None:
                continue
            try:
                values[key] = float(raw)
            except ValueError:
                pass  # non-numeric named groups are not metrics
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
    if getattr(args, f"{prefix}_rank") is not None:
        raise ValueError(f"--{prefix}-rank is only supported for text logs with a named 'rank' group")
    mapping = {}
    for item in getattr(args, f"{prefix}_map"):
        metric, separator, key = item.partition("=")
        if not separator or not metric.strip() or not key.strip() or metric in mapping:
            raise ValueError(f"--{prefix}-map requires unique METRIC=DOTTED.KEY entries; got {item!r}")
        mapping[metric] = key
    offset = getattr(args, f"{prefix}_offset")
    side = Side(name)
    for line_number, line in enumerate(logical_lines(Path(path)), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{name}: {path}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(record, dict):
            continue
        if marker and dotted_get(record, marker) is None:
            continue  # select completed training rows, not eval-only or progress rows
        raw_step = dotted_get(record, step_key)
        if raw_step is None:
            continue
        update = _integer(raw_step, f"{name}: {path}:{line_number} step") + offset
        side.matched_lines += 1
        values = {}
        for metric, key in mapping.items():
            raw = dotted_get(record, key)
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                try:
                    values[metric] = float(raw)
                except OverflowError as exc:
                    raise ValueError(f"{name}: update {update}, metric {metric}: numeric conversion overflow") from exc
        if update in side.records:
            side.duplicates += 1
            if args.on_duplicate == "error":
                raise ValueError(f"{name}: duplicate logical update {update}; use --on-duplicate first/last")
            if args.on_duplicate == "first":
                continue
        side.records[update] = values
    return side


def compare(
    a: Side, b: Side, metrics: list[str], *, expected_updates: tuple[int, int] | None = None
) -> tuple[list[dict], dict]:
    """Pair shared logical updates, requiring full step coverage if a window is declared.

    Differences are always A - B. Missing metrics remain null for descriptive
    use; non-finite requested values and arithmetic overflow are input errors,
    including on unpaired updates within the selected window.
    """
    _validate_metrics(metrics)
    if expected_updates is not None:
        start, end = expected_updates
        if start > end:
            raise ValueError("expected updates require START <= END")
    selected = []
    summary = {"metrics_requested": metrics, "expected_updates": None, "metrics": {}}
    coverage_errors = []
    for key, side in (("a", a), ("b", b)):
        records = {
            update: values
            for update, values in side.records.items()
            if expected_updates is None or start <= update <= end
        }
        selected.append(records)
        summary[key] = {
            "records": len(side.records),
            "matched_lines": side.matched_lines,
            "duplicates": side.duplicates,
            "compared_records": len(records),
            "outside_window_records": len(side.records) - len(records),
        }
        if expected_updates is not None and len(records) != end - start + 1:
            first_missing = start
            for update in sorted(records):
                if update != first_missing:
                    break
                first_missing += 1
            coverage_errors.append(
                f"{side.name}: missing {end - start + 1 - len(records)} expected updates "
                f"(first: {first_missing})"
            )
        for update, values in records.items():
            for metric in metrics:
                if metric in values:
                    _finite(values[metric], f"{side.name}: update {update}, metric {metric}")
    if coverage_errors:
        raise ValueError("; ".join(coverage_errors))
    if expected_updates is not None:
        summary["expected_updates"] = {"start": start, "end": end}
    a_records, b_records = selected
    common = sorted(set(a_records) & set(b_records))
    rows = []
    for update in common:
        row = {"update": update}
        for metric in metrics:
            av, bv = a_records[update].get(metric), b_records[update].get(metric)
            row[f"a_{metric}"], row[f"b_{metric}"] = av, bv
            row[f"diff_{metric}"] = (
                _finite(av - bv, f"update {update}, metric {metric} difference")
                if av is not None and bv is not None
                else None
            )
        rows.append(row)
    summary["paired_updates"] = len(rows)
    summary["a_only_updates"] = sorted(set(a_records) - set(b_records))
    summary["b_only_updates"] = sorted(set(b_records) - set(a_records))
    for metric in metrics:
        diffs = [(r["update"], r[f"diff_{metric}"]) for r in rows if r[f"diff_{metric}"] is not None]
        info = summary["metrics"][metric] = {
            "paired": len(diffs),
            "a_missing_updates": sorted(u for u, v in a_records.items() if metric not in v),
            "b_missing_updates": sorted(u for u, v in b_records.items() if metric not in v),
            "mean_diff": None,
            "mean_abs_diff": None,
            "max_abs_diff": None,
            "max_abs_diff_update": None,
        }
        if diffs:
            worst = max(diffs, key=lambda item: abs(item[1]))
            info.update(
                {
                    "mean_diff": _finite(sum(d for _, d in diffs) / len(diffs), f"metric {metric} mean difference"),
                    "mean_abs_diff": _finite(
                        sum(abs(d) for _, d in diffs) / len(diffs), f"metric {metric} mean absolute difference"
                    ),
                    "max_abs_diff": abs(worst[1]),
                    "max_abs_diff_update": worst[0],
                }
            )
    return rows, summary


def apply_criteria(rows: list[dict], summary: dict, metrics: list[str], atol: float, rtol: float) -> bool:
    """Check |A-B| <= atol + rtol*|B| only on a complete, finite declared window."""
    _validate_metrics(metrics)
    _validate_tolerances(atol, rtol)
    window = summary.get("expected_updates")
    if window is None:
        raise ValueError("--expected-updates is required when --atol or --rtol is provided")
    updates = sorted(r["update"] for r in rows)
    if (
        not updates
        or len(updates) != window["end"] - window["start"] + 1
        or any(update != window["start"] + i for i, update in enumerate(updates))
    ):
        raise ValueError("criteria require every expected update exactly once on both sides")
    passed = True
    for metric in metrics:
        failures = []
        for row in rows:
            context = f"update {row['update']}, metric {metric}"
            for kind in ("a", "b", "diff"):
                value = row.get(f"{kind}_{metric}")
                if value is None:
                    raise ValueError(f"{context}: missing {kind} value; criteria require all requested metrics")
                _finite(value, f"{context} {kind}")
            diff, bv = row[f"diff_{metric}"], row[f"b_{metric}"]
            bound = _finite(atol + rtol * abs(bv), f"{context} tolerance bound")
            if abs(diff) > bound:
                failures.append((row["update"], diff, bv))
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
    # Validate JSON before opening either output. CSV columns never depend on the first row.
    encoded = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    fields = ["update"] + [
        f"{kind}_{metric}" for metric in summary["metrics_requested"] for kind in ("a", "b", "diff")
    ]
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with prefix.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    prefix.with_suffix(".json").write_text(encoded)


def plot(rows: list[dict], metrics: list[str], path: Path) -> None:
    try:
        import matplotlib
    except ImportError as exc:
        raise ValueError("--plot needs matplotlib; the CSV/JSON outputs were already written without it") from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    updates = [r["update"] for r in rows]
    fig, axes = plt.subplots(len(metrics), 2, figsize=(12, 3.2 * len(metrics)), squeeze=False)
    for i, metric in enumerate(metrics):
        for key, label in ((f"a_{metric}", "A"), (f"b_{metric}", "B")):
            pairs = [(r["update"], r[key]) for r in rows if r.get(key) is not None]
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
    parser.add_argument("--on-duplicate", choices=["first", "last", "error"], default="error")
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument(
        "--expected-updates", metavar="START:END",
        help="inclusive logical update window after offsets; required with --atol or --rtol",
    )
    parser.add_argument("--plot", default=None, help="optional PNG path; requires matplotlib")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = [m.strip() for m in args.metrics.split(",")]
    _validate_metrics(metrics)
    _validate_tolerances(args.atol, args.rtol)
    expected = _expected_updates(args.expected_updates) if args.expected_updates is not None else None
    has_criteria = args.atol is not None or args.rtol is not None
    if has_criteria and expected is None:
        raise ValueError("--expected-updates is required when --atol or --rtol is provided")
    sides = []
    for prefix, name in (("a", "A"), ("b", "B")):
        parse = parse_jsonl_side if getattr(args, f"{prefix}_format") == "jsonl" else parse_text_side
        side = parse(args, prefix, name)
        if not side.records:
            raise ValueError(f"side {name}: no records parsed; check the regex/keys against real log lines")
        sides.append(side)
    a, b = sides
    rows, summary = compare(a, b, metrics, expected_updates=expected)
    summary["diff_convention"] = "A - B"
    summary["on_duplicate"] = args.on_duplicate
    if not rows:
        raise ValueError("no shared logical updates; check offsets and step ranges")
    passed = None
    if has_criteria:
        passed = apply_criteria(rows, summary, metrics, args.atol or 0.0, args.rtol or 0.0)
        summary["passed"] = passed
    prefix = Path(args.out_prefix)
    write_outputs(rows, summary, prefix)
    if args.plot:
        plot(rows, metrics, Path(args.plot))
    print(json.dumps(summary, indent=2, allow_nan=False))
    print(f"wrote {prefix.with_suffix('.csv')} and {prefix.with_suffix('.json')}")
    if passed is None:
        print("no thresholds given: reported differences only, no pass/fail verdict")
    print("scalar metric criteria do not establish overall training alignment")
    return 0 if passed is not False else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, re.error, OverflowError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
