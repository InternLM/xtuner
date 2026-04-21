#!/usr/bin/env python3
"""Parse a CTRF JSON report + pytest log → emit a ``JudgerResult`` line to stdout.

Honors ``@pytest.mark.weight(N)`` by reading the ``extra`` section of each test
(pytest-json-ctrf's convention).  Tests with no explicit weight get 1.0.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _extract_weight(test: dict) -> float:
    for section in ("extra", "metadata"):
        extras = test.get(section) or []
        if isinstance(extras, list):
            for e in extras:
                if isinstance(e, dict) and e.get("key") == "weight":
                    try:
                        return float(e.get("value", 1.0))
                    except (TypeError, ValueError):
                        return 1.0
        elif isinstance(extras, dict):
            w = extras.get("weight")
            if w is not None:
                try:
                    return float(w)
                except (TypeError, ValueError):
                    return 1.0
    return 1.0


def _log_tail(path: Path, bytes_: int = 800) -> str:
    try:
        return path.read_text(errors="replace")[-bytes_:]
    except Exception:
        return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctrf", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--pytest-rc", type=int, required=True)
    ap.add_argument("--judger-name", default="rule_grader")
    args = ap.parse_args()

    ctrf_path = Path(args.ctrf)
    log_path = Path(args.log)

    try:
        data = json.loads(ctrf_path.read_text())
    except Exception as exc:
        print(json.dumps({
            "judger_name": args.judger_name,
            "total": 0.0,
            "error": f"ctrf missing/parse failed: {exc}. pytest tail: {_log_tail(log_path)}",
        }, ensure_ascii=False))
        return 0

    tests = (data.get("results", {}) or {}).get("tests", []) or []
    criteria: dict[str, dict[str, float]] = {}
    for t in tests:
        name = t.get("name", "unknown")
        passed = (t.get("status") == "passed")
        weight = _extract_weight(t)
        criteria[name] = {"score": 1.0 if passed else 0.0, "weight": weight}

    total_w = sum(c["weight"] for c in criteria.values())
    if total_w <= 0:
        total = 0.0
    else:
        total = sum(c["score"] * c["weight"] for c in criteria.values()) / total_w

    print(json.dumps({
        "judger_name": args.judger_name,
        "total": round(total, 4),
        "criteria": criteria,
        "metadata": {
            "pytest_rc": args.pytest_rc,
            "test_count": len(tests),
        },
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
