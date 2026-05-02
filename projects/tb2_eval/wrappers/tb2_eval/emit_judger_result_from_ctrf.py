#!/usr/bin/env python3
"""Emit a ``JudgerResult`` line to stdout that matches official TB2 scoring.

The bench's ``tests/test.sh`` writes the authoritative binary outcome to
``/logs/verifier/reward.txt`` (``1`` iff every pytest invocation in that
script exited 0, else ``0``) — including the multi-pytest case in tasks
like ``fix-code-vulnerability``. We read that file as the source of truth
for ``total``. CTRF is parsed only for per-test observability in the
``criteria`` field and never used for scoring.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _log_tail(path: Path, bytes_: int = 800) -> str:
    try:
        return path.read_text(errors="replace")[-bytes_:]
    except Exception:
        return ""


def _read_reward(path: Path) -> float | None:
    try:
        raw = path.read_text().strip()
    except Exception:
        return None
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _parse_criteria(ctrf_path: Path) -> tuple[dict[str, dict[str, float]], int, str | None]:
    """Parse CTRF into per-test criteria for observability only.

    Returns:
        tuple[dict[str, dict[str, float]], int, str | None]: ``(criteria, test_count,
        error)``. ``criteria`` maps test name to ``{"score": 0.0|1.0}``. ``error`` is
        ``None`` on success or a message describing why CTRF was unreadable.
    """
    try:
        data = json.loads(ctrf_path.read_text())
    except Exception as exc:
        return {}, 0, f"ctrf missing/parse failed: {exc}"
    tests = (data.get("results", {}) or {}).get("tests", []) or []
    criteria: dict[str, dict[str, float]] = {}
    for t in tests:
        name = t.get("name", "unknown")
        passed = t.get("status") == "passed"
        criteria[name] = {"score": 1.0 if passed else 0.0}
    return criteria, len(tests), None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctrf", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--reward-file", required=True)
    ap.add_argument("--pytest-rc", type=int, required=True)
    ap.add_argument("--judger-name", default="rule_grader")
    args = ap.parse_args()

    ctrf_path = Path(args.ctrf)
    log_path = Path(args.log)
    reward_path = Path(args.reward_file)

    reward = _read_reward(reward_path)
    criteria, test_count, ctrf_error = _parse_criteria(ctrf_path)

    result: dict = {
        "judger_name": args.judger_name,
        "criteria": criteria,
        "metadata": {
            "pytest_rc": args.pytest_rc,
            "test_count": test_count,
            "reward_source": "reward.txt" if reward is not None else "pytest_rc",
        },
    }

    if reward is not None:
        result["total"] = round(reward, 4)
    else:
        # reward.txt missing/unreadable: fall back to test.sh exit code.
        result["total"] = 1.0 if args.pytest_rc == 0 else 0.0
        result["error"] = (
            f"reward file unreadable at {reward_path}; fell back to pytest_rc. "
            f"log tail: {_log_tail(log_path)}"
        )

    if ctrf_error is not None:
        # CTRF is observability-only; surface the parse error but don't change total.
        result.setdefault("error", ctrf_error)

    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
