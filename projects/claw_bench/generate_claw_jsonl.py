#!/usr/bin/env python3
"""Scan claw-bench ``tasks/`` tree and emit one JSONL record per task directory.

Each record corresponds to a directory that directly contains ``task.toml``
(e.g. ``calendar/cal-001-create-meeting`` relative to ``tasks_root``).  All
paths in the JSON are POSIX-style paths relative to ``tasks_root``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# {
#     "claw-bench": {
#         "root_path": "/mnt/shared-storage-user/llmit/user/liukuikun/workspace/bench/claw-bench/tasks",
#         "annotation": "/mnt/shared-storage-user/huanghaian/code/temp/xtuner/workspace/agent_dev/claw_tasks.jsonl",
#         "sample_ratio": 1.0
#     }
# }


def iter_task_dirs(tasks_root: Path) -> list[Path]:
    """Return sorted task directories (parents of ``task.toml``)."""
    roots: list[Path] = []
    for toml in sorted(tasks_root.rglob("task.toml")):
        parent = toml.parent
        try:
            parent.relative_to(tasks_root)
        except ValueError:
            continue
        roots.append(parent)
    return roots


def rel_posix(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _is_junk_path(p: Path) -> bool:
    parts = set(p.parts)
    return "__pycache__" in parts or ".pytest_cache" in parts


def build_record(tasks_root: Path, task_dir: Path, *, skip_junk: bool) -> dict:
    task_rel = rel_posix(tasks_root, task_dir)
    files: list[str] = []
    for p in sorted(task_dir.rglob("*")):
        if not p.is_file():
            continue
        if skip_junk and _is_junk_path(p.relative_to(task_dir)):
            continue
        files.append(rel_posix(tasks_root, p))
    return {"task_dir": task_rel}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks-root",
        type=Path,
        default=Path("/mnt/shared-storage-user/llmit/user/liukuikun/workspace/bench/claw-bench/tasks"),
        help="Absolute path to the bench ``tasks`` directory.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(
            "/mnt/shared-storage-user/llmit/user/wangziyi/projs/xtuner_agent_dev/examples/demo_data/agent_dev/claw_tasks.jsonl"
        ),
        help="Write JSONL here.  Default: stdout.",
    )
    parser.add_argument(
        "--include-junk",
        action="store_true",
        help="Include __pycache__ / .pytest_cache files in ``files``.",
    )
    args = parser.parse_args()

    tasks_root = args.tasks_root.resolve()
    if not tasks_root.is_dir():
        print(f"tasks root is not a directory: {tasks_root}", file=sys.stderr)
        return 1

    out_fp = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    skip_junk = not args.include_junk
    try:
        for task_dir in iter_task_dirs(tasks_root):
            rec = build_record(tasks_root, task_dir, skip_junk=skip_junk)
            out_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except BrokenPipeError:
        return 0
    finally:
        if args.output:
            out_fp.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
