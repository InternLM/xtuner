#!/usr/bin/env python3
"""Scan tb2-rl ``tasks/`` tree and emit one JSONL record per task directory.

Each record corresponds to a directory that directly contains ``task.toml``
(e.g. ``data_processing_task_0`` relative to ``tasks_root``).  All paths in
the JSON are POSIX-style paths relative to ``tasks_root``.

Example meta.json pointing at the produced JSONL::

    {
        "tb2-rl": {
            "root_path": "/mnt/shared-storage-user/llmit1/user/liukuikun/delivery/data/data_process",
            "annotation": "<path>/tb2_rl_tasks.jsonl",
            "sample_ratio": 1.0
        }
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def iter_task_dirs(tasks_root: Path) -> list[Path]:
    """Return sorted task directories (parents of ``task.toml``).

    Args:
        tasks_root (Path): Root of the bench tasks tree.

    Returns:
        list[Path]: Directories that directly contain ``task.toml``.
    """
    roots: list[Path] = []
    for toml in sorted(tasks_root.rglob("task.toml")):
        parent = toml.parent
        try:
            parent.relative_to(tasks_root)
        except ValueError:
            continue
        roots.append(parent)
    return roots


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tasks-root",
        type=Path,
        default=Path("/mnt/shared-storage-user/llmit1/user/liukuikun/delivery/data/data_process"),
        help="Absolute path to the bench tasks directory.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write JSONL here.  Defaults to <tasks_root>/../tb2_rl_tasks.jsonl",
    )
    args = parser.parse_args()

    tasks_root = args.tasks_root.resolve()
    if not tasks_root.is_dir():
        print(f"tasks root is not a directory: {tasks_root}", file=sys.stderr)
        return 1

    output = args.output or (tasks_root.parent / "tb2_rl_tasks.jsonl")
    with open(output, "w", encoding="utf-8") as fp:
        for task_dir in iter_task_dirs(tasks_root):
            rel = task_dir.relative_to(tasks_root).as_posix()
            fp.write(json.dumps({"task_dir": rel}, ensure_ascii=False) + "\n")
    print(f"wrote: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
