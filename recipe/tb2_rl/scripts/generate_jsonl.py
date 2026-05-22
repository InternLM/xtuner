#!/usr/bin/env python3
"""Scan tb2-rl ``tasks/`` tree and emit one JSONL record per task directory.

Each record fully describes a task — downstream datasets do nothing but
field-mapping; ``task.toml`` parsing happens here only.

Record fields:
    task_dir              absolute path to the task directory
    id                    sample id (task.toml `id` or dir name)
    instruction           relative path to the instruction file
    tags                  list[str]
    ability               domain / category
    pipeline_overrides    dict to deep-merge into Runner._pool config
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import Any


image_mapping = {
    "data_processing": "t-data-processing-v1",
    "data_querying": "t-data-querying-v1",
    "data_science": "t-data-science-v1",
    "debugging": "t-debugging-v1",
    "dependency_management": "t-dependency-management-v1",
    "file_operations": "t-file-operations-v1",
    "scientific_computing": "t-scientific-computing-v1",
    "security": "t-security-v1",
    "software_engineering": "t-data-science-v1",
    "system_administration": "t-data-science-v1",
}


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


def _load_task_toml(path: Path) -> dict:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    if "task" in raw and isinstance(raw["task"], dict):
        for k, v in raw.pop("task").items():
            raw.setdefault(k, v)
    return raw


def build_record(task_dir: Path, tasks_root: Path) -> dict[str, Any]:
    toml = _load_task_toml(task_dir / "task.toml")
    tags = list((toml.get("metadata") or {}).get("tags") or toml.get("tags") or [])
    rel = task_dir.relative_to(tasks_root).as_posix()
    image = image_mapping[rel.split("/")[0]]
    return {
        "task_dir": str(task_dir.resolve()),
        "id": toml.get("id") or task_dir.name,
        "instruction": "instruction.md",
        "tags": tags,
        "ability": toml.get("domain") or (tags[1] if len(tags) > 1 else None),
        "pipeline_overrides": {"pool": {"specs": {"main": {"image": image}}}},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tasks-root",
        type=Path,
        default=Path("/mnt/shared-storage-user/llmit1/user/liukuikun/delivery/data/terminalbench2_rl_data"),
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
    skipped = 0
    with open(output, "w", encoding="utf-8") as fp:
        for task_dir in iter_task_dirs(tasks_root):
            try:
                rec = build_record(task_dir, tasks_root)
            except Exception as exc:
                print(f"warning: skipping {task_dir}: {exc}", file=sys.stderr)
                skipped += 1
                continue
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote: {output}" + (f"  ({skipped} skipped)" if skipped else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
