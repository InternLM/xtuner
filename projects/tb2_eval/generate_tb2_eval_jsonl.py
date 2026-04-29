#!/usr/bin/env python3
"""Scan tb2-eval ``tasks/`` tree and emit one JSONL record per task directory.

Unlike tb2-rl (which has category subdirectories), the eval bench uses a
**flat** layout: each task directory sits directly under ``tasks_root`` and
contains ``task.toml``.  The docker image is read per-task from
``task.toml``'s ``[environment] docker_image`` field.

Example meta.json pointing at the produced JSONL::

    {
        "tb2-eval": {
            "root_path": "/mnt/shared-storage-user/llmit/user/liukuikun/workspace/bench/terminal-bench-2",
            "annotation": "<path>/tb2_eval_tasks.jsonl",
            "sample_ratio": 1.0
        }
    }
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path


def iter_task_dirs(tasks_root: Path) -> list[Path]:
    """Return sorted task directories (parents of ``task.toml``).

    Only immediate children of ``tasks_root`` are returned (flat layout).

    Args:
        tasks_root (Path): Root of the eval tasks tree.

    Returns:
        list[Path]: Directories that directly contain ``task.toml``.
    """
    roots: list[Path] = []
    for toml in sorted(tasks_root.rglob("task.toml")):
        parent = toml.parent
        # Only accept direct children (depth == 1)
        try:
            rel = parent.relative_to(tasks_root)
        except ValueError:
            continue
        if len(rel.parts) == 1:
            roots.append(parent)
    return roots


def _read_docker_image(task_dir: Path) -> str:
    """Read ``[environment] docker_image`` from ``task.toml``.

    Args:
        task_dir (Path): Task directory containing ``task.toml``.

    Returns:
        str: The docker image tag.

    Raises:
        KeyError: If the field is missing.
    """
    raw = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
    env = raw.get("environment") or {}
    image = env.get("docker_image")
    if not image:
        raise KeyError(f"[environment] docker_image missing in {task_dir / 'task.toml'}")
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tasks-root",
        type=Path,
        default=Path(
            "/mnt/shared-storage-user/llmit/user/wangziyi/projs/terminalbench2-harbor-p-cluster/terminal-bench-2"
        ),
        help="Absolute path to the eval tasks directory.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write JSONL here.  Defaults to <tasks_root>/../tb2_eval_tasks.jsonl",
    )
    args = parser.parse_args()

    tasks_root = args.tasks_root.resolve()
    if not tasks_root.is_dir():
        print(f"tasks root is not a directory: {tasks_root}", file=sys.stderr)
        return 1

    output = args.output or (tasks_root.parent / "tb2_eval_tasks.jsonl")
    skipped = 0
    with open(output, "w", encoding="utf-8") as fp:
        for task_dir in iter_task_dirs(tasks_root):
            rel = task_dir.relative_to(tasks_root).as_posix()
            try:
                image = _read_docker_image(task_dir)
            except (KeyError, Exception) as exc:
                print(f"warning: skipping {rel}: {exc}", file=sys.stderr)
                skipped += 1
                continue
            fp.write(json.dumps({"task_dir": rel, "image": image}, ensure_ascii=False) + "\n")
    print(f"wrote: {output}" + (f"  ({skipped} skipped)" if skipped else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
