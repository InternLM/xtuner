"""claw-bench dataset: iterate task.toml dirs, load TaskData.

The dataset is deliberately thin:
  - ``iter_tasks()`` yields ``(task_dir, TaskData)`` for every ``task.toml``
    under ``tasks_root``.
  - ``pipeline`` is whatever the caller's config built — :class:`ClawBench`
    never constructs one on its own.  Judger config lives inside the
    pipeline factory, not here.

Example config::

    from claw_bench.dataset import ClawBench
    from claw_bench.pipeline import claw_pipeline

    dataset = ClawBench(
        tasks_root="/data/bench/claw-bench/tasks",
        pipeline=claw_pipeline(),
    )
"""

from __future__ import annotations

import logging
import tomllib
from pathlib import Path
from typing import Iterator

from xtuner.v1.ray.environment.rl_task.runner import Runner
from xtuner.v1.ray.environment.rl_task.schemas import TaskData


logger = logging.getLogger(__name__)


class ClawBench:
    """Claw-bench dataset iterator.  One pipeline for every task it yields."""

    name = "claw-bench"

    def __init__(self, tasks_root: str | Path, *, pipeline: Runner):
        self.tasks_root = Path(tasks_root).resolve()
        self.pipeline = pipeline

    def iter_tasks(self) -> Iterator[tuple[Path, TaskData]]:
        """Yield ``(task_dir, TaskData)`` for every task.toml under ``tasks_root``."""
        for toml_path in sorted(self.tasks_root.rglob("task.toml")):
            task_dir = toml_path.parent
            try:
                yield task_dir, self.load_task(task_dir)
            except Exception as exc:
                logger.warning("skipping %s: %s", task_dir, exc)

    def load_task(self, task_dir: Path) -> TaskData:
        toml = _load_task_toml(task_dir / "task.toml")
        return TaskData(
            id=toml.get("id") or task_dir.name,
            data_source=self.name,
            ability=toml.get("domain"),
            tags=list(toml.get("tags") or []),
            instruction="instruction.md",
        )


def _load_task_toml(path: Path) -> dict:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    if "task" in raw and isinstance(raw["task"], dict):
        task_section = raw.pop("task")
        for k, v in task_section.items():
            raw.setdefault(k, v)
    return raw
