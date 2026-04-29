"""tb2-eval dataset: iterate flat task dirs, load TaskData.

Upstream layout (flat — each task dir is a direct child of ``tasks_root``)::

    <task-name>/
        task.toml              # metadata (tags, timeouts, docker_image)
        instruction.md         # natural-language task
        environment/           # Dockerfile + files/
        tests/                 # test.sh, test_outputs.py, ...
        solution/              # reference solution

Unlike tb2-rl, there are no category subdirectories.  Each task ships its
own docker image in ``task.toml``'s ``[environment] docker_image`` field.
"""

from __future__ import annotations

import logging
import tomllib
from pathlib import Path
from typing import Iterator

from xtuner.v1.ray.environment.rl_task.runner import Runner
from xtuner.v1.ray.environment.rl_task.schemas import TaskData

logger = logging.getLogger(__name__)


class TB2EvalBench:
    """tb2-eval dataset iterator.  One pipeline for every task it yields."""

    name = "tb2-eval"

    def __init__(
        self,
        tasks_root: str | Path,
        *,
        pipeline: Runner,
        skip_ids: set[str] | list[str] | None = None,
    ):
        """
        Args:
            tasks_root (str | Path): Root dir containing flat task sub-dirs,
                each with ``task.toml``.
            pipeline (Runner): Shared Runner for every task under this dataset.
            skip_ids (set[str] | list[str] | None): Task ids (dir names) to
                exclude from iteration.  Defaults to ``None``.
        """
        self.tasks_root = Path(tasks_root).resolve()
        self.pipeline = pipeline
        self.skip_ids = set(skip_ids or ())

    def iter_tasks(self) -> Iterator[tuple[Path, TaskData]]:
        """Yield ``(task_dir, TaskData)`` for every direct-child ``task.toml``
        under ``tasks_root``, minus anything in ``skip_ids``.

        Returns:
            Iterator[tuple[Path, TaskData]]: per-task dir + metadata.
        """
        for toml_path in sorted(self.tasks_root.rglob("task.toml")):
            task_dir = toml_path.parent
            # Only accept direct children (flat layout, depth == 1)
            try:
                rel = task_dir.relative_to(self.tasks_root)
            except ValueError:
                continue
            if len(rel.parts) != 1:
                continue
            if self._is_skipped(task_dir.name):
                logger.info("skipping %s (in skip_ids)", task_dir.name)
                continue
            try:
                yield task_dir, self.load_task(task_dir)
            except Exception as exc:
                logger.warning("skipping %s: %s", task_dir, exc)

    def load_task(self, task_dir: Path) -> TaskData:
        """Load ``TaskData`` from a task directory.

        Args:
            task_dir (Path): Directory containing ``task.toml`` and ``instruction.md``.

        Returns:
            TaskData: Metadata + instruction-relative path.
        """
        toml = _load_task_toml(task_dir / "task.toml")
        tags = list((toml.get("metadata") or {}).get("tags") or toml.get("tags") or [])
        return TaskData(
            id=toml.get("id") or task_dir.name,
            data_source=self.name,
            ability=toml.get("domain") or (tags[1] if len(tags) > 1 else None),
            tags=tags,
            instruction="instruction.md",
        )

    def _is_skipped(self, dir_name: str) -> bool:
        for skip in self.skip_ids:
            if dir_name == skip or dir_name.startswith(skip + "-"):
                return True
        return False


def _load_task_toml(path: Path) -> dict:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    if "task" in raw and isinstance(raw["task"], dict):
        task_section = raw.pop("task")
        for k, v in task_section.items():
            raw.setdefault(k, v)
    return raw
