"""claw-bench dataset: iterate task.toml dirs, load AgentRolloutItem.

The dataset is deliberately thin:
  - ``iter_tasks()`` yields ``(task_dir, AgentRolloutItem)`` for every
    ``task.toml`` under ``tasks_root``.
  - ``pipeline`` is a runner object or a lazy ``dict(type=Runner, ...)``.
    :class:`ClawBench` never constructs one on its own.

Example config::

    from claw_bench.dataset import ClawBench
    from claw_bench.pipeline import runner

    dataset = ClawBench(
        tasks_root="/data/bench/claw-bench/tasks",
        pipeline=runner,
    )
"""

from __future__ import annotations

import logging
import tomllib
from pathlib import Path
from typing import Iterator

from xtuner.v1.ray.environment.rl_task.runner import Runner
from xtuner.v1.ray.environment.rl_task.schemas import AgentRolloutItem


logger = logging.getLogger(__name__)


class ClawBench:
    """Claw-bench dataset iterator.  One pipeline for every task it yields."""

    name = "claw-bench"

    def __init__(
        self,
        tasks_root: str | Path,
        *,
        pipeline: Runner | dict,
        skip_ids: set[str] | list[str] | None = None,
    ):
        """
        Args:
            tasks_root: Root dir of upstream task.toml layout.
            pipeline: Shared Runner config for every task under this dataset.
            skip_ids: Task ids (dir names) to exclude from iteration — use
                for known-broken upstream scripts (e.g. solve.sh bugs) so
                batch runs don't report them as infra failures.
        """
        self.tasks_root = Path(tasks_root).resolve()
        self.pipeline = pipeline
        self.skip_ids = set(skip_ids or ())

    def iter_tasks(self) -> Iterator[tuple[Path, AgentRolloutItem]]:
        """Yield ``(task_dir, AgentRolloutItem)`` for every task.toml under
        ``tasks_root``, minus anything in ``skip_ids``.

        Matches an entry in ``skip_ids`` either as the full dir name
        (e.g. ``"db-001"``) or as an id prefix with a ``-`` boundary
        (e.g. ``"fin-008"`` matches ``"fin-008-calculate-wacc-..."``).
        The boundary prevents ``"fin-00"`` from wrongly matching
        ``"fin-001"`` / ``"fin-008"``.
        """
        for toml_path in sorted(self.tasks_root.rglob("task.toml")):
            task_dir = toml_path.parent
            if self._is_skipped(task_dir.name):
                logger.info("skipping %s (in skip_ids)", task_dir.name)
                continue
            try:
                yield task_dir, self.load_task(task_dir)
            except Exception as exc:
                logger.warning("skipping %s: %s", task_dir, exc)

    def _is_skipped(self, dir_name: str) -> bool:
        for skip in self.skip_ids:
            if dir_name == skip or dir_name.startswith(skip + "-"):
                return True
        return False

    def load_task(self, task_dir: Path) -> AgentRolloutItem:
        toml = _load_task_toml(task_dir / "task.toml")
        return AgentRolloutItem(
            id=toml.get("id") or task_dir.name,
            data_source=self.name,
            ability=toml.get("domain"),
            tags=list(toml.get("tags") or []),
            instruction="instruction.md",
            task_root=task_dir,
            pipeline=self.pipeline,
        )


def _load_task_toml(path: Path) -> dict:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    if "task" in raw and isinstance(raw["task"], dict):
        task_section = raw.pop("task")
        for k, v in task_section.items():
            raw.setdefault(k, v)
    return raw
