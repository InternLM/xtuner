"""Dataset = a family of tasks sharing one pipeline.

One file per data source (claw-bench, terminal-bench, …).  The dataset
knows:
  - where its tasks live on disk
  - how to turn a task dir into :class:`TaskData` (meta + outputs)
  - which :class:`Runner` (pipeline) runs every task in the family

Usage::

    from datasets.claw_bench import ClawBench
    ds = ClawBench(tasks_root="/path/to/bench/claw-bench/tasks")
    for task_dir, data in ds.iter_tasks():
        await ds.pipeline.run_single(task_dir, data, uid, provider=...)
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Iterator

from claw_bench.pipeline import PATHS, claw_pipeline
from judgers import Judger, pytest_ctrf_judger
from runner import Runner
from schemas import TaskData, WorkspaceArtifact


# Extract ``workspace / "FILE.EXT"`` references in the verifier's Python
# source to build the output contract.  Verifiers are authoritative: they
# name the files they actually check.
_VERIFIER_PATH_RE = re.compile(
    r"""workspace\s*/\s*['"]([A-Za-z0-9_./-]+?\.[a-zA-Z0-9]{1,6})['"]"""
)


class ClawBench:
    """Claw-bench dataset.  Every task under ``tasks_root`` uses the same pipeline.

    Layout expected::

        tasks_root/
          <domain>/
            <task-id>/
              task.toml
              instruction.md
              environment/{setup.sh, data/*}
              solution/solve.sh
              verifier/test_output.py
    """

    name = "claw-bench"

    def __init__(
        self,
        tasks_root: str | Path,
        *,
        judgers: list[Judger] | None = None,
    ):
        self.tasks_root = Path(tasks_root).resolve()
        self._judgers = judgers
        self._pipeline: Runner | None = None

    @property
    def pipeline(self) -> Runner:
        if self._pipeline is None:
            judgers = self._judgers or [self._default_judger()]
            self._pipeline = claw_pipeline(judgers=judgers)
        return self._pipeline

    def iter_tasks(self) -> Iterator[tuple[Path, TaskData]]:
        """Yield ``(task_dir, TaskData)`` for every task.toml under ``tasks_root``."""
        for toml_path in sorted(self.tasks_root.rglob("task.toml")):
            task_dir = toml_path.parent
            try:
                yield task_dir, self.load_task(task_dir)
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning(
                    "skipping %s: %s", task_dir, exc,
                )

    def load_task(self, task_dir: Path) -> TaskData:
        toml = _load_task_toml(task_dir / "task.toml")
        return TaskData(
            id=toml.get("id") or task_dir.name,
            data_source=self.name,
            ability=toml.get("domain"),
            tags=list(toml.get("tags") or []),
            instruction="instruction.md",
            outputs=_infer_outputs(task_dir / "verifier" / "test_output.py"),
        )

    # -- private --

    def _default_judger(self) -> Judger:
        return pytest_ctrf_judger(
            name="rule_grader",
            target="verifier/test_output.py",
            wrapper=f"{PATHS.wrappers_bench}/pytest_ctrf.sh",
            verifier_root=PATHS.verifier,
            weight=1.0,
            timeout=300,
        )


def _load_task_toml(path: Path) -> dict:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    if "task" in raw and isinstance(raw["task"], dict):
        task_section = raw.pop("task")
        for k, v in task_section.items():
            raw.setdefault(k, v)
    return raw


def _infer_outputs(verifier_path: Path) -> list[WorkspaceArtifact]:
    """Scan the pytest verifier for ``workspace / "X"`` references."""
    if not verifier_path.is_file():
        return []
    text = verifier_path.read_text(encoding="utf-8")
    seen: list[str] = []
    for m in _VERIFIER_PATH_RE.finditer(text):
        if m.group(1) not in seen:
            seen.append(m.group(1))
    artifacts: list[WorkspaceArtifact] = []
    for p in seen:
        ext = Path(p).suffix.lower()
        artifacts.append(WorkspaceArtifact(
            path=p, required=True, kind=_KIND_BY_EXT.get(ext, "text"),
        ))
    return artifacts


_KIND_BY_EXT: dict[str, str] = {
    ".json": "json",
    ".csv": "text", ".txt": "text", ".md": "text", ".html": "text",
    ".xml": "text", ".yml": "text", ".yaml": "text",
    ".py": "text", ".sh": "text",
    ".pdf": "binary", ".png": "binary", ".jpg": "binary", ".zip": "binary",
}
