"""tb2-rl dataset: read prebuilt JSONL records, yield AgentRolloutItem.

The JSONL is produced by ``recipe.tb2_rl.scripts.generate_jsonl``. Each record
fully describes a task (id, instruction, tags, ability, pipeline_overrides),
so this module does nothing but field-mapping — no ``task.toml`` parsing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

from xtuner.v1.rl.agent_loop.rl_task.runner import Runner
from xtuner.v1.rl.agent_loop.rl_task.schemas import AgentRolloutItem


logger = logging.getLogger(__name__)


class TB2RLBench:
    """tb2-rl dataset iterator.  One pipeline (with per-task overrides) for
    every task it yields."""

    name = "tb2-rl"

    def __init__(
        self,
        jsonl_path: str | Path,
        *,
        pipeline: Runner | dict,
        skip_ids: set[str] | list[str] | None = None,
    ):
        """
        Args:
            jsonl_path (str | Path): JSONL produced by
                ``scripts/generate_jsonl.py`` (one record per task).
            pipeline: Shared Runner config for every task under this dataset.
            skip_ids (set[str] | list[str] | None): Sample ids to exclude.
        """
        self.pipeline = pipeline
        self.skip_ids = set(skip_ids or ())
        self._records = [
            json.loads(line)
            for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def iter_tasks(self) -> Iterator[tuple[Path, AgentRolloutItem]]:
        for rec in self._records:
            if rec["id"] in self.skip_ids:
                logger.info("skipping %s (in skip_ids)", rec["id"])
                continue
            try:
                yield Path(rec["task_dir"]), self.load_task(rec)
            except Exception as exc:
                logger.warning("skipping %s: %s", rec.get("id"), exc)

    def load_task(self, rec: dict) -> AgentRolloutItem:
        return AgentRolloutItem(
            id=rec["id"],
            data_source=self.name,
            ability=rec.get("ability"),
            tags=rec.get("tags", []),
            instruction=rec["instruction"],
            task_root=Path(rec["task_dir"]),
            pipeline=self.pipeline,
            pipeline_overrides=rec.get("pipeline_overrides", {}),
        )
