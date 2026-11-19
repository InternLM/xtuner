import json
import logging
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


logger = logging.getLogger(__name__)

RUN_INFO_FILE = "swanlab_run.json"
TRAIN_STATE_FILE = "train_state.json"


class SwanLabConfig(BaseModel):
    """Optional SwanLab settings shared by the V1 trainers."""

    model_config = ConfigDict(extra="forbid")

    project: str | None = None
    workspace: str | None = None
    experiment_name: str | None = None
    group: str | None = None
    mode: Literal["online", "offline", "local", "disabled"] | None = None
    run_id: str | None = None
    resume: Literal["allow", "must", "never"] | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class SwanLabWriter:
    """Write existing XTuner metric names and steps to one SwanLab run."""

    def __init__(
        self,
        log_dir: str | Path | None = None,
        *,
        config: SwanLabConfig | None = None,
        resume_run: dict[str, Any] | None = None,
        checkpoint_path: str | Path | None = None,
        rank: int | None = None,
        default_group: str | None = None,
    ):
        try:
            import swanlab
        except ImportError as exc:
            raise ImportError("SwanLab logging requires `pip install 'xtuner[swanlab]'`.") from exc

        config = config or SwanLabConfig()
        self._rank = rank
        directory = Path(log_dir) if log_dir is not None else Path()
        directory.mkdir(parents=True, exist_ok=True)
        run_info_path = directory / RUN_INFO_FILE

        checkpoint_run = (
            None if config.resume == "never" else self._load_checkpoint_run(checkpoint_path, rank=rank)
        )
        checkpoint_group = (
            None if config.resume == "never" else self._load_checkpoint_group(checkpoint_path)
        )
        saved_run = None if config.resume == "never" else (checkpoint_run or resume_run)
        if config.resume != "never" and saved_run is None and run_info_path.exists():
            saved_run = self._read_json(run_info_path)

        configured_run_id = config.run_id
        if configured_run_id is not None and rank is not None and rank != 0:
            configured_run_id = f"{configured_run_id}-rank{rank}"
        if config.resume == "never" and configured_run_id is not None:
            raise ValueError("swanlab_config.run_id cannot be used with resume='never'.")

        if saved_run is not None and configured_run_id is not None:
            if configured_run_id != saved_run.get("id"):
                saved_run = None

        run_id = configured_run_id or (saved_run or {}).get("id")
        project = (
            config.project
            or (saved_run or {}).get("project")
            or os.getenv("SWANLAB_PROJ_NAME")
            or "xtuner"
        )
        workspace = config.workspace or (saved_run or {}).get("workspace") or os.getenv("SWANLAB_WORKSPACE")
        group = (
            config.group
            or (saved_run or {}).get("group")
            or checkpoint_group
            or os.getenv("SWANLAB_GROUP")
            or default_group
        )
        experiment_name = config.experiment_name or group or "xtuner"
        if rank is not None:
            experiment_name = f"{experiment_name}-rank{rank}"

        if saved_run is not None:
            for field, value in (("project", project), ("workspace", workspace)):
                if saved_run.get(field) is not None and value != saved_run[field]:
                    raise ValueError(
                        f"Cannot resume SwanLab run {run_id!r} in a different {field}. "
                        "Use resume='never' to start a new experiment."
                    )

        effective_resume = config.resume or ("allow" if run_id is not None else None)
        if (
            rank is not None
            and rank != 0
            and config.run_id is not None
            and config.resume == "must"
            and checkpoint_run is None
            and saved_run is None
        ):
            effective_resume = "allow"
            logger.warning(
                "Using resume='allow' for rank %s because its derived run ID is not available "
                "on the first multi-rank launch.",
                rank,
            )

        self._run = swanlab.init(
            project=project,
            workspace=workspace,
            experiment_name=experiment_name if rank is not None else config.experiment_name,
            group=group,
            logdir=str(directory),
            mode=config.mode,
            id=run_id,
            resume=effective_resume,
            config=config.config,
        )
        self._closed = False
        self.run_info: dict[str, Any] = {
            "id": self._run.id,
            "project": project,
            "workspace": workspace,
            "group": group,
        }
        if self._run.mode == "online":
            self._update_scope_from_run_path(project=project, workspace=workspace)
        self._write_json_best_effort(run_info_path, self.run_info, "local run identity")

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read SwanLab run identity from {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"SwanLab run identity at {path} must be a JSON object.")
        return value

    @classmethod
    def _load_checkpoint_run(cls, checkpoint_path: str | Path | None, *, rank: int | None) -> dict[str, Any] | None:
        if checkpoint_path is None:
            return None

        checkpoint = Path(checkpoint_path)
        if rank is not None:
            rank_path = checkpoint / "swanlab" / f"rank{rank}.json"
            if rank_path.exists():
                return cls._read_json(rank_path)

        state_path = checkpoint / TRAIN_STATE_FILE
        if not state_path.exists():
            return None
        state = cls._read_json(state_path)
        if rank is None or rank == 0:
            identity = state.get("swanlab_run")
            if identity is not None and not isinstance(identity, dict):
                raise ValueError(f"SwanLab run identity in {state_path} must be a JSON object.")
            return identity
        return None

    @classmethod
    def _load_checkpoint_group(cls, checkpoint_path: str | Path | None) -> str | None:
        if checkpoint_path is None:
            return None
        state_path = Path(checkpoint_path) / TRAIN_STATE_FILE
        if not state_path.exists():
            return None
        state = cls._read_json(state_path)
        identity = state.get("swanlab_run")
        if identity is None:
            return None
        if not isinstance(identity, dict):
            raise ValueError(f"SwanLab run identity in {state_path} must be a JSON object.")
        group = identity.get("group")
        return group if isinstance(group, str) else None

    def _update_scope_from_run_path(self, *, project: str, workspace: str | None) -> None:
        resolved_workspace = getattr(self._run, "workspace", None) or workspace
        resolved_project = getattr(self._run, "project", None) or project
        path = getattr(self._run, "path", None)
        parts = [part for part in str(path or "").strip("/").split("/") if part]
        if len(parts) >= 3:
            resolved_workspace = parts[-3]
            resolved_project = parts[-2]
        elif path:
            logger.warning("Could not resolve SwanLab project/workspace from run path %r.", path)
        self.run_info.update(project=resolved_project, workspace=resolved_workspace)

    @staticmethod
    def _write_json_best_effort(path: Path, value: dict[str, Any], description: str) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps(value, indent=2) + "\n")
            temporary.replace(path)
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("Could not persist SwanLab %s at %s: %s", description, path, exc)

    def add_scalar(
        self,
        *,
        tag: str,
        scalar_value: float,
        global_step: int,
    ) -> None:
        self.add_scalars(tag_scalar_dict={tag: scalar_value}, global_step=global_step)

    def add_scalars(
        self,
        *,
        tag_scalar_dict: dict[str, float],
        global_step: int,
    ) -> None:
        if not self._closed and tag_scalar_dict:
            # Different series may use rollout steps and optimizer steps.
            self._run.log(tag_scalar_dict, step=global_step)

    def get_state(self) -> dict[str, Any] | None:
        return dict(self.run_info)

    def persist_state(self, checkpoint_dir: str | Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        if self._rank is None:
            path = checkpoint_dir / RUN_INFO_FILE
        else:
            path = checkpoint_dir / "swanlab" / f"rank{self._rank}.json"
        self._write_json_best_effort(path, self.run_info, "checkpoint run identity")

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._run.finish()
