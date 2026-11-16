import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


RUN_INFO_FILE = "swanlab_run.json"


class SwanLabConfig(BaseModel):
    """Optional SwanLab settings shared by the V1 trainers."""

    model_config = ConfigDict(extra="forbid")

    project: str | None = None
    workspace: str | None = None
    experiment_name: str | None = None
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
    ):
        try:
            import swanlab
        except ImportError as exc:
            raise ImportError("SwanLab logging requires `pip install 'xtuner[swanlab]'`.") from exc

        config = config or SwanLabConfig()
        if config.resume == "never" and config.run_id is not None:
            raise ValueError("swanlab_config.run_id cannot be used with resume='never'.")

        directory = Path(log_dir) if log_dir is not None else Path()
        directory.mkdir(parents=True, exist_ok=True)
        run_info_path = directory / RUN_INFO_FILE
        if config.resume != "never" and resume_run is None and run_info_path.exists():
            resume_run = json.loads(run_info_path.read_text())
        if config.resume == "never" or (
            resume_run is not None and config.run_id is not None and config.run_id != resume_run["id"]
        ):
            resume_run = None

        saved = resume_run or {}
        run_id = config.run_id or saved.get("id")
        project = config.project or saved.get("project") or os.getenv("SWANLAB_PROJ_NAME") or "xtuner"
        workspace = config.workspace or saved.get("workspace") or os.getenv("SWANLAB_WORKSPACE")
        if saved:
            for field, value in (("project", project), ("workspace", workspace)):
                if saved.get(field) is not None and value != saved[field]:
                    raise ValueError(
                        f"Cannot resume SwanLab run {run_id!r} in a different {field}. "
                        "Use resume='never' to start a new experiment."
                    )

        self._run = swanlab.init(
            project=project,
            workspace=workspace,
            experiment_name=config.experiment_name,
            logdir=str(directory),
            mode=config.mode,
            id=run_id,
            resume=config.resume or ("allow" if run_id is not None else None),
            config=config.config,
        )
        self._closed = False
        try:
            self.run_info: dict[str, Any] = {"id": self._run.id, "project": project, "workspace": workspace}
            if self._run.mode == "online":
                # The online run path includes the resolved account/organization.
                workspace, project, _ = self._run.path.strip("/").split("/")
                self.run_info.update(project=project, workspace=workspace)
            temporary = run_info_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(self.run_info, indent=2) + "\n")
            temporary.replace(run_info_path)
        except Exception:
            self.close()
            raise

    def add_scalar(
        self,
        *,
        tag: str,
        scalar_value: float,
        global_step: int,
    ):
        self.add_scalars(tag_scalar_dict={tag: scalar_value}, global_step=global_step)

    def add_scalars(
        self,
        *,
        tag_scalar_dict: dict[str, float],
        global_step: int,
    ):
        if not self._closed and tag_scalar_dict:
            # Different series may use rollout steps and optimizer steps.
            self._run.log(tag_scalar_dict, step=global_step)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._run.finish()
