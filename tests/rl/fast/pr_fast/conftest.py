"""PR-fast import hygiene.

The tests in this directory exercise RL state machines and trainer control
flow with fakes. They should not require model kernels, dataset builders, or
real training workers during collection.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict


# PR-fast Ray tests start their own local clusters. Disable Ray's uv runtime-env
# hook by default because it can inspect sandbox parent processes and fail before
# the test logic runs.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

_PR_FAST_DIR = Path(__file__).resolve().parent


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _pytest_invocation_paths() -> list[Path]:
    paths = []
    for arg in sys.argv[1:]:
        if not arg or arg.startswith("-"):
            continue
        path_arg = arg.split("::", 1)[0]
        path = Path(path_arg)
        if path.exists():
            paths.append(path.resolve())
    return paths


def _should_install_import_stubs() -> bool:
    paths = _pytest_invocation_paths()
    return bool(paths) and all(_is_relative_to(path, _PR_FAST_DIR) for path in paths)


def _new_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.setdefault("__all__", [])
    return module


def _install_dataset_stubs() -> None:
    if "xtuner.v1.datasets.config" in sys.modules:
        return

    datasets_pkg = _new_module("xtuner.v1.datasets")
    datasets_pkg.__path__ = []

    class DataloaderConfig(BaseModel):
        model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

        def build(self, *args: Any, **kwargs: Any):
            raise RuntimeError("PR-fast tests should provide fake dataloaders explicitly.")

    class Dataloader:
        pass

    config_mod = _new_module("xtuner.v1.datasets.config")
    config_mod.DataloaderConfig = DataloaderConfig
    config_mod.__all__ = ["DataloaderConfig"]

    dataloader_mod = _new_module("xtuner.v1.datasets.dataloader")
    dataloader_mod.Dataloader = Dataloader
    dataloader_mod.__all__ = ["Dataloader"]

    datasets_pkg.DataloaderConfig = DataloaderConfig
    datasets_pkg.Dataloader = Dataloader

    sys.modules.setdefault("xtuner.v1.datasets", datasets_pkg)
    sys.modules.setdefault("xtuner.v1.datasets.config", config_mod)
    sys.modules.setdefault("xtuner.v1.datasets.dataloader", dataloader_mod)


def _install_train_trainer_stub() -> None:
    if "xtuner.v1.train.trainer" in sys.modules:
        return

    class LoadCheckpointConfig(BaseModel):
        model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    class ResumeConfig(BaseModel):
        model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    class TrainerConfig(BaseModel):
        model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    class Trainer:
        pass

    class XTunerMeta:
        @classmethod
        def build(cls, *args: Any, **kwargs: Any) -> "XTunerMeta":
            return cls()

    trainer_mod = _new_module("xtuner.v1.train.trainer")
    trainer_mod.LoadCheckpointConfig = LoadCheckpointConfig
    trainer_mod.ResumeConfig = ResumeConfig
    trainer_mod.Trainer = Trainer
    trainer_mod.TrainerConfig = TrainerConfig
    trainer_mod.XTunerMeta = XTunerMeta
    trainer_mod.__all__ = [
        "LoadCheckpointConfig",
        "ResumeConfig",
        "Trainer",
        "TrainerConfig",
        "XTunerMeta",
    ]

    sys.modules.setdefault("xtuner.v1.train.trainer", trainer_mod)


def _install_rl_trainer_worker_stubs() -> None:
    if "xtuner.v1.rl.trainer.worker" in sys.modules:
        return

    trainer_pkg = _new_module("xtuner.v1.rl.trainer")
    trainer_pkg.__path__ = []

    class TrainingController:
        pass

    class WorkerConfig(BaseModel):
        model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    class TrainingWorker:
        pass

    controller_mod = _new_module("xtuner.v1.rl.trainer.controller")
    controller_mod.TrainingController = TrainingController
    controller_mod.ColateItem = object
    controller_mod.__all__ = ["TrainingController", "ColateItem"]

    worker_mod = _new_module("xtuner.v1.rl.trainer.worker")
    worker_mod.TrainingWorker = TrainingWorker
    worker_mod.WorkerConfig = WorkerConfig
    worker_mod.WorkerInputItem = dict
    worker_mod.WorkerLogItem = dict
    worker_mod.WorkerTrainLogItem = dict
    worker_mod.__all__ = [
        "TrainingWorker",
        "WorkerConfig",
        "WorkerInputItem",
        "WorkerLogItem",
        "WorkerTrainLogItem",
    ]

    trainer_pkg.TrainingController = TrainingController
    trainer_pkg.WorkerConfig = WorkerConfig
    trainer_pkg.TrainingWorker = TrainingWorker
    trainer_pkg.WorkerInputItem = dict
    trainer_pkg.WorkerLogItem = dict
    trainer_pkg.WorkerTrainLogItem = dict
    trainer_pkg.ColateItem = object

    sys.modules.setdefault("xtuner.v1.rl.trainer", trainer_pkg)
    sys.modules.setdefault("xtuner.v1.rl.trainer.controller", controller_mod)
    sys.modules.setdefault("xtuner.v1.rl.trainer.worker", worker_mod)


# These stubs are only safe for isolated PR-fast runs. Full RL collection also
# imports real smoke and integration tests, which require the real modules.
if _should_install_import_stubs():
    _install_dataset_stubs()
    _install_train_trainer_stub()
    _install_rl_trainer_worker_stubs()
