import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from xtuner.v1._writer import SwanLabConfig, SwanLabWriter, get_writer
from xtuner.v1._writer.swanlab_writer import RUN_INFO_FILE


@pytest.fixture
def sdk(monkeypatch):
    run = SimpleNamespace(id="test-run-id", mode="offline", log=Mock(), finish=Mock())
    sdk = SimpleNamespace(init=Mock(return_value=run), run=run)
    monkeypatch.setitem(sys.modules, "swanlab", sdk)
    return sdk


def test_metrics_keep_names_and_independent_step_axes(tmp_path, sdk):
    writer = get_writer(
        writer_type="swanlab",
        log_dir=tmp_path,
        swanlab_config=SwanLabConfig(project="rl", mode="offline", config={"batch_size": 48}),
    )
    writer.add_scalars(tag_scalar_dict={"train_metrics/worker_0/loss": 0.8}, global_step=7)
    writer.add_scalars(tag_scalar_dict={"response/rewards/mean": 0.5}, global_step=3)
    writer.add_scalar(tag="time/step", scalar_value=12.0, global_step=3)
    writer.close()
    writer.close()
    writer.add_scalar(tag="ignored", scalar_value=1.0, global_step=8)

    assert sdk.run.log.call_args_list == [
        (({"train_metrics/worker_0/loss": 0.8},), {"step": 7}),
        (({"response/rewards/mean": 0.5},), {"step": 3}),
        (({"time/step": 12.0},), {"step": 3}),
    ]
    sdk.run.finish.assert_called_once()
    assert sdk.init.call_args.kwargs["config"] == {"batch_size": 48}
    assert json.loads((tmp_path / RUN_INFO_FILE).read_text()) == writer.run_info


def test_resume_uses_persisted_run_id(tmp_path, sdk):
    original = SwanLabWriter(tmp_path, config=SwanLabConfig(project="rl", workspace="team", group="training"))
    original.close()
    resumed = SwanLabWriter(tmp_path)
    resumed.close()

    assert sdk.init.call_args.kwargs["id"] == "test-run-id"
    assert sdk.init.call_args.kwargs["resume"] == "allow"
    assert sdk.init.call_args.kwargs["project"] == "rl"
    assert sdk.init.call_args.kwargs["workspace"] == "team"
    assert sdk.init.call_args.kwargs["group"] == "training"


def test_checkpoint_run_identity_survives_a_new_log_directory(tmp_path, sdk):
    saved = {"id": "checkpoint-run", "project": "rl", "workspace": "team", "group": "training"}
    writer = SwanLabWriter(tmp_path / "new-exp", resume_run=saved)
    writer.close()

    assert sdk.init.call_args.kwargs["id"] == "checkpoint-run"
    assert sdk.init.call_args.kwargs["resume"] == "allow"
    assert sdk.init.call_args.kwargs["project"] == "rl"
    assert sdk.init.call_args.kwargs["workspace"] == "team"
    assert sdk.init.call_args.kwargs["group"] == "training"


def test_online_run_persists_resolved_project_and_workspace(tmp_path, sdk):
    sdk.run.mode = "online"
    sdk.run.path = "/resolved-team/resolved-project/test-run-id"
    writer = SwanLabWriter(tmp_path)
    writer.close()

    assert writer.run_info == {
        "id": "test-run-id",
        "project": "resolved-project",
        "workspace": "resolved-team",
        "group": None,
    }


def test_resume_never_starts_a_new_run(tmp_path, sdk):
    saved = {"id": "previous-run", "project": "old-project", "workspace": "old-team"}
    (tmp_path / RUN_INFO_FILE).write_text(json.dumps(saved))
    writer = SwanLabWriter(tmp_path, config=SwanLabConfig(resume="never"), resume_run=saved)
    writer.close()

    assert sdk.init.call_args.kwargs["id"] is None
    assert sdk.init.call_args.kwargs["resume"] == "never"


def test_explicit_id_overrides_checkpoint_identity(tmp_path, sdk):
    writer = SwanLabWriter(
        tmp_path,
        config=SwanLabConfig(project="new-project", run_id="explicit-run", resume="must"),
        resume_run={"id": "previous-run", "project": "old-project", "workspace": None},
    )
    writer.close()

    assert sdk.init.call_args.kwargs["id"] == "explicit-run"
    assert sdk.init.call_args.kwargs["resume"] == "must"
    assert sdk.init.call_args.kwargs["project"] == "new-project"


def test_resuming_in_a_different_project_fails_before_init(tmp_path, sdk):
    with pytest.raises(ValueError, match="different project"):
        SwanLabWriter(
            tmp_path,
            config=SwanLabConfig(project="different"),
            resume_run={"id": "previous-run", "project": "original", "workspace": None},
        )
    sdk.init.assert_not_called()


def test_identity_save_failure_is_non_fatal(tmp_path, sdk, monkeypatch, caplog):
    monkeypatch.setattr(type(tmp_path), "replace", Mock(side_effect=OSError("storage unavailable")))
    writer = SwanLabWriter(tmp_path)
    writer.close()

    assert "Could not persist SwanLab local run identity" in caplog.text
    sdk.run.finish.assert_called_once()


def test_checkpoint_identity_is_persisted_by_the_writer(tmp_path, sdk):
    checkpoint = tmp_path / "checkpoint"
    writer = SwanLabWriter(tmp_path / "logs", rank=2)
    writer.persist_state(checkpoint)
    writer.close()

    assert json.loads((checkpoint / "swanlab/rank2.json").read_text()) == writer.run_info


def test_missing_optional_sdk_has_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "swanlab", None)
    with pytest.raises(ImportError, match=r"xtuner\[swanlab\]"):
        get_writer(writer_type="swanlab", log_dir=tmp_path)


def test_real_swanlab_sdk_offline_contract(tmp_path):
    pytest.importorskip("swanlab")
    writer = SwanLabWriter(
        tmp_path,
        config=SwanLabConfig(
            project="xtuner-writer-test",
            experiment_name="offline-contract",
            mode="offline",
        ),
    )
    writer.add_scalars(tag_scalar_dict={"train_metrics/loss": 0.8}, global_step=2)
    writer.add_scalars(tag_scalar_dict={"response/reward": 0.5}, global_step=1)
    writer.close()

    assert json.loads((tmp_path / RUN_INFO_FILE).read_text()) == writer.run_info


def test_tensorboard_remains_usable_without_swanlab(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "swanlab", None)
    writer = get_writer(writer_type="tensorboard", log_dir=tmp_path)
    writer.add_scalar(tag="loss", scalar_value=0.25, global_step=2)
    writer.add_scalars(tag_scalar_dict={"train_metrics/worker_0/loss": 0.75}, global_step=7)
    writer.add_scalars(tag_scalar_dict={"response/rewards/mean": 0.5}, global_step=3)
    writer.close()
    accumulator = EventAccumulator(str(tmp_path)).Reload()
    event = accumulator.Scalars("loss")[0]
    assert (event.step, event.value) == (2, 0.25)
    assert accumulator.Scalars("train_metrics/worker_0/loss")[0].step == 7
    assert accumulator.Scalars("response/rewards/mean")[0].step == 3
