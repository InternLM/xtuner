"""CPU checks for experiment tracking through the ordinary Trainer lifecycle."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from xtuner.v1._writer import SwanLabConfig
from xtuner.v1.model.utils.misc import ModelForwardExtraLogInfo
from xtuner.v1.train import trainer as trainer_module
from xtuner.v1.train.trainer import HooksConfig, LoadCheckpointConfig, Trainer, TrainerConfig


@pytest.fixture
def sdk(monkeypatch):
    sdk = SimpleNamespace(runs=[])

    def init(**kwargs):
        run = SimpleNamespace(
            id=kwargs.get("id") or f"sft-run-{len(sdk.runs)}",
            mode="offline",
            log=Mock(),
            finish=Mock(),
        )
        sdk.runs.append(run)
        sdk.run = run
        return run

    sdk.init = Mock(side_effect=init)
    monkeypatch.setitem(sys.modules, "swanlab", sdk)
    return sdk


def test_all_ranks_create_independent_runs_in_one_group(tmp_path, monkeypatch, sdk):
    trainer = Trainer.__new__(Trainer)
    trainer._meta = SimpleNamespace(latest_exp=SimpleNamespace(exp_dir=str(tmp_path / "exp")))
    trainer._load_checkpoint_cfg = LoadCheckpointConfig()
    for rank in (1, 0):
        monkeypatch.setattr(trainer_module, "get_rank", lambda: rank)
        writer = trainer._init_tracker("swanlab", tmp_path / f"rank{rank}", SwanLabConfig(mode="offline"))
        writer.add_scalar(tag="loss", scalar_value=0.5, global_step=2)
        writer.close()

    assert sdk.init.call_count == 2
    assert len({run.id for run in sdk.runs}) == 2
    calls = sdk.init.call_args_list
    assert calls[0].kwargs["group"] == calls[1].kwargs["group"]
    assert calls[0].kwargs["group"]
    assert [call.kwargs["experiment_name"].rsplit("-", 1)[-1] for call in calls] == ["rank1", "rank0"]
    for rank, run in zip((1, 0), sdk.runs):
        run.log.assert_called_once_with({"loss": 0.5}, step=2)
        run.finish.assert_called_once()
        assert json.loads((tmp_path / f"rank{rank}/swanlab_run.json").read_text())["id"] == run.id
    assert TrainerConfig.model_fields["exp_tracker"].default == "jsonl"


def test_explicit_id_and_name_distinguish_ranks(tmp_path, monkeypatch, sdk):
    trainer = Trainer.__new__(Trainer)
    trainer._meta = SimpleNamespace(latest_exp=SimpleNamespace(exp_dir=str(tmp_path / "exp")))
    trainer._load_checkpoint_cfg = LoadCheckpointConfig()
    config = SwanLabConfig(group="training", experiment_name="sft", run_id="manual-run", resume="must")
    for rank in (0, 1):
        monkeypatch.setattr(trainer_module, "get_rank", lambda: rank)
        trainer._init_tracker("swanlab", tmp_path / f"rank{rank}", config).close()

    assert [call.kwargs["id"] for call in sdk.init.call_args_list] == ["manual-run", "manual-run-rank1"]
    assert [call.kwargs["experiment_name"] for call in sdk.init.call_args_list] == ["sft-rank0", "sft-rank1"]
    assert all(call.kwargs["group"] == "training" for call in sdk.init.call_args_list)
    assert config.run_id == "manual-run"
    assert config.experiment_name == "sft"


@pytest.mark.parametrize("rank", [0, 2])
def test_legacy_checkpoint_identity_is_only_reused_by_rank_zero(tmp_path, monkeypatch, sdk, rank):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "train_state.json").write_text(
        json.dumps({"swanlab_run": {"id": "original-run", "project": "sft", "group": "original-group"}})
    )
    monkeypatch.setattr(trainer_module, "get_rank", lambda: rank)
    trainer = Trainer.__new__(Trainer)
    trainer._meta = SimpleNamespace(latest_exp=SimpleNamespace(exp_dir=str(tmp_path / "new-exp")))
    trainer._load_checkpoint_cfg = LoadCheckpointConfig(checkpoint_path=checkpoint)
    trainer._init_tracker("swanlab", tmp_path / f"rank{rank}").close()

    assert sdk.init.call_args.kwargs["id"] == ("original-run" if rank == 0 else None)
    assert sdk.init.call_args.kwargs["group"] == "original-group"


@pytest.mark.parametrize("backend", ["tensorboard", "jsonl"])
def test_existing_backends_keep_per_rank_logs(tmp_path, monkeypatch, backend):
    monkeypatch.setitem(sys.modules, "swanlab", None)
    monkeypatch.setattr(trainer_module, "get_rank", lambda: 1)
    trainer = Trainer.__new__(Trainer)
    writer = trainer._init_tracker(backend, tmp_path / "rank1")
    writer.add_scalar(tag="loss", scalar_value=0.25, global_step=3)
    writer.close()

    if backend == "tensorboard":
        event = EventAccumulator(str(tmp_path / "rank1")).Reload().Scalars("loss")[0]
        assert (event.step, event.value) == (3, 0.25)
    else:
        assert json.loads((tmp_path / "rank1/tracker.jsonl").read_text()) == {"loss": 0.25, "step": 3}


@pytest.fixture
def make_trainer(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer_module, "get_rank", lambda: 0)
    monkeypatch.setattr(trainer_module.dist, "barrier", lambda: None)
    monkeypatch.setattr(trainer_module, "ProberList", Mock())
    device = Mock()
    device.max_memory_allocated.return_value = 0
    device.max_memory_reserved.return_value = 0
    monkeypatch.setattr(trainer_module, "DEVICE_MODULE", device)

    def build(name, total_steps, checkpoint=None, *, auto_resume=False, rank=0):
        # Exercise fit, logging and checkpoint I/O with only the engine and
        # distributed resources replaced; no model assets or GPU are needed.
        trainer = Trainer.__new__(Trainer)
        monkeypatch.setattr(trainer_module, "get_rank", lambda: rank)
        trainer._init_async_save_resources(async_hf_export=False, async_checkpoint=False)
        trainer._work_dir = tmp_path / name
        trainer._work_dir.mkdir(exist_ok=True)
        trainer._meta = SimpleNamespace(
            latest_exp=SimpleNamespace(
                exp_dir=str(trainer._work_dir / "exp"),
                checkpoint_list=[],
                latest_checkpoint=checkpoint,
                history=[{}],
            ),
            model_dump_json=Mock(return_value="{}"),
        )
        trainer.logger = Mock()
        trainer._load_checkpoint_cfg = trainer._resolve_load_checkpoint_cfg(
            auto_resume, LoadCheckpointConfig(checkpoint_path=None if auto_resume else checkpoint)
        )
        trainer._exp_tracker = trainer._init_tracker(
            "swanlab", trainer.exp_dir / f"logs/exp_tracking/rank{rank}", SwanLabConfig(project="sft", mode="offline")
        )
        trainer._cur_step = 0
        trainer._cur_epoch = 1
        trainer._total_step = total_steps
        trainer._total_epoch = None
        trainer._local_total_consumed_tokens = 0
        trainer._init_total_tokens = 0
        trainer._train_time = 0
        trainer._train_time_offset = 0
        trainer._checkpoint_interval = 1
        trainer._checkpoint_maxkeep = None
        trainer._snapshot_interval = None
        trainer._check_health_interval = None
        trainer._hf_interval = None
        trainer._trainer_cfg = None
        trainer._metrics_recorder = None
        trainer._profile_time = False
        trainer._profile_memory = False
        trainer._offload_optimizer_first_step = False
        trainer._do_clip = True
        trainer._grad_norm_dtype = torch.float32
        trainer.hooks_config = HooksConfig()
        trainer._dataloader = Mock()
        trainer._dataloader.get_state_dict.return_value = {}
        trainer._reduce_number_across_rank = lambda value: value
        trainer._data_iter = lambda: (None for _ in range(trainer.cur_step, total_steps))
        trainer._prepare_model_input = Mock(return_value=[])
        trainer._engine = Mock()
        trainer._engine.train_step.return_value = {
            "total_loss": 0.5 + rank,
            "step_consumed_tokens": 8,
            "step_seqlen_tokens": 8,
            "efficient_attn_ratio": 1.0,
            "img_efficient_attn_ratio": 0.0,
            "logs_info": {"reduced_llm_loss": 0.5},
            "extra_info": ModelForwardExtraLogInfo(),
        }
        trainer._engine.clip_grad_norm.return_value = torch.tensor(1.0)
        trainer._engine.save_dcp.side_effect = lambda weights_dir: weights_dir.mkdir(parents=True, exist_ok=True)
        trainer._lr_scheduler = Mock()
        trainer._lr_scheduler.get_last_lr.return_value = [0.001]
        trainer._lr_scheduler.state_dict.return_value = {}
        if checkpoint is not None:
            trainer._load_checkpoint()
        return trainer

    return build


@pytest.mark.parametrize("auto_resume", [False, True])
def test_fit_saves_run_identity_and_resumes_in_a_new_directory(make_trainer, sdk, auto_resume):
    identities = []
    for rank in (0, 1):
        original = make_trainer("original", 1, rank=rank)
        original.fit()
        checkpoint = original.checkpoint_dir / "ckpt-step-1"
        identity = json.loads((checkpoint / f"swanlab/rank{rank}.json").read_text())
        assert identity["id"] == sdk.run.id
        identities.append(identity)
    state = json.loads((checkpoint / "train_state.json").read_text())
    assert state["swanlab_run"] == identities[0]
    assert identities[0]["id"] != identities[1]["id"]
    assert identities[0]["group"] == identities[1]["group"] == "original-exp"

    for rank in (0, 1):
        resumed = make_trainer("resumed", 2, checkpoint, auto_resume=auto_resume, rank=rank)
        assert sdk.init.call_args.kwargs["id"] == identities[rank]["id"]
        assert sdk.init.call_args.kwargs["group"] == identities[rank]["group"]
        assert sdk.init.call_args.kwargs["resume"] == "allow"
        assert resumed.cur_step == 1
        resumed.fit()

    for rank in (0, 1):
        runs = [run for run in sdk.runs if run.id == identities[rank]["id"]]
        calls = [call for run in runs for call in run.log.call_args_list]
        assert [call.kwargs["step"] for call in calls] == [1, 2]
        assert all(call.args[0]["loss/local_loss"] == 0.5 + rank for call in calls)
        assert all(call.args[0]["loss/reduced_llm_loss"] == 0.5 for call in calls)
        assert all(run.finish.call_count == 1 for run in runs)
    assert json.loads((resumed.checkpoint_dir / "ckpt-step-2/train_state.json").read_text())["cur_step"] == 2


def test_fit_closes_the_run_on_training_failure(make_trainer, sdk):
    trainer = make_trainer("failed", 1)
    trainer._engine.train_step.side_effect = RuntimeError("training failed")
    with pytest.raises(RuntimeError, match="training failed"):
        trainer.fit()
    sdk.run.finish.assert_called_once()
