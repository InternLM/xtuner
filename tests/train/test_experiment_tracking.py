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
    run = SimpleNamespace(id="sft-run", mode="offline", log=Mock(), finish=Mock())
    sdk = SimpleNamespace(init=Mock(return_value=run), run=run)
    monkeypatch.setitem(sys.modules, "swanlab", sdk)
    return sdk


def test_only_rank_zero_creates_a_swanlab_run(tmp_path, monkeypatch, sdk):
    trainer = Trainer.__new__(Trainer)
    trainer._load_checkpoint_cfg = LoadCheckpointConfig()
    for rank in (1, 0):
        monkeypatch.setattr(trainer_module, "get_rank", lambda: rank)
        writer = trainer._init_tracker("swanlab", tmp_path / f"rank{rank}", SwanLabConfig(mode="offline"))
        writer.add_scalar(tag="loss", scalar_value=0.5, global_step=2)
        writer.close()

    sdk.init.assert_called_once()
    sdk.run.log.assert_called_once_with({"loss": 0.5}, step=2)
    sdk.run.finish.assert_called_once()
    assert not (tmp_path / "rank1").exists()
    assert TrainerConfig.model_fields["exp_tracker"].default == "jsonl"


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

    def build(name, total_steps, checkpoint=None, *, auto_resume=False):
        # Exercise fit, logging and checkpoint I/O with only the engine and
        # distributed resources replaced; no model assets or GPU are needed.
        trainer = Trainer.__new__(Trainer)
        trainer._init_async_save_resources(async_hf_export=False, async_checkpoint=False)
        trainer._work_dir = tmp_path / name
        trainer._work_dir.mkdir()
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
            "swanlab", trainer.exp_dir / "logs/exp_tracking/rank0", SwanLabConfig(project="sft", mode="offline")
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
            "total_loss": 0.5,
            "step_consumed_tokens": 8,
            "step_seqlen_tokens": 8,
            "efficient_attn_ratio": 1.0,
            "img_efficient_attn_ratio": 0.0,
            "logs_info": {"reduced_llm_loss": 0.5},
            "extra_info": ModelForwardExtraLogInfo(),
        }
        trainer._engine.clip_grad_norm.return_value = torch.tensor(1.0)
        trainer._engine.save_dcp.side_effect = lambda weights_dir: weights_dir.mkdir(parents=True)
        trainer._lr_scheduler = Mock()
        trainer._lr_scheduler.get_last_lr.return_value = [0.001]
        trainer._lr_scheduler.state_dict.return_value = {}
        if checkpoint is not None:
            trainer._load_checkpoint()
        return trainer

    return build


@pytest.mark.parametrize("auto_resume", [False, True])
def test_fit_saves_run_identity_and_resumes_in_a_new_directory(make_trainer, sdk, auto_resume):
    original = make_trainer("original", 1)
    original.fit()
    checkpoint = original.checkpoint_dir / "ckpt-step-1"
    state = json.loads((checkpoint / "train_state.json").read_text())
    assert state["swanlab_run"] == {"id": "sft-run", "project": "sft", "workspace": None}

    resumed = make_trainer("resumed", 2, checkpoint, auto_resume=auto_resume)
    assert sdk.init.call_args.kwargs["id"] == "sft-run"
    assert sdk.init.call_args.kwargs["resume"] == "allow"
    assert resumed.cur_step == 1
    resumed.fit()

    assert [call.kwargs["step"] for call in sdk.run.log.call_args_list] == [1, 2]
    assert all(call.args[0]["loss/reduced_llm_loss"] == 0.5 for call in sdk.run.log.call_args_list)
    assert sdk.run.finish.call_count == 2
    assert json.loads((resumed.checkpoint_dir / "ckpt-step-2/train_state.json").read_text())["cur_step"] == 2


def test_fit_closes_the_run_on_training_failure(make_trainer, sdk):
    trainer = make_trainer("failed", 1)
    trainer._engine.train_step.side_effect = RuntimeError("training failed")
    with pytest.raises(RuntimeError, match="training failed"):
        trainer.fit()
    sdk.run.finish.assert_called_once()
