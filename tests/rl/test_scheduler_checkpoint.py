"""Worker checkpoint wiring with real CPU optimizers and LR schedules."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from xtuner.v1.config.optim import LRConfig
from xtuner.v1.rl.trainer.actor_worker import ActorWorker
from xtuner.v1.rl.trainer.critic_worker import CriticWorker
from xtuner.v1.rl.trainer.schedule import build_lr_scheduler, resume_lr_scheduler, save_lr_scheduler
from xtuner.v1.rl.trainer.worker import TrainingWorker
from xtuner.v1.train.trainer import LoadCheckpointConfig


def scheduled(kind="linear", warmup=2, total=12):
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.ones(1))], lr=1e-3)
    scheduler = build_lr_scheduler(optimizer, LRConfig(lr_type=kind, warmup_ratio=warmup, lr_min=1e-5), total)
    return optimizer, scheduler


def advance(optimizer, scheduler, count):
    values = []
    for _ in range(count):
        optimizer.step()
        scheduler.step()
        values.append(optimizer.param_groups[0]["lr"])
    return values


@pytest.mark.parametrize("kind", ["constant", "linear", "cosine"])
@pytest.mark.parametrize("checkpoint_step", [1, 5])
def test_resume_matches_uninterrupted_schedule(tmp_path, kind, checkpoint_step):
    opt, sched = scheduled(kind)
    advance(opt, sched, checkpoint_step)
    opt_state = deepcopy(opt.state_dict())
    save_lr_scheduler(sched, tmp_path)
    expected = advance(opt, sched, 4)
    resumed_opt, resumed_sched = scheduled(kind)
    resumed_opt.load_state_dict(opt_state)
    resume_lr_scheduler(resumed_sched, tmp_path, load_scheduler=True)
    assert advance(resumed_opt, resumed_sched, 4) == pytest.approx(expected)


def test_missing_schedule_is_explicit_and_reset_restarts(tmp_path):
    opt, sched = scheduled()
    with pytest.raises(FileNotFoundError, match="load_scheduler=False"):
        resume_lr_scheduler(sched, tmp_path, load_scheduler=True)
    advance(opt, sched, 5)
    save_lr_scheduler(sched, tmp_path)
    other_opt, other_sched = scheduled(total=24)
    other_opt.load_state_dict(opt.state_dict())
    resume_lr_scheduler(other_sched, tmp_path, load_scheduler=False)
    assert other_opt.param_groups[0]["lr"] == other_sched.get_last_lr()[0] == 0.0005


@pytest.mark.parametrize("load_optimizer_args", [False, True])
def test_actor_and_critic_checkpoint_wiring(tmp_path, load_optimizer_args):
    saved_optimizers = {}

    class Engine:
        def __init__(self, optimizer):
            self.optimizer = optimizer

        def save_dcp(self, weights_dir, save_optimizer):
            weights_dir.mkdir(parents=True)
            (weights_dir / ".metadata").touch()
            saved_optimizers[str(weights_dir)] = deepcopy(self.optimizer.state_dict())

        def load_dcp(self, weights_dir, load_states, load_args):
            if load_args:
                self.optimizer.load_state_dict(saved_optimizers[str(weights_dir)])

    def worker():
        host = TrainingWorker.__new__(TrainingWorker)
        actor = ActorWorker.__new__(ActorWorker)
        critic = CriticWorker.__new__(CriticWorker)
        for role, kind in [(actor, "linear"), (critic, "cosine")]:
            optimizer, role.lr_scheduler = scheduled(kind)
            role.engine = Engine(optimizer)
        actor._sft_dataloader = None
        host.actor, host.critic = actor, critic
        host.logger = Mock()
        return host

    host = worker()
    advance(host.actor.engine.optimizer, host.actor.lr_scheduler, 3)
    advance(host.critic.engine.optimizer, host.critic.lr_scheduler, 5)
    host.save(tmp_path)
    assert (tmp_path / "actor/lr_scheduler.json").exists()
    assert (tmp_path / "critic/lr_scheduler.json").exists()
    expected_actor = advance(host.actor.engine.optimizer, host.actor.lr_scheduler, 2)
    expected_critic = advance(host.critic.engine.optimizer, host.critic.lr_scheduler, 2)
    restored = worker()
    restored.resume(LoadCheckpointConfig(checkpoint_path=tmp_path, load_optimizer_args=load_optimizer_args))
    assert advance(restored.actor.engine.optimizer, restored.actor.lr_scheduler, 2) == pytest.approx(expected_actor)
    assert advance(restored.critic.engine.optimizer, restored.critic.lr_scheduler, 2) == pytest.approx(expected_critic)


def test_role_saves_scheduler_without_optimizer(tmp_path):
    role = ActorWorker.__new__(ActorWorker)
    _, role.lr_scheduler = scheduled()
    role.engine = SimpleNamespace(save_dcp=Mock())
    role.save(tmp_path, no_save_optimizer=True)
    role.engine.save_dcp.assert_called_once_with(weights_dir=tmp_path, save_optimizer=False)
    assert (tmp_path / "lr_scheduler.json").exists()


def test_scheduler_restore_without_optimizer_arguments(tmp_path):
    optimizer, scheduler = scheduled()
    advance(optimizer, scheduler, 6)
    save_lr_scheduler(scheduler, tmp_path)
    other, resumed = scheduled()
    resume_lr_scheduler(resumed, tmp_path, load_scheduler=True)
    assert other.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    assert advance(other, resumed, 2) == pytest.approx(advance(optimizer, scheduler, 2))
