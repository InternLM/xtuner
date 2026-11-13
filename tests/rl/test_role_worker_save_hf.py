"""HF export selects one resident role and restores placement after failures."""

from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from xtuner.v1.rl.trainer.role_worker import RoleName, RoleWorker
from xtuner.v1.rl.trainer.train_batch import TrainBatch
from xtuner.v1.rl.trainer.worker import TrainingWorker, WorkerLogItem


class _SaveRole(RoleWorker):
    def fit(self, batch: TrainBatch) -> WorkerLogItem:
        raise NotImplementedError


def _role(name: RoleName, *, model: bool, optimizer: bool) -> _SaveRole:
    role = _SaveRole.__new__(_SaveRole)
    role.role_name = name
    role.device = torch.device("cuda")
    role.logger = Mock()
    role._model_on_device = model
    role._optimizer_on_device = optimizer
    role.engine = Mock()
    # No tensors or CUDA runtime are needed for placement bookkeeping.
    role.engine.put_model_to_device.return_value = False
    role.engine.put_optimizer_to_device.return_value = False
    return role


class TestRoleWorkerSaveHf:
    def test_offloaded_role_rejects_save_before_nccl_collectives(self, tmp_path: Path) -> None:
        role = _role("actor", model=False, optimizer=False)
        with pytest.raises(RuntimeError, match="actor model is not on the compute device"):
            role.save_hf(tmp_path, torch.bfloat16)
        role.engine.save_hf.assert_not_called()

    def test_resident_role_saves_without_moving_optimizer(self, tmp_path: Path) -> None:
        role = _role("actor", model=True, optimizer=False)
        role.save_hf(tmp_path, torch.float32)
        role.engine.save_hf.assert_called_once_with(hf_dir=tmp_path, save_dtype=torch.float32)
        role.engine.put_optimizer_to_device.assert_not_called()


class TestTrainingWorkerSaveHf:
    @pytest.mark.parametrize("model_role", [None, "actor", "critic"])
    @pytest.mark.parametrize("optimizer_role", [None, "actor", "critic"])
    @pytest.mark.parametrize("fail_role", [None, "actor", "critic"])
    def test_sequential_saves_restore_residency(
        self,
        tmp_path: Path,
        model_role: RoleName | None,
        optimizer_role: RoleName | None,
        fail_role: RoleName | None,
    ) -> None:
        worker = TrainingWorker.__new__(TrainingWorker)
        worker.actor = _role("actor", model=model_role == "actor", optimizer=optimizer_role == "actor")
        worker.critic = _role("critic", model=model_role == "critic", optimizer=optimizer_role == "critic")
        roles = worker._roles()
        saved: list[str] = []

        def move_model(name: str, device: torch.device | str) -> bool:
            if device != "cpu":
                assert not any(role.model_on_device for other, role in roles.items() if other != name)
            return False

        def save(name: str, hf_dir: Path, save_dtype: torch.dtype) -> None:
            assert hf_dir == tmp_path / name
            assert save_dtype == torch.float32
            assert roles[name].model_on_device
            assert not any(role.optimizer_on_device for role in roles.values())
            saved.append(name)
            if name == fail_role:
                raise RuntimeError("HF write failed")

        for name, role in roles.items():
            role.engine.put_model_to_device.side_effect = lambda device, name=name: move_model(name, device)
            role.engine.save_hf.side_effect = lambda hf_dir, save_dtype, name=name: save(name, hf_dir, save_dtype)

        if fail_role is None:
            worker.save_hf(str(tmp_path), torch.float32)
        else:
            with pytest.raises(RuntimeError, match="HF write failed"):
                worker.save_hf(str(tmp_path), torch.float32)

        assert saved == (["actor"] if fail_role == "actor" else ["actor", "critic"])
        for name, role in roles.items():
            assert role.model_on_device == (name == model_role)
            assert role.optimizer_on_device == (name == optimizer_role)

    def test_failed_critic_onload_restores_actor_residency(self, tmp_path: Path) -> None:
        worker = TrainingWorker.__new__(TrainingWorker)
        worker.actor = _role("actor", model=True, optimizer=True)
        worker.critic = _role("critic", model=False, optimizer=False)
        worker.critic.engine.put_model_to_device.side_effect = RuntimeError("onload failed")

        with pytest.raises(RuntimeError, match="onload failed"):
            worker.save_hf(str(tmp_path))

        worker.actor.engine.save_hf.assert_called_once()
        worker.critic.engine.save_hf.assert_not_called()
        assert worker.actor.model_on_device and worker.actor.optimizer_on_device
        assert not worker.critic.model_on_device and not worker.critic.optimizer_on_device

    def test_actor_only_save_restores_fully_offloaded_state(self, tmp_path: Path) -> None:
        worker = TrainingWorker.__new__(TrainingWorker)
        worker.actor = _role("actor", model=False, optimizer=False)
        worker.critic = None

        def save(hf_dir: Path, save_dtype: torch.dtype) -> None:
            assert worker.actor.model_on_device
            assert not worker.actor.optimizer_on_device
            assert hf_dir == tmp_path / "actor"
            assert save_dtype == torch.bfloat16

        worker.actor.engine.save_hf.side_effect = save
        worker.save_hf(str(tmp_path))

        worker.actor.engine.save_hf.assert_called_once()
        assert not worker.actor.model_on_device
        assert not worker.actor.optimizer_on_device
        worker.actor.engine.put_optimizer_to_device.assert_not_called()
