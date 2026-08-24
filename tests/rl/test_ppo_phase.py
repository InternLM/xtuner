"""Tests for the PPO actor/critic phase machine in the training worker.

Actor and critic are full models that cannot both hold accelerator memory in a
colocated run, so the worker swaps one for the other. These tests pin the
transitions using a fake engine, with no real model, Ray or accelerator.
"""

import unittest
from unittest.mock import MagicMock, patch

from xtuner.v1.rl.trainer.worker import PPOPhase, TrainingWorker


# `DEVICE_MODULE` is the accelerator module; on a CPU-only host it resolves to
# `torch.cpu`, which has no memory-management API. The phase machine's logic is
# device independent, so stub it out.
_FAKE_DEVICE_MODULE = MagicMock()
_FAKE_DEVICE_MODULE.memory_allocated.return_value = 0
_FAKE_DEVICE_MODULE.memory_reserved.return_value = 0
_DEVICE_MODULE = patch("xtuner.v1.rl.trainer.worker.DEVICE_MODULE", _FAKE_DEVICE_MODULE)
_DEVICE = patch("xtuner.v1.rl.trainer.worker.DEVICE", "cuda")


def setUpModule() -> None:
    _DEVICE_MODULE.start()
    _DEVICE.start()


def tearDownModule() -> None:
    _DEVICE_MODULE.stop()
    _DEVICE.stop()


class _FakeEngine:
    """Records device placement so swaps can be asserted on."""

    def __init__(self, name: str, log: list[tuple[str, str, str]]):
        self.name = name
        self.log = log
        self.model_device = "cpu"
        self.optimizer_device = "cpu"

    def put_model_to_device(self, device):
        self.model_device = str(device)
        self.log.append((self.name, "model", str(device)))
        return True

    def put_optimizer_to_device(self, device):
        self.optimizer_device = str(device)
        self.log.append((self.name, "optimizer", str(device)))
        return True

    def save_hf(self, hf_dir, save_dtype):
        self.log.append((self.name, "save_hf", str(hf_dir)))


def _worker(*, with_critic: bool) -> tuple[TrainingWorker, list[tuple[str, str, str]]]:
    worker = TrainingWorker.__new__(TrainingWorker)
    log: list[tuple[str, str, str]] = []
    worker._engine = _FakeEngine("actor", log)
    worker._critic_engine = _FakeEngine("critic", log) if with_critic else None
    worker._ppo_phase = PPOPhase.ALL_OFFLOADED if with_critic else PPOPhase.ACTOR_READY
    worker.logger = MagicMock()
    worker.rank = 0
    return worker, log


class TestPhaseTransitions(unittest.TestCase):
    def test_is_ppo_reflects_critic_presence(self):
        self.assertTrue(_worker(with_critic=True)[0].is_ppo)
        self.assertFalse(_worker(with_critic=False)[0].is_ppo)

    def test_full_step_cycle(self):
        worker, log = _worker(with_critic=True)

        worker._onload_critic()
        self.assertEqual(worker._ppo_phase, PPOPhase.CRITIC_TRAIN)
        self.assertEqual(worker._critic_engine.model_device, "cuda")
        self.assertEqual(worker._engine.model_device, "cpu")

        worker._offload_critic()
        self.assertEqual(worker._ppo_phase, PPOPhase.ALL_OFFLOADED)
        self.assertEqual(worker._critic_engine.model_device, "cpu")

        worker._onload_actor()
        self.assertEqual(worker._ppo_phase, PPOPhase.ACTOR_TRAIN)
        self.assertEqual(worker._engine.model_device, "cuda")

    def test_actor_is_evicted_before_the_critic_is_faulted_in(self):
        """Ordering is the whole point: both resident at once would OOM."""
        worker, log = _worker(with_critic=True)
        worker._engine.model_device = "cuda"

        worker._onload_critic()

        actor_offload = log.index(("actor", "model", "cpu"))
        critic_onload = log.index(("critic", "model", "cuda"))
        self.assertLess(actor_offload, critic_onload)

    def test_critic_optimizer_is_released_before_its_model(self):
        worker, log = _worker(with_critic=True)
        worker._onload_critic()
        log.clear()

        worker._offload_critic()

        self.assertEqual(
            log,
            [("critic", "optimizer", "cpu"), ("critic", "model", "cpu")],
        )

    def test_cannot_onload_critic_during_critic_phase(self):
        worker, _ = _worker(with_critic=True)
        worker._onload_critic()
        with self.assertRaisesRegex(RuntimeError, "Cannot onload the critic"):
            worker._onload_critic()

    def test_cannot_offload_critic_outside_critic_phase(self):
        worker, _ = _worker(with_critic=True)
        with self.assertRaisesRegex(RuntimeError, "Cannot offload the critic"):
            worker._offload_critic()

    def test_cannot_onload_actor_while_critic_resident(self):
        worker, _ = _worker(with_critic=True)
        worker._onload_critic()
        with self.assertRaisesRegex(RuntimeError, "Cannot onload the actor"):
            worker._onload_actor()

    def test_onload_critic_allowed_from_actor_ready(self):
        # The next step's critic phase starts from where the previous one ended.
        worker, _ = _worker(with_critic=True)
        worker._ppo_phase = PPOPhase.ACTOR_READY
        worker._onload_critic()
        self.assertEqual(worker._ppo_phase, PPOPhase.CRITIC_TRAIN)

    def test_onload_actor_allowed_from_actor_ready(self):
        """The KL reward phase re-enters the actor after a warmup step.

        A warmup step leaves the actor resident and the phase at ACTOR_READY;
        the next rollout's KL phase needs a behavior forward and must not be
        rejected, nor should it redundantly transfer the model again.
        """
        worker, log = _worker(with_critic=True)
        worker._ppo_phase = PPOPhase.ACTOR_READY
        worker._engine.model_device = "cuda"

        worker._onload_actor()

        self.assertEqual(worker._ppo_phase, PPOPhase.ACTOR_TRAIN)
        # No redundant H2D: the model was already resident.
        self.assertNotIn(("actor", "model", "cuda"), log)


class TestExternalLifecycleGuards(unittest.TestCase):
    """The trainer drives offload/onload; it must not fault in the wrong model."""

    def test_external_onload_model_rejected_during_critic_phase(self):
        worker, _ = _worker(with_critic=True)
        worker._onload_critic()
        with self.assertRaisesRegex(RuntimeError, "cannot be resident at the same time"):
            TrainingWorker.onload_model(worker)

    def test_external_offload_model_rejected_during_critic_phase(self):
        worker, _ = _worker(with_critic=True)
        worker._onload_critic()
        with self.assertRaisesRegex(RuntimeError, "while the critic owns"):
            TrainingWorker.offload_model(worker)

    def test_external_onload_model_marks_actor_ready(self):
        worker, _ = _worker(with_critic=True)
        worker._clear_cublas_workspaces = MagicMock()
        TrainingWorker.onload_model(worker)
        self.assertEqual(worker._ppo_phase, PPOPhase.ACTOR_READY)

    def test_external_offload_model_marks_all_offloaded(self):
        worker, _ = _worker(with_critic=True)
        worker._ppo_phase = PPOPhase.ACTOR_READY
        worker._clear_cublas_workspaces = MagicMock()
        TrainingWorker.offload_model(worker)
        self.assertEqual(worker._ppo_phase, PPOPhase.ALL_OFFLOADED)

    def test_non_ppo_worker_is_unaffected(self):
        worker, _ = _worker(with_critic=False)
        worker._clear_cublas_workspaces = MagicMock()
        TrainingWorker.offload_model(worker)
        TrainingWorker.onload_model(worker)
        # The phase is inert without a critic.
        self.assertEqual(worker._ppo_phase, PPOPhase.ACTOR_READY)


class TestSaveHF(unittest.TestCase):
    def test_saves_actor_and_critic_and_restores_phase(self):
        worker, log = _worker(with_critic=True)
        worker._ppo_phase = PPOPhase.ACTOR_READY

        TrainingWorker.save_hf(worker, "/tmp/hf-out")

        saved = [entry for entry in log if entry[1] == "save_hf"]
        self.assertEqual(saved[0][0], "actor")
        self.assertEqual(saved[1][0], "critic")
        self.assertTrue(saved[1][2].endswith("/critic"))
        # The worker must be usable for the next step.
        self.assertEqual(worker._ppo_phase, PPOPhase.ACTOR_READY)

    def test_rejected_outside_actor_ready(self):
        worker, _ = _worker(with_critic=True)
        worker._onload_critic()
        with self.assertRaisesRegex(RuntimeError, "Cannot save a PPO HF checkpoint"):
            TrainingWorker.save_hf(worker, "/tmp/hf-out")

    def test_non_ppo_worker_saves_only_the_actor(self):
        worker, log = _worker(with_critic=False)
        TrainingWorker.save_hf(worker, "/tmp/hf-out")
        saved = [entry for entry in log if entry[1] == "save_hf"]
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0][0], "actor")
