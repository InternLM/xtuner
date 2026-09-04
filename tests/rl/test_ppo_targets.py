"""Tests for the PPO target computation inside the training worker.

Covers `_compute_ppo_targets`, the seam between the critic forward and the
actor/critic losses, using a real GAE estimator but no model or accelerator.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch

from xtuner.v1.rl.advantage import GAEAdvantageConfig
from xtuner.v1.rl.trainer.worker import TrainingWorker


_FAKE_DEVICE_MODULE = MagicMock()
_FAKE_DEVICE_MODULE.memory_allocated.return_value = 0
_FAKE_DEVICE_MODULE.memory_reserved.return_value = 0
_DEVICE_MODULE = patch("xtuner.v1.rl.trainer.worker.DEVICE_MODULE", _FAKE_DEVICE_MODULE)


def setUpModule() -> None:
    _DEVICE_MODULE.start()


def tearDownModule() -> None:
    _DEVICE_MODULE.stop()


def _worker(*, normalize: bool, gamma: float = 1.0, gae_lambda: float = 1.0) -> TrainingWorker:
    worker = TrainingWorker.__new__(TrainingWorker)
    worker._advantage_estimator = GAEAdvantageConfig(gamma=gamma, gae_lambda=gae_lambda).build()
    worker._normalize_advantage = normalize
    worker.logger = MagicMock()
    return worker


class TestComputePPOTargets(unittest.TestCase):
    def test_returns_are_advantage_plus_value(self):
        worker = _worker(normalize=False)
        values = torch.tensor([[0.1, 0.2, 0.3]])
        token_rewards = torch.tensor([[0.0, 0.0, 1.0]])
        action_mask = torch.ones(1, 3, dtype=torch.bool)

        advantages, returns = worker._compute_ppo_targets(
            [values], [token_rewards], [action_mask], [torch.tensor([0, 3])]
        )

        # gamma = lambda = 1, so the return target is the reward-to-go.
        torch.testing.assert_close(returns[0], torch.ones(1, 3))
        torch.testing.assert_close(advantages[0], torch.ones(1, 3) - values)

    def test_observation_tokens_get_no_advantage(self):
        worker = _worker(normalize=False)
        action_mask = torch.tensor([[True, False, True]])

        advantages, returns = worker._compute_ppo_targets(
            [torch.tensor([[0.1, 9.9, 0.3]])],
            [torch.tensor([[0.0, 0.0, 1.0]])],
            [action_mask],
            [torch.tensor([0, 3])],
        )

        self.assertEqual(advantages[0][0, 1].item(), 0.0)
        self.assertEqual(returns[0][0, 1].item(), 0.0)

    def test_normalization_spans_the_whole_batch(self):
        """Normalizing per pack would make the step size depend on packing."""
        worker = _worker(normalize=True)
        packs = [torch.tensor([[0.0, 0.0]]), torch.tensor([[0.0, 0.0]])]
        rewards = [torch.tensor([[0.0, 1.0]]), torch.tensor([[0.0, 3.0]])]
        masks = [torch.ones(1, 2, dtype=torch.bool)] * 2
        bounds = [torch.tensor([0, 2])] * 2

        advantages, _ = worker._compute_ppo_targets(packs, rewards, masks, bounds)

        combined = torch.cat([a.reshape(-1) for a in advantages])
        self.assertAlmostEqual(float(combined.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(combined.std(unbiased=False)), 1.0, places=4)

    def test_normalization_can_be_disabled(self):
        worker = _worker(normalize=False)
        packs = [torch.tensor([[0.0, 0.0]])]
        rewards = [torch.tensor([[0.0, 2.0]])]
        masks = [torch.ones(1, 2, dtype=torch.bool)]

        advantages, _ = worker._compute_ppo_targets(packs, rewards, masks, [torch.tensor([0, 2])])

        # Raw reward-to-go, untouched.
        torch.testing.assert_close(advantages[0], torch.tensor([[2.0, 2.0]]))

    def test_normalization_preserves_pack_shapes(self):
        worker = _worker(normalize=True)
        packs = [torch.zeros(1, 3), torch.zeros(1, 5)]
        rewards = [torch.zeros(1, 3), torch.zeros(1, 5)]
        rewards[0][0, -1] = 1.0
        rewards[1][0, -1] = -1.0
        masks = [torch.ones(1, 3, dtype=torch.bool), torch.ones(1, 5, dtype=torch.bool)]
        bounds = [torch.tensor([0, 3]), torch.tensor([0, 5])]

        advantages, _ = worker._compute_ppo_targets(packs, rewards, masks, bounds)

        self.assertEqual(advantages[0].shape, (1, 3))
        self.assertEqual(advantages[1].shape, (1, 5))

    def test_shape_mismatch_names_the_sequence_parallel_cause(self):
        """A short gather is the realistic failure; the message must say so."""
        worker = _worker(normalize=False)
        with self.assertRaisesRegex(RuntimeError, "sp_size > 1"):
            worker._compute_ppo_targets(
                [torch.zeros(1, 2)],
                [torch.zeros(1, 4)],
                [torch.ones(1, 4, dtype=torch.bool)],
                [torch.tensor([0, 4])],
            )

    def test_packed_trajectories_do_not_leak_rewards(self):
        worker = _worker(normalize=False)
        # Two trajectories in one pack; only the second is rewarded.
        values = torch.zeros(1, 4)
        token_rewards = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        masks = torch.ones(1, 4, dtype=torch.bool)

        advantages, _ = worker._compute_ppo_targets([values], [token_rewards], [masks], [torch.tensor([0, 2, 4])])

        torch.testing.assert_close(advantages[0], torch.tensor([[0.0, 0.0, 1.0, 1.0]]))
