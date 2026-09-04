"""Tests for the PPO KL-in-reward penalty and critic warmup."""

import unittest
from unittest.mock import MagicMock, patch

import torch

from xtuner.v1.rl.loss import kl_divergence_per_token
from xtuner.v1.rl.trainer.critic_config import KLRewardConfig
from xtuner.v1.rl.trainer.worker import PPOPhase, TrainingWorker


_FAKE_DEVICE_MODULE = MagicMock()
_FAKE_DEVICE_MODULE.memory_allocated.return_value = 0
_FAKE_DEVICE_MODULE.memory_reserved.return_value = 0
_DEVICE_MODULE = patch("xtuner.v1.rl.trainer.worker.DEVICE_MODULE", _FAKE_DEVICE_MODULE)
_DEVICE = patch("xtuner.v1.rl.trainer.worker.DEVICE", "cpu")


def setUpModule() -> None:
    _DEVICE_MODULE.start()
    _DEVICE.start()


def tearDownModule() -> None:
    _DEVICE_MODULE.stop()
    _DEVICE.stop()


class _FakeEngine:
    def __init__(self):
        self.model_device = "cpu"

    def put_model_to_device(self, device):
        self.model_device = str(device)
        return True

    def put_optimizer_to_device(self, device):
        return True


class _FakeSPMesh:
    def size(self):
        return 1


def _worker(kl_cfg: KLRewardConfig, behavior: list[torch.Tensor], ref: list[torch.Tensor]):
    worker = TrainingWorker.__new__(TrainingWorker)
    worker._engine = _FakeEngine()
    worker._critic_engine = _FakeEngine()
    worker._kl_reward_cfg = kl_cfg
    worker._ppo_phase = PPOPhase.ALL_OFFLOADED
    worker.sp_mesh = _FakeSPMesh()
    worker.logger = MagicMock()
    worker.config = MagicMock()
    worker._actor_forward_logprobs = lambda seq_ctx, labels: behavior
    worker._ref_forward_logprobs = lambda seq_ctx, labels: ref
    return worker


def _prepared(token_rewards: torch.Tensor, action_mask: torch.Tensor) -> dict:
    return {
        "seq_ctx_list": [MagicMock()],
        "shifted_labels_list": [torch.zeros_like(token_rewards, dtype=torch.long)],
        "rollout_logprobs_list": [None],
        "token_rewards_list": [token_rewards],
        "action_mask_list": [action_mask],
        "cu_seq_lens_list": [torch.tensor([0, token_rewards.numel()])],
    }


class TestKLReward(unittest.TestCase):
    def test_penalty_is_subtracted_from_token_rewards(self):
        behavior = torch.tensor([[0.0, -1.0, -2.0]])
        ref = torch.tensor([[0.0, -1.5, -1.0]])
        kl_cfg = KLRewardConfig(coef=0.5, kl_type="k1")
        worker = _worker(kl_cfg, [behavior], [ref])

        token_rewards = torch.tensor([[0.0, 0.0, 1.0]])
        prepared = _prepared(token_rewards, torch.ones(1, 3, dtype=torch.bool))
        log: dict = {}

        worker._apply_kl_reward(prepared, log)

        expected = token_rewards - 0.5 * kl_divergence_per_token(behavior, ref, "k1")
        torch.testing.assert_close(prepared["token_rewards_list"][0], expected)

    def test_terminal_reward_is_preserved_alongside_the_penalty(self):
        """The penalty is additive; it must not clobber the task reward."""
        behavior = torch.tensor([[0.0, 0.0]])
        ref = torch.tensor([[0.0, 0.0]])
        worker = _worker(KLRewardConfig(coef=1.0, kl_type="k1"), [behavior], [ref])

        prepared = _prepared(torch.tensor([[0.0, 1.0]]), torch.ones(1, 2, dtype=torch.bool))
        worker._apply_kl_reward(prepared, {})

        # Identical policies give zero KL, so the reward is unchanged.
        torch.testing.assert_close(prepared["token_rewards_list"][0], torch.tensor([[0.0, 1.0]]))

    def test_non_action_tokens_are_not_penalized(self):
        behavior = torch.tensor([[5.0, 0.0]])
        ref = torch.tensor([[0.0, 0.0]])
        worker = _worker(KLRewardConfig(coef=1.0, kl_type="k1"), [behavior], [ref])

        # Position 0 is an observation token with a large divergence.
        prepared = _prepared(torch.zeros(1, 2), torch.tensor([[False, True]]))
        worker._apply_kl_reward(prepared, {})

        self.assertEqual(prepared["token_rewards_list"][0][0, 0].item(), 0.0)

    def test_zero_coefficient_is_a_no_op(self):
        worker = _worker(KLRewardConfig(coef=0.0), [torch.tensor([[1.0]])], [torch.tensor([[0.0]])])
        prepared = _prepared(torch.tensor([[1.0]]), torch.ones(1, 1, dtype=torch.bool))

        worker._apply_kl_reward(prepared, {})

        torch.testing.assert_close(prepared["token_rewards_list"][0], torch.tensor([[1.0]]))

    def test_mean_kl_is_logged(self):
        behavior = torch.tensor([[0.0, -1.0]])
        ref = torch.tensor([[0.0, -3.0]])
        worker = _worker(KLRewardConfig(coef=0.1, kl_type="k1"), [behavior], [ref])
        prepared = _prepared(torch.zeros(1, 2), torch.ones(1, 2, dtype=torch.bool))
        log: dict = {}

        worker._apply_kl_reward(prepared, log)

        # k1 KL is (0 - 0, -1 - -3) = (0, 2); the mean over two tokens is 1.
        self.assertAlmostEqual(log["kl_reward_mean"], 1.0, places=5)

    def test_actor_is_evicted_after_the_behavior_forward(self):
        """The critic phase follows, so the actor must not stay resident."""
        worker = _worker(
            KLRewardConfig(coef=0.1, behavior_logprobs="old"),
            [torch.zeros(1, 2)],
            [torch.zeros(1, 2)],
        )
        prepared = _prepared(torch.zeros(1, 2), torch.ones(1, 2, dtype=torch.bool))

        worker._apply_kl_reward(prepared, {})

        self.assertEqual(worker._engine.model_device, "cpu")
        self.assertEqual(worker._ppo_phase, PPOPhase.ALL_OFFLOADED)

    def test_rollout_behavior_skips_the_actor_forward(self):
        worker = _worker(
            KLRewardConfig(coef=0.1, kl_type="k1", behavior_logprobs="rollout"),
            [torch.full((1, 2), 99.0)],  # would poison the result if used
            [torch.zeros(1, 2)],
        )
        prepared = _prepared(torch.zeros(1, 2), torch.ones(1, 2, dtype=torch.bool))
        prepared["rollout_logprobs_list"] = [torch.tensor([[0.0, -2.0]])]

        worker._apply_kl_reward(prepared, {})

        expected = -0.1 * torch.tensor([[0.0, -2.0]])
        torch.testing.assert_close(prepared["token_rewards_list"][0], expected)
        # The actor was never faulted in.
        self.assertEqual(worker._engine.model_device, "cpu")

    def test_missing_rollout_logprobs_raises(self):
        worker = _worker(
            KLRewardConfig(behavior_logprobs="rollout"),
            [torch.zeros(1, 2)],
            [torch.zeros(1, 2)],
        )
        prepared = _prepared(torch.zeros(1, 2), torch.ones(1, 2, dtype=torch.bool))

        with self.assertRaisesRegex(ValueError, "rollout logprobs"):
            worker._apply_kl_reward(prepared, {})


class TestKLRewardConfig(unittest.TestCase):
    def test_defaults_match_the_documented_behavior(self):
        cfg = KLRewardConfig()
        self.assertEqual(cfg.behavior_logprobs, "old")
        self.assertEqual(cfg.kl_type, "low_var_kl")

    def test_negative_coefficient_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "coef must be non-negative"):
            KLRewardConfig(coef=-0.1)


class TestCriticWarmup(unittest.TestCase):
    """Warmup trains only the critic, so early noisy values never move the policy."""

    @staticmethod
    def _worker(warmup_steps: int, rollout_step: int) -> TrainingWorker:
        worker = TrainingWorker.__new__(TrainingWorker)
        worker._critic_warmup_steps = warmup_steps
        worker._rollout_step = rollout_step
        return worker

    def test_warmup_is_active_for_the_configured_steps(self):
        for step in range(3):
            worker = self._worker(warmup_steps=3, rollout_step=step)
            self.assertLess(worker._rollout_step, worker._critic_warmup_steps, f"step {step}")

    def test_warmup_ends_after_the_configured_steps(self):
        worker = self._worker(warmup_steps=3, rollout_step=3)
        self.assertGreaterEqual(worker._rollout_step, worker._critic_warmup_steps)

    def test_zero_warmup_trains_the_actor_immediately(self):
        worker = self._worker(warmup_steps=0, rollout_step=0)
        self.assertGreaterEqual(worker._rollout_step, worker._critic_warmup_steps)
