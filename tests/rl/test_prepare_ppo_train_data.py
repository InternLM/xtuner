"""PPO-specific contract tests for RLTrainer._prepare_train_data.

Pure data-construction logic: no trainer, Ray worker, model or rollout backend.
Covers the PPO branch, where the group-relative advantage is replaced by a
terminal token reward that the worker later turns into GAE advantages.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch

from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.train.rl_trainer import BaseRLTrainer, _terminal_token_rewards


class _UnusedEstimator:
    """Fails loudly if the PPO path consults a group-baseline estimator."""

    def compute(self, rewards_tensor, group):  # pragma: no cover - must not run
        raise AssertionError("PPO must not use a group-baseline advantage estimator.")


class TestTerminalTokenRewards(unittest.TestCase):
    def test_reward_lands_on_last_trainable_token(self):
        labels = torch.tensor([[-100, 20, -100, 22, -100]])
        rewards = _terminal_token_rewards(labels, 2.5)
        # Index 3 is the final non-ignored label.
        self.assertEqual(rewards.tolist(), [[0.0, 0.0, 0.0, 2.5, 0.0]])

    def test_shape_and_dtype_match_labels(self):
        labels = torch.tensor([[-100, 20, 21]])
        rewards = _terminal_token_rewards(labels, 1.0)
        self.assertEqual(rewards.shape, labels.shape)
        self.assertEqual(rewards.dtype, torch.float32)

    def test_negative_reward_is_preserved(self):
        rewards = _terminal_token_rewards(torch.tensor([[20, 21]]), -1.0)
        self.assertEqual(rewards.tolist(), [[0.0, -1.0]])

    def test_zero_reward_without_trainable_tokens_is_allowed(self):
        rewards = _terminal_token_rewards(torch.tensor([[-100, -100]]), 0.0)
        self.assertEqual(rewards.tolist(), [[0.0, 0.0]])

    def test_nonzero_reward_without_trainable_tokens_raises(self):
        with self.assertRaisesRegex(ValueError, "no trainable token"):
            _terminal_token_rewards(torch.tensor([[-100, -100]]), 1.0)


class TestPreparePPOTrainData(unittest.TestCase):
    def _build_trainer(self, *, is_ppo: bool = True):
        trainer = BaseRLTrainer.__new__(BaseRLTrainer)
        trainer._is_ppo = is_ppo
        trainer._advantage_estimator = None if is_ppo else _UnusedEstimator()
        trainer.tokenizer = MagicMock(return_value={"input_ids": torch.tensor([[999]])})
        trainer.logger = MagicMock()
        return trainer

    def _state(self, *, uid=1, group_id=1, score=1.0, **overrides) -> RolloutState:
        kwargs = dict(
            rollout_id=uid,
            group_id=group_id,
            message=[{"role": "user", "content": "prompt"}],
            prompt_ids=[10, 11, 12],
            response="response",
            response_ids=[20, 21, 22],
            logprobs=[0.1, 0.2, 0.3],
            response_mask=None,
            reward={"score": score},
            status=Status.COMPLETED,
            finish_reason="stop",
            extra_fields={},
        )
        kwargs.update(overrides)
        return RolloutState(**kwargs)

    def _prepare(self, trainer, data_groups, pack_max_length=128):
        with patch("xtuner.v1.train.rl_trainer.XTUNER_DETERMINISTIC", True):
            return trainer._prepare_train_data(data_groups, pack_max_length=pack_max_length)

    def test_token_rewards_are_attached_on_the_text_path(self):
        trainer = self._build_trainer()
        data_batches, _ = self._prepare(trainer, [[self._state(score=1.0)]])

        batch = data_batches[0]
        self.assertIn("token_rewards", batch)
        token_rewards = batch["token_rewards"]
        self.assertEqual(token_rewards.shape, batch["shifted_labels"].shape)
        # shifted_labels = [-100, -100, 20, 21, 22], so the terminal action is
        # the last position.
        self.assertEqual(token_rewards.tolist(), [[0.0, 0.0, 0.0, 0.0, 1.0]])

    def test_token_rewards_respect_the_response_mask(self):
        trainer = self._build_trainer()
        # The final response token is masked out, so the reward must move to the
        # last token the policy actually controls.
        state = self._state(response_mask=[1, 1, 0])
        data_batches, _ = self._prepare(trainer, [[state]])

        batch = data_batches[0]
        self.assertEqual(batch["shifted_labels"].tolist(), [[-100, -100, 20, 21, -100]])
        self.assertEqual(batch["token_rewards"].tolist(), [[0.0, 0.0, 0.0, 1.0, 0.0]])

    def test_each_sample_keeps_its_own_reward(self):
        trainer = self._build_trainer()
        group = [self._state(uid=1, score=1.0), self._state(uid=2, score=-1.0)]

        data_batches, _ = self._prepare(trainer, [group])

        self.assertEqual(len(data_batches), 2)
        totals = sorted(float(batch["token_rewards"].sum()) for batch in data_batches)
        self.assertEqual(totals, [-1.0, 1.0])

    def test_advantages_are_left_to_the_worker(self):
        """PPO must not precompute advantages: the critic has not run yet."""
        trainer = self._build_trainer()
        data_batches, _ = self._prepare(trainer, [[self._state(score=1.0)]])

        advantage = data_batches[0]["advantage"]
        self.assertTrue(all(value == 0.0 for value in advantage), advantage)

    def test_uniform_reward_group_is_still_trained_on(self):
        """A learned baseline gives signal where a group baseline gives none."""
        trainer = self._build_trainer()
        group = [self._state(uid=1, score=1.0), self._state(uid=2, score=1.0)]

        data_batches, info = self._prepare(trainer, [group])

        self.assertEqual(len(data_batches), 2)
        self.assertEqual(info["training_samples"], 2)
        for batch in data_batches:
            self.assertEqual(float(batch["token_rewards"].sum()), 1.0)

    def test_pre_tokenized_path_attaches_token_rewards(self):
        trainer = self._build_trainer()
        state = self._state(
            input_ids=[10, 11, 20, 21],
            labels=[-100, -100, 20, 21],
            logprobs=[0.0, 0.0, 0.1, 0.2],
            prompt_ids=None,
            response_ids=None,
        )

        data_batches, _ = self._prepare(trainer, [[state]])

        batch = data_batches[0]
        self.assertEqual(batch["shifted_labels"].tolist(), [[-100, 20, 21]])
        self.assertEqual(batch["token_rewards"].tolist(), [[0.0, 0.0, 1.0]])

    def test_non_ppo_path_omits_token_rewards(self):
        """Group-baseline algorithms must not pay for PPO tensors."""
        trainer = BaseRLTrainer.__new__(BaseRLTrainer)
        trainer._is_ppo = False

        class _Fixed:
            def compute(self, rewards_tensor, group):
                return torch.tensor([1.5] * len(group), dtype=torch.float32)

        trainer._advantage_estimator = _Fixed()
        trainer.tokenizer = MagicMock(return_value={"input_ids": torch.tensor([[999]])})
        trainer.logger = MagicMock()

        data_batches, _ = self._prepare(trainer, [[self._state()]])

        self.assertNotIn("token_rewards", data_batches[0])
        self.assertEqual(data_batches[0]["advantage"][0], 1.5)
