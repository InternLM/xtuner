"""Critic update aggregation and token-weighted session value health metrics."""

import unittest

import torch

from xtuner.v1.rl.advantage.stats import critic_value_stats
from xtuner.v1.rl.trainer.controller import TrainingController


class TestCriticLogScalars(unittest.TestCase):
    def test_clip_counters_sum_and_frac_is_count_weighted(self) -> None:
        steps = [
            {
                "reduced_llm_loss": 1.0,
                "grad_norm": 0.5,
                "reduced_critic_value_clip_count": 1.0,
                "reduced_critic_value_valid_count": 3.0,
                "reduced_critic_value_clip_frac": 1.0 / 3.0,
            },
            {
                "reduced_llm_loss": 3.0,
                "grad_norm": 0.7,
                "reduced_critic_value_clip_count": 2.0,
                "reduced_critic_value_valid_count": 10.0,
                "reduced_critic_value_clip_frac": 0.2,
            },
        ]
        # Each rank reports the same global counts; never sum across workers again.
        scalars = TrainingController._critic_log_scalars([{"train_metrics": steps}] * 2)
        self.assertEqual(scalars["critic/value_clip_count"], 3.0)
        self.assertEqual(scalars["critic/value_valid_count"], 13.0)
        self.assertAlmostEqual(scalars["critic/value_clip_frac"], 3.0 / 13.0)
        self.assertAlmostEqual(scalars["critic/value_loss"], 2.0)
        self.assertAlmostEqual(scalars["critic/grad_norm"], 0.6)

    def test_empty_logs_return_no_scalars(self) -> None:
        self.assertEqual(TrainingController._critic_log_scalars(None), {})
        self.assertEqual(TrainingController._critic_log_scalars([]), {})
        self.assertEqual(TrainingController._critic_log_scalars([{"train_metrics": []}]), {})

    def test_zero_valid_count_guards_division(self) -> None:
        scalars = TrainingController._critic_log_scalars(
            [{"train_metrics": [{"reduced_critic_value_clip_count": 0.0, "reduced_critic_value_valid_count": 0.0}]}]
        )
        self.assertEqual(scalars["critic/value_clip_frac"], 0.0)

    def test_non_numeric_values_are_skipped(self) -> None:
        scalars = TrainingController._critic_log_scalars(
            [{"train_metrics": [{"reduced_llm_loss": "nan-string", "grad_norm": True}]}]
        )
        self.assertEqual(scalars, {})


class TestSessionValueStats(unittest.TestCase):
    def test_perfect_critic_explains_all_variance(self) -> None:
        returns = [torch.tensor([0.0, 1.0, 2.0]), torch.tensor([3.0, 0.5])]
        masks = [torch.ones_like(row, dtype=torch.bool) for row in returns]
        stats = critic_value_stats(returns, returns, masks)
        self.assertAlmostEqual(stats["critic/explained_variance"], 1.0)
        self.assertAlmostEqual(stats["critic/value_mean"], stats["critic/return_mean"])
        self.assertAlmostEqual(stats["critic/value_std"], stats["critic/return_std"])
        # Unequal row lengths must not turn this into a mean of per-sample means.
        self.assertAlmostEqual(stats["critic/value_mean"], 1.3)
        self.assertAlmostEqual(stats["critic/value_std"], torch.cat(returns).double().std(correction=0).item())

    def test_constant_critic_scores_zero(self) -> None:
        returns = [torch.tensor([0.0, 1.0, 2.0, 3.0])]
        values = [torch.full((4,), 0.5)]
        stats = critic_value_stats(values, returns, [torch.ones(4, dtype=torch.bool)])
        self.assertAlmostEqual(stats["critic/explained_variance"], 0.0)
        self.assertAlmostEqual(stats["critic/value_std"], 0.0)

    def test_anticorrelated_critic_scores_negative(self) -> None:
        returns = [torch.tensor([0.0, 1.0, 2.0, 3.0])]
        values = [torch.tensor([3.0, 2.0, 1.0, 0.0])]
        stats = critic_value_stats(values, returns, [torch.ones(4, dtype=torch.bool)])
        self.assertEqual(stats["critic/explained_variance"], -3.0)

    def test_constant_returns_guard_zero_variance(self) -> None:
        for length in (1, 2):
            returns = [torch.ones(length)]
            stats = critic_value_stats(returns, returns, [torch.ones(length, dtype=torch.bool)])
            self.assertEqual(stats["critic/explained_variance"], 0.0)
            self.assertEqual(stats["critic/value_std"], 0.0)

    def test_ignored_positions_are_excluded(self) -> None:
        values = [torch.tensor([float("nan"), 0.5, 1.5]), torch.empty(0)]
        returns = [torch.tensor([9.0, 0.5, 1.5]), torch.empty(0)]
        masks = [torch.tensor([False, True, True]), torch.empty(0, dtype=torch.bool)]
        stats = critic_value_stats(values, returns, masks)
        self.assertAlmostEqual(stats["critic/value_mean"], 1.0)
        self.assertAlmostEqual(stats["critic/explained_variance"], 1.0)
        self.assertEqual(critic_value_stats([torch.ones(1)], [torch.ones(1)], [torch.tensor([False])]), {})
        self.assertEqual(critic_value_stats([], [], []), {})
