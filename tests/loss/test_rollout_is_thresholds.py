"""Rollout importance-sampling thresholds are ``(upper, lower)`` pairs, as documented."""

from __future__ import annotations

import pytest
import torch

from xtuner.v1.rl.rollout_is import RolloutImportanceSampling


def _compute(config: RolloutImportanceSampling) -> tuple[torch.Tensor | None, torch.Tensor]:
    # Two packed sequences of three tokens: log ratios (0.5, 0, -0.5) and (0, 0, 0).
    old_log_prob = torch.tensor([[0.5, 0.0, -0.5, 0.0, 0.0, 0.0]])
    rollout_log_prob = torch.zeros_like(old_log_prob)
    response_mask = torch.ones_like(old_log_prob)
    weights, mask, _, _ = config.compute_rollout_importance_weights_and_metrics(
        old_log_prob=old_log_prob,
        rollout_log_prob=rollout_log_prob,
        num_tokens=torch.tensor([3, 3]),
        response_mask=response_mask,
    )
    return weights, mask


class TestRolloutISThresholds:
    def test_documented_example_clamps_weights(self) -> None:
        config = RolloutImportanceSampling(
            rollout_is_level="token",
            rollout_is_mode="truncate",
            rollout_is_threshold=(1.2, 0.8),
        )
        weights, _ = _compute(config)
        assert weights is not None
        torch.testing.assert_close(weights[0, :3], torch.tensor([1.2, 1.0, 0.8]))

    def test_lower_upper_order_is_rejected(self) -> None:
        config = RolloutImportanceSampling(rollout_is_threshold=(0.8, 1.2))
        with pytest.raises(AssertionError, match="upper_threshold must be greater"):
            _compute(config)

    def test_veto_drops_whole_sequence(self) -> None:
        config = RolloutImportanceSampling(rollout_is_threshold=(2.0, 0.5), rollout_is_veto_threshold=(1.5, 0.1))
        _, mask = _compute(config)
        assert mask.tolist() == [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]
