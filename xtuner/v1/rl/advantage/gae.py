"""Per-sample generalized advantage estimation for PPO.

GAE runs on one original sample. Packed sequences are only a training layout and must not be the sequence GAE walks.
"""

from __future__ import annotations

from typing import Any

import torch

from .base import AdvantageEstimator


class GAEEstimator(AdvantageEstimator):
    """Outcome-reward GAE over the supervised tokens of one sample."""

    def __init__(self, gamma: float = 1.0, lam: float = 0.95) -> None:
        self.gamma = gamma
        self.lam = lam

    def compute(self, rewards: torch.Tensor, group: list[Any]) -> torch.Tensor:
        """Group-scalar advantages are not the PPO target.

        Token advantages come from :meth:`compute_token_gae` after the critic
        has produced per-token values.
        """
        del rewards, group
        raise NotImplementedError(
            "GAEEstimator.compute does not estimate group-scalar advantages. "
            "Use compute_token_gae on one sample after the critic value pass."
        )

    def compute_token_gae(
        self,
        values: list[float],
        mask: list[bool],
        reward: float,
    ) -> tuple[list[float], list[float]]:
        """Return ``(advantages, returns)`` aligned with ``values``.

        ``mask`` marks supervised positions. The outcome ``reward`` is placed on
        the last supervised token. Prompt and other masked-off positions stay 0.
        """
        if len(values) != len(mask):
            raise ValueError(f"values length {len(values)} does not match mask length {len(mask)}")
        advantages = [0.0] * len(values)
        returns = [0.0] * len(values)
        valid = [index for index, supervised in enumerate(mask) if supervised]
        if not valid:
            return advantages, returns

        rewards = [0.0] * len(values)
        rewards[valid[-1]] = reward
        last_gae = 0.0
        next_value = 0.0
        for index in reversed(valid):
            delta = rewards[index] + self.gamma * next_value - values[index]
            last_gae = delta + self.gamma * self.lam * last_gae
            advantages[index] = last_gae
            returns[index] = last_gae + values[index]
            next_value = values[index]
        return advantages, returns
