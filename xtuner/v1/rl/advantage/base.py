"""Base classes for advantage estimation in RL training.

Example:
    Create a custom advantage estimator::

        import torch
        from xtuner.v1.data_proto.rl_data import RLDataFlowItem
        from xtuner.v1.rl.advantage.base import AdvantageEstimator

        class MyCustomEstimator(AdvantageEstimator):
            def compute(self, rewards: torch.Tensor, group: list[RLDataFlowItem]) -> torch.Tensor:
                # rewards: (K,) tensor of rewards for the group
                # group: list of RLDataFlowItem with full rollout info
                mean = rewards.mean()
                std = rewards.std() + 1e-8
                return (rewards - mean) / std
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

import torch


class AdvantageEstimator(ABC):
    """Abstract base class for advantage estimation.

    Subclasses must implement the :meth:`compute` method to define how advantages
    are computed from rewards. The framework handles batching, logging, and integration
    with the training loop automatically.

    Example:
        GRPO-style advantage estimation::

            class GRPOEstimator(AdvantageEstimator):
                def compute(self, rewards, group):
                    mean = rewards.mean()
                    std = rewards.std() + 1e-8
                    return (rewards - mean) / std
    """

    @abstractmethod
    def compute(self, rewards: torch.Tensor, group: list[Any]) -> torch.Tensor:
        """Compute advantages from rewards for a single prompt group.

        Args:
            rewards (torch.Tensor): Tensor of shape ``(K,)`` containing the reward
                scores for each of the ``K`` completions of the prompt group.
            group (list[RLDataFlowItem]): List of ``K`` data flow items containing
                full rollout information (logprobs, response_ids, finish_reason,
                etc.) for each completion. Can be used for more sophisticated
                advantage computations that require additional signals.

        Returns:
            torch.Tensor: Tensor of shape ``(K,)`` containing the computed advantages.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class TokenAdvantageEstimator(ABC):
    """Interface for token-level trajectory advantage estimation.

    Token estimators are defined in parallel with :class:`AdvantageEstimator`:
    they consume complete trajectories (values, terminal scores, action masks and
    packed boundaries) and return unnormalized ``(advantages, returns)``. They do
    not implement group-scalar ``compute``.
    """

    reward_scope: Literal["segment", "session"]
    normalize_actor_advantage: bool

    @abstractmethod
    def compute_segment_gae(
        self,
        values: torch.Tensor,
        scores: torch.Tensor,
        mask: torch.Tensor,
        cu_seq_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return unnormalized actor advantages and critic returns.

        Args:
            values (torch.Tensor): Frozen critic values of shape ``[T]``.
            scores (torch.Tensor): One terminal score per trajectory.
            mask (torch.Tensor): Action-token mask of shape ``[T]``.
            cu_seq_lens (torch.Tensor): Packed trajectory boundaries.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Actor advantages and critic returns.
        """
        ...
