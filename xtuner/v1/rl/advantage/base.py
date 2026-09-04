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
from typing import Any

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


class TokenLevelAdvantageEstimator(ABC):
    """Abstract base class for advantage estimators that need a value function.

    :class:`AdvantageEstimator` derives a single scalar advantage per completion
    from a group of rewards, which cannot express a token-level estimator such
    as GAE: GAE needs per-token value predictions and recurses along the
    sequence. Estimators of this kind therefore run inside the training worker,
    after the critic forward pass, rather than during trainer-side data
    preparation.

    Implementations operate on one packed batch, where several trajectories are
    concatenated along the sequence dimension and delimited by ``cu_seq_lens``.
    """

    @abstractmethod
    def compute(
        self,
        values: torch.Tensor,
        token_rewards: torch.Tensor,
        action_mask: torch.Tensor,
        cu_seq_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-token advantages and value-regression targets.

        Args:
            values (torch.Tensor): Frozen critic value predictions, shaped
                ``[T]`` or ``[1, T]``.
            token_rewards (torch.Tensor): Per-token rewards with the same shape
                as ``values``. Typically sparse, holding one terminal reward per
                trajectory plus any dense shaping term.
            action_mask (torch.Tensor): Boolean-like mask marking the tokens the
                policy controls. Non-action positions (prompt and observation
                tokens) are skipped by the recursion.
            cu_seq_lens (torch.Tensor): Cumulative packed sequence boundaries
                ``[0, ..., T]``, used to reset the recursion per trajectory.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Advantages and returns, both
                shaped like ``values`` and zero at non-action positions.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
