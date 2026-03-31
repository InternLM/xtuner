from typing import TYPE_CHECKING

import torch

from xtuner.v1.rl.advantage.base import AdvantageEstimator


if TYPE_CHECKING:
    from xtuner.v1.data_proto.rl_data import RLDataFlowItem


class GRPOEstimator(AdvantageEstimator):
    """Group Relative Policy Optimization (GRPO) advantage estimator. https://arxiv.org/abs/2402.03300

    Computes advantages by normalizing rewards within each prompt group using
    z-score normalization (mean=0, std=1).

    For a group of K completions from the same prompt, the advantage for
    completion i is::

        adv_i = (reward_i - mean(rewards)) / (std(rewards) + eps)

    Args:
        eps (float): Small constant for numerical stability. Default 1e-8.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps

    def compute(self, rewards: torch.Tensor, group: list["RLDataFlowItem"]) -> torch.Tensor:
        mean = rewards.mean()
        std = rewards.std() + self.eps
        return (rewards - mean) / std

    def __repr__(self) -> str:
        return f"GRPOEstimator(eps={self.eps})"


class DrGRPOEstimator(AdvantageEstimator):
    """DrGRPO advantage estimator. https://arxiv.org/abs/2503.20783

    Similar to GRPO but scales advantages by the response length relative to a
    max length. Longer responses contribute more
    to the gradient.

    For a group of K completions::

        adv_i = (reward_i - mean(rewards)) / std * len_i / max_length

    Args:
        max_length (float): Reference response length for scaling. Default 32768.
        eps (float): Small constant for numerical stability. Default 1e-8.
    """

    def __init__(self, max_length: float, eps: float = 1e-8) -> None:
        self.max_length = max_length
        self.eps = eps

    def compute(self, rewards: torch.Tensor, group: list["RLDataFlowItem"]) -> torch.Tensor:
        mean = rewards.mean()
        std = rewards.std() + self.eps
        z = (rewards - mean) / std

        lengths = torch.tensor(
            [len(d.env.rollout.response_ids) for d in group],  # type: ignore
            dtype=torch.float32,
            device=rewards.device,
        )
        return z * lengths / self.max_length

    def __repr__(self) -> str:
        return f"DrGRPOEstimator(max_length={self.max_length}, eps={self.eps})"
