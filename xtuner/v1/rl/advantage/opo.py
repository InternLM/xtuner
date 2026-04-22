from typing import Any

import torch

from xtuner.v1.rl.advantage.base import AdvantageEstimator


class OPOEstimator(AdvantageEstimator):
    """OPO advantage estimator.

    On-Policy RL with Optimal Reward Baseline https://arxiv.org/abs/2505.23585

    Uses a length-weighted average as the baseline. For each completion,
    the baseline is the weighted mean of all rewards in the group, where
    the weight is the response length.

    For a group of K completions::

        baseline = sum(reward_i * len_i) / sum(len_i)
        adv_i = reward_i - baseline

    This encourages the model to produce longer correct responses and
    shorter incorrect responses.

    Args:
        eps (float): Small constant for numerical stability. Default 1e-8.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps

    def compute(self, rewards: torch.Tensor, group: list[Any]) -> torch.Tensor:
        lengths = torch.tensor(
            [_response_len(data) for data in group],
            dtype=torch.float32,
            device=rewards.device,
        )
        baseline = (rewards * lengths).sum() / (lengths.sum() + self.eps)
        return rewards - baseline

    def __repr__(self) -> str:
        return "OPOEstimator()"


def _response_len(data: Any) -> int:
    if hasattr(data, "response_ids"):
        return len(data.response_ids or [])
    return len(data.env.rollout.response_ids or [])
