from typing import TYPE_CHECKING

import torch

from xtuner.v1.rl.advantage.base import AdvantageEstimator


if TYPE_CHECKING:
    from xtuner.v1.data_proto.rl_data import RLDataFlowItem


class RLOOEstimator(AdvantageEstimator):
    """REINFORCE Leave-One-Out (RLOO) advantage estimator. https://arxiv.org/abs/2402.14740

    Uses a leave-one-out baseline: for each completion i, the baseline is the
    mean of all other completions in the same group.

    For a group of K completions::

        baseline_i = (sum(rewards) - reward_i) / (K - 1)
        adv_i = reward_i - baseline_i

    When K=1, returns the raw reward as the advantage.
    """

    def compute(self, rewards: torch.Tensor, group: list["RLDataFlowItem"]) -> torch.Tensor:
        k = len(rewards)
        if k == 1:
            return rewards
        sum_rewards = rewards.sum()
        baseline = (sum_rewards - rewards) / (k - 1)
        return rewards - baseline

    def __repr__(self) -> str:
        return "RLOOEstimator()"
