from typing import TYPE_CHECKING

import torch

from xtuner.v1.rl.advantage.base import AdvantageEstimator


if TYPE_CHECKING:
    from xtuner.v1.data_proto.rl_data import RLDataFlowItem


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
    """

    def compute(self, rewards: torch.Tensor, group: list["RLDataFlowItem"]) -> torch.Tensor:
        lengths = torch.tensor(
            [len(d.env.rollout.response_ids) for d in group],  # type: ignore
            dtype=torch.float32,
            device=rewards.device,
        )
        baseline = (rewards * lengths).sum() / (lengths.sum() + 1e-8)
        return rewards - baseline

    def __repr__(self) -> str:
        return "OPOEstimator()"
