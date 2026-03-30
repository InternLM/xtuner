from typing import TYPE_CHECKING

import torch

from xtuner.v1.rl.advantage.base import AdvantageEstimator


if TYPE_CHECKING:
    from xtuner.v1.data_proto.rl_data import RLDataFlowItem


class PassKEstimator(AdvantageEstimator):
    """Pass@k Training for Adaptively Balancing Exploration
    and Exploitation of Large Reasoning Models. https://arxiv.org/pdf/2508.10751

    Converts binary correct/incorrect rewards into pass@k-style advantages.
    For a group where ``c`` out of ``K`` completions are correct (reward > 0),
    the advantages are computed as follows:

    1. Compute the empirical pass rate: ``rho = 1 - C(K-c, K) / C(K, K)``
    2. Compute sigma: ``sigma = sqrt(rho * (1 - rho))``
    3. For correct completions: ``adv = (1 - rho) / (sigma + eps)``
    4. For incorrect completions: ``adv = (1 - rho - C(K-c-1, K-1) / C(K-1, K-1)) / (sigma + eps)``

    This estimator is designed for tasks where each completion is either
    correct or incorrect (e.g., math problems).

    Args:
        k (int): The K in pass@k. Default 4.
        eps (float): Small constant for numerical stability. Default 1e-6.
    """

    def __init__(self, k: int = 4, eps: float = 1e-6) -> None:
        self.k = k
        self.eps = eps

    def _comb(self, n: int, r: int) -> float:
        if r < 0 or r > n:
            return 0.0
        if r == 0 or r == n:
            return 1.0
        from math import comb

        return float(comb(n, r))

    def compute(self, rewards: torch.Tensor, group: list["RLDataFlowItem"]) -> torch.Tensor:
        import numpy as np

        n = len(rewards)
        val = rewards.cpu().numpy()
        binary_val = np.where(val > 0, 1.0, 0.0)
        c = int(np.sum(binary_val))

        if n < self.k:
            rho = c / max(n, 1)
        else:
            rho = 1.0 - self._comb(n - c, self.k) / max(self._comb(n, self.k), 1e-10)

        sigma = np.sqrt(rho * (1 - rho)) + self.eps

        new_reward = np.zeros_like(binary_val, dtype=np.float32)
        new_reward[binary_val > 0] = (1 - rho) / sigma

        if c > 0 and n > 1:
            cond_prob = self._comb(n - c - 1, self.k - 1) / max(self._comb(n - 1, self.k - 1), 1e-10)
            new_reward[binary_val == 0] = (1 - rho - cond_prob) / sigma
        else:
            new_reward[binary_val == 0] = -1.0 / sigma

        return torch.tensor(new_reward, dtype=rewards.dtype, device=rewards.device)

    def __repr__(self) -> str:
        return f"PassKEstimator(k={self.k}, eps={self.eps})"
