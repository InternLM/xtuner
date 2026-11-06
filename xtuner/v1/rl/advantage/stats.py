"""Token-level advantage statistics for PPO step logging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


@dataclass(frozen=True)
class TokenStats:
    """Effective action-token statistics of normalized advantages.

    ``min``/``max`` are ``None`` only when ``count`` is zero so empty DP
    representatives never contaminate the extrema.

    Args:
        sum (float): Sum over kept action tokens.
        count (int): Number of kept action tokens.
        min (float | None): Minimum kept advantage, or None when empty.
        max (float | None): Maximum kept advantage, or None when empty.
    """

    sum: float
    count: int
    min: float | None
    max: float | None

    @classmethod
    def empty(cls) -> TokenStats:
        """Return the empty statistics record."""
        return cls(0.0, 0, None, None)

    @classmethod
    def from_masked(cls, values: torch.Tensor, mask: torch.Tensor) -> TokenStats:
        """Build statistics from normalized values and an action mask.

        Args:
            values (torch.Tensor): Flat advantage values.
            mask (torch.Tensor): Boolean mask selecting kept tokens.

        Returns:
            TokenStats: Statistics over the kept tokens.
        """
        kept = values[mask.to(device=values.device).bool()]
        if kept.numel() == 0:
            return cls.empty()
        return cls(
            float(kept.float().sum().item()), int(kept.numel()), float(kept.min().item()), float(kept.max().item())
        )

    @classmethod
    def merge(cls, stats: Iterable[TokenStats]) -> TokenStats:
        """Merge per-replica statistics, ignoring empty extrema.

        Args:
            stats (Iterable[TokenStats]): Per-rank statistics.

        Returns:
            TokenStats: Merged statistics.
        """
        valid: Sequence[TokenStats] = [item for item in stats if item.count]
        if not valid:
            return cls.empty()
        return cls(
            sum(item.sum for item in valid),
            sum(item.count for item in valid),
            min(item.min for item in valid if item.min is not None),
            max(item.max for item in valid if item.max is not None),
        )

    @property
    def mean(self) -> float:
        """Mean kept-token advantage (0.0 when empty)."""
        return self.sum / self.count if self.count else 0.0
