"""Data contracts for rollout requests spanning model weight updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CrossWeightUpdateRolloutConfig:
    """Keep live Agent rollouts running across model weight updates.

    Admitted LLM requests settle before switching weights; later requests may use
    the new version. Live Agent/tool state is not serialized in checkpoints.

    Args:
        wait_timeout (float): Maximum time an LLM request may wait for admission to reopen.
        freeze_timeout (float): Budget for freezing topology and for the install/resume/reopen control batch.
        drain_timeout (float): Maximum time to wait for already admitted requests to settle.
    """

    wait_timeout: float = 7200
    freeze_timeout: float = 30
    drain_timeout: float = 3900

    # TODO(timeout-reconciliation): wait_timeout bounds only per-request admission (a timeout fails that Agent
    # request, not training). freeze_timeout currently bounds both individual pause stages and the shared
    # install/resume/reopen batch, and a drain timeout becomes a permanent barrier failure. Future work should split
    # these budgets and keep waiting/observing inside the same sealed barrier instead of terminating training on a
    # wait timeout. (PR #2297 review.)

    def __post_init__(self) -> None:
        for name in ("wait_timeout", "freeze_timeout", "drain_timeout"):
            if getattr(self, name) <= 0:
                raise ValueError(f"CrossWeightUpdateRolloutConfig.{name} must be positive")


@dataclass(frozen=True)
class RequestBarrier:
    """Fixed entrypoint snapshot that a pause sealed.

    Args:
        epoch (int): Monotonic admission epoch.
        endpoints (tuple[str, ...]): SessionServer URLs captured under the freeze.
        servers (tuple[Any, ...]): SessionServer actor handles for the same endpoints.
    """

    epoch: int
    endpoints: tuple[str, ...]
    servers: tuple[Any, ...]
