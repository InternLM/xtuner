from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from xtuner.v1.rl.weight_update.data import RolloutWeightUpdateTarget

    from .rollout_topology import RolloutTopology
    from .worker import RolloutWorker

__all__ = [
    "RolloutWorkerRegistry",
    "WorkerGroup",
    "WorkerLifecycleState",
    "WorkerSnapshot",
]


class WorkerLifecycleState(str, Enum):
    # Can serve rollout generation and control requests.
    ACTIVE = "active"
    # Not serving rollout requests; the rollout server may still hold resources.
    INACTIVE = "inactive"
    # Temporarily owned by recovery shutdown/init/check_health.
    RECOVERING = "recovering"


@dataclass(frozen=True)
class WorkerSnapshot:
    """Read-only snapshot for one rollout server process."""

    # Worker rank that owns the runtime snapshot.
    rank: int
    # Ray actor handle for the rollout worker.
    actor: RolloutWorker
    # Base URL of the rollout server process.
    url: str
    # Session server URL used only by proxy/session routing.
    session_url: str | None = None
    # Whether this worker can receive rollout generation requests.
    is_request_entrypoint: bool = True
    # Current lifecycle state observed by registry and health manager.
    lifecycle_state: WorkerLifecycleState = WorkerLifecycleState.ACTIVE

    def is_active(self) -> bool:
        return self.lifecycle_state is WorkerLifecycleState.ACTIVE


@dataclass(frozen=True)
class WorkerGroup:
    # Worker ranks that share one lifecycle action.
    ranks: tuple[int, ...]
    # Runtime snapshots for registered workers in this lifecycle group.
    workers: tuple[WorkerSnapshot, ...]


class RolloutWorkerRegistry:
    """Own runtime rollout worker state and expose consistent query
    snapshots."""

    def __init__(
        self,
        *,
        rollout_topology: RolloutTopology,
    ):
        """Initialize an empty registry with the rollout topology."""
        self._rollout_topology = rollout_topology
        self._workers: dict[int, WorkerSnapshot] = {}
        self._lock = threading.RLock()

    def register_started_server(
        self,
        *,
        rank: int,
        actor: RolloutWorker,
        server_url: str,
        session_url: str | None = None,
        lifecycle_state: WorkerLifecycleState = WorkerLifecycleState.ACTIVE,
    ) -> None:
        """Register one worker actor after its rollout server process has
        started."""
        with self._lock:
            self._workers[rank] = WorkerSnapshot(
                rank=rank,
                actor=actor,
                url=server_url,
                session_url=session_url,
                is_request_entrypoint=self._rollout_topology.is_request_entrypoint_rank(rank),
                lifecycle_state=lifecycle_state,
            )

    def all_workers(self) -> tuple[WorkerSnapshot, ...]:
        """Return a stable rank-ordered snapshot of all registered server-
        process workers."""
        with self._lock:
            return tuple(self._workers[rank] for rank in sorted(self._workers))

    def inactive_workers(self) -> tuple[WorkerSnapshot, ...]:
        """Return registered workers that cannot currently serve rollout
        traffic."""
        with self._lock:
            return tuple(worker for worker in self.all_workers() if not worker.is_active())

    def all_actors(self) -> tuple[RolloutWorker, ...]:
        """Return actor handles for all registered workers, including inactive
        ones."""
        with self._lock:
            return tuple(info.actor for info in self._workers.values())

    def active_workers(self) -> tuple[WorkerSnapshot, ...]:
        """Return workers whose lifecycle state is active."""
        with self._lock:
            return tuple(worker for worker in self._workers.values() if worker.is_active())

    def active_entrypoints(self) -> tuple[WorkerSnapshot, ...]:
        """Return active workers that can receive rollout generation
        requests."""
        with self._lock:
            return tuple(
                worker for worker in self._workers.values() if worker.is_active() and worker.is_request_entrypoint
            )

    def active_entrypoint_by_rank(self, rank: int) -> WorkerSnapshot | None:
        """Return the active request entrypoint for a rank, if that rank is
        usable."""
        with self._lock:
            worker = self._workers.get(rank)
            if worker is None or not worker.is_active() or not worker.is_request_entrypoint:
                return None
            return worker

    def lifecycle_groups(self) -> tuple[tuple[int, ...], ...]:
        """Return registered lifecycle groups in rank order."""
        with self._lock:
            return tuple(sorted(self._rollout_topology.lifecycle_groups()))

    def _build_worker_groups(self) -> dict[tuple[int, ...], WorkerGroup]:
        grouped_ranks = {
            self._rollout_topology.lifecycle_group_for_server_rank(worker.rank) for worker in self._workers.values()
        }
        return {
            group_ranks: WorkerGroup(
                ranks=group_ranks,
                workers=tuple(self._workers[rank] for rank in group_ranks if rank in self._workers),
            )
            for group_ranks in grouped_ranks
        }

    def claim_inactive_groups_for_recovery(self) -> tuple[WorkerGroup, ...]:
        """Claim non-active worker groups by moving them to recovering
        state."""
        with self._lock:
            worker_groups = self._build_worker_groups()
            inactive_groups = [
                group
                for group in worker_groups.values()
                if any(worker.lifecycle_state is not WorkerLifecycleState.ACTIVE for worker in group.workers)
            ]
            sorted_groups = tuple(sorted(inactive_groups, key=lambda group: group.ranks))
            for group in sorted_groups:
                for rank in group.ranks:
                    worker = self._workers.get(rank)
                    if worker is not None:
                        self._workers[rank] = replace(worker, lifecycle_state=WorkerLifecycleState.RECOVERING)
            return sorted_groups

    def mark_unhealthy_ranks(self, ranks: set[int]) -> tuple[WorkerGroup, ...]:
        """Mark every lifecycle group containing a failed rank as inactive."""
        with self._lock:
            failed_group_ranks = {
                self._rollout_topology.lifecycle_group_for_server_rank(rank) for rank in ranks if rank in self._workers
            }
            for group_ranks in failed_group_ranks:
                for rank in group_ranks:
                    worker = self._workers.get(rank)
                    if worker is not None:
                        self._workers[rank] = replace(worker, lifecycle_state=WorkerLifecycleState.INACTIVE)
            worker_groups = self._build_worker_groups()
            return tuple(
                worker_groups[group_ranks]
                for group_ranks in sorted(failed_group_ranks)
                if group_ranks in worker_groups
            )

    def set_group_recovery_result(
        self,
        group: WorkerGroup,
        *,
        recovered: bool,
    ) -> WorkerGroup | None:
        """Apply the final lifecycle state for a completed group recovery
        attempt and return the updated group snapshot."""
        with self._lock:
            lifecycle_state = WorkerLifecycleState.ACTIVE if recovered else WorkerLifecycleState.INACTIVE
            for rank in group.ranks:
                worker = self._workers.get(rank)
                if worker is not None:
                    self._workers[rank] = replace(worker, lifecycle_state=lifecycle_state)
            worker_groups = self._build_worker_groups()
            return worker_groups.get(group.ranks)

    def weight_update_targets(self) -> tuple[RolloutWeightUpdateTarget, ...]:
        """Return weight-update targets resolved with current runtime state."""
        from xtuner.v1.rl.weight_update.data import RolloutWeightUpdateTarget

        with self._lock:
            targets: list[RolloutWeightUpdateTarget] = []
            for server in self._rollout_topology.weight_update_endpoint_processes():
                worker = self._workers.get(server.worker_rank)
                if worker is None:
                    raise RuntimeError(
                        f"Rollout weight update endpoint rank={server.worker_rank} has not been registered."
                    )
                targets.append(
                    RolloutWeightUpdateTarget(
                        endpoint_rank=server.worker_rank,
                        update_ranks=server.weight_update_ranks,
                        server_url=worker.url,
                        lifecycle_state=worker.lifecycle_state.value,
                    )
                )
            return tuple(sorted(targets, key=lambda target: target.endpoint_rank))
