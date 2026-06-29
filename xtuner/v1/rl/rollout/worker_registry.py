from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from .rollout_topology import RolloutTopology
    from .worker import RolloutConfig, RolloutWorker

__all__ = [
    "RolloutWorkerEndpointMetadata",
    "RolloutWorkerMetadata",
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

    rank: int
    actor: RolloutWorker
    url: str
    session_url: str | None = None
    is_request_entrypoint: bool = True
    lifecycle_state: WorkerLifecycleState = WorkerLifecycleState.ACTIVE

    def is_active(self) -> bool:
        return self.lifecycle_state is WorkerLifecycleState.ACTIVE


@dataclass(frozen=True)
class WorkerGroup:
    ranks: tuple[int, ...]
    workers: tuple[WorkerSnapshot, ...]


@dataclass(frozen=True)
class RolloutWorkerEndpointMetadata:
    """URL and lifecycle state for one request-serving rollout endpoint."""

    rank: int
    server_url: str
    session_url: str | None
    lifecycle_state: WorkerLifecycleState

    @property
    def is_active(self) -> bool:
        return self.lifecycle_state is WorkerLifecycleState.ACTIVE


@dataclass(frozen=True)
class RolloutWorkerMetadata:
    """Structured rollout worker metadata consumed by trainer/update-weight
    code."""

    rollout_config: RolloutConfig
    training_engine_mesh: tuple[tuple[int, ...], ...]
    request_endpoints: tuple[RolloutWorkerEndpointMetadata, ...]

    def to_legacy(self) -> dict[str, Any]:
        """Serialize to the current trainer-facing rollout metadata dict."""
        return {
            "engine_rank_mesh_array": [list(engine_ranks) for engine_ranks in self.training_engine_mesh],
            "server_url_dict": {endpoint.rank: endpoint.server_url for endpoint in self.request_endpoints},
            "rollout_config": self.rollout_config,
            "worker_server_urls_status": {
                endpoint.server_url: endpoint.is_active for endpoint in self.request_endpoints
            },
            "worker_session_url_dict": {
                endpoint.rank: endpoint.session_url
                for endpoint in self.request_endpoints
                if endpoint.session_url is not None
            },
            "worker_session_urls_status": {
                endpoint.session_url: endpoint.is_active
                for endpoint in self.request_endpoints
                if endpoint.session_url is not None
            },
        }


class RolloutWorkerRegistry:
    """Own runtime rollout worker state and expose consistent query
    snapshots."""

    def __init__(
        self,
        *,
        rollout_topology: RolloutTopology,
        rollout_config: RolloutConfig,
    ):
        """Initialize an empty registry with the rollout topology."""
        self._rollout_topology = rollout_topology
        self._rollout_config = rollout_config
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

    def metadata(self) -> RolloutWorkerMetadata:
        """Build trainer/update-weight metadata from one registry snapshot."""
        with self._lock:
            request_endpoints = tuple(
                RolloutWorkerEndpointMetadata(
                    rank=worker.rank,
                    server_url=worker.url,
                    session_url=worker.session_url,
                    lifecycle_state=worker.lifecycle_state,
                )
                for worker in self.all_workers()
                if worker.is_request_entrypoint
            )
            return RolloutWorkerMetadata(
                rollout_config=self._rollout_config,
                training_engine_mesh=self._rollout_topology.training_engine_mesh,
                request_endpoints=request_endpoints,
            )
