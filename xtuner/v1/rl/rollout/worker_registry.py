from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Iterable, TypedDict


if TYPE_CHECKING:
    from .worker import RolloutConfig, RolloutWorker

__all__ = [
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

    actor: RolloutWorker
    url: str
    session_url: str | None = None
    lifecycle_state: WorkerLifecycleState = WorkerLifecycleState.ACTIVE
    lifecycle_group_ranks: tuple[int, ...] = ()
    is_request_entrypoint: bool = True
    rank: int = -1

    def __post_init__(self) -> None:
        lifecycle_state = (
            WorkerLifecycleState.ACTIVE if self.lifecycle_state is None else WorkerLifecycleState(self.lifecycle_state)
        )
        object.__setattr__(self, "lifecycle_state", lifecycle_state)
        object.__setattr__(self, "lifecycle_group_ranks", tuple(self.lifecycle_group_ranks))

    def is_active(self) -> bool:
        return self.lifecycle_state is WorkerLifecycleState.ACTIVE


@dataclass(frozen=True)
class WorkerGroup:
    ranks: tuple[int, ...]
    workers: tuple[WorkerSnapshot, ...]


class RolloutWorkerMetadata(TypedDict):
    """Legacy rollout worker metadata consumed by trainer/update-weight
    code."""

    engine_rank_mesh_array: list[list[int]]
    server_url_dict: dict[int, str]
    rollout_config: RolloutConfig
    worker_server_urls_status: dict[str, bool]
    worker_session_url_dict: dict[int, str]
    worker_session_urls_status: dict[str, bool]


def _build_worker_groups(workers: Iterable[WorkerSnapshot]) -> dict[tuple[int, ...], WorkerGroup]:
    grouped_workers: dict[tuple[int, ...], list[WorkerSnapshot]] = {}
    for worker in workers:
        group_ranks = worker.lifecycle_group_ranks or (worker.rank,)
        grouped_workers.setdefault(group_ranks, []).append(worker)

    return {
        group_ranks: WorkerGroup(
            ranks=group_ranks,
            workers=tuple(sorted(group_workers, key=lambda worker: worker.rank)),
        )
        for group_ranks, group_workers in grouped_workers.items()
    }


class RolloutWorkerRegistry:
    """Own runtime rollout worker state and expose consistent query
    snapshots."""

    def __init__(
        self,
        *,
        engine_rank_mesh_array: list[list[int]],
        rollout_config: RolloutConfig,
    ):
        """Initialize an empty registry with the training-side metadata
        projection."""
        self._engine_rank_mesh_array = [list(engine_ranks) for engine_ranks in engine_rank_mesh_array]
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
        lifecycle_group_ranks: tuple[int, ...] = (),
        is_request_entrypoint: bool = True,
    ) -> None:
        """Register one worker actor after its rollout server process has
        started."""
        with self._lock:
            self._workers[rank] = WorkerSnapshot(
                rank=rank,
                actor=actor,
                url=server_url,
                session_url=session_url,
                lifecycle_group_ranks=lifecycle_group_ranks or (rank,),
                is_request_entrypoint=is_request_entrypoint,
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

    def claim_inactive_groups_for_recovery(self) -> tuple[WorkerGroup, ...]:
        """Claim non-active worker groups by moving them to recovering
        state."""
        with self._lock:
            worker_groups = _build_worker_groups(self._workers.values())
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
                worker.lifecycle_group_ranks or (worker.rank,)
                for rank, worker in self._workers.items()
                if rank in ranks
            }
            for group_ranks in failed_group_ranks:
                for rank in group_ranks:
                    worker = self._workers.get(rank)
                    if worker is not None:
                        self._workers[rank] = replace(worker, lifecycle_state=WorkerLifecycleState.INACTIVE)
            worker_groups = _build_worker_groups(self._workers.values())
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
            worker_groups = _build_worker_groups(self._workers.values())
            return worker_groups.get(group.ranks)

    def training_metadata_snapshot(self) -> RolloutWorkerMetadata:
        """Build the legacy trainer/update-weight metadata from one registry
        snapshot."""
        with self._lock:
            request_entrypoints = {rank: info for rank, info in self._workers.items() if info.is_request_entrypoint}
            worker_server_urls_map = {rank: info.url for rank, info in request_entrypoints.items()}
            worker_server_urls_status = {info.url: info.is_active() for info in request_entrypoints.values()}
            worker_session_url_dict: dict[int, str] = {}
            worker_session_urls_status: dict[str, bool] = {}
            for rank, info in request_entrypoints.items():
                if info.session_url is None:
                    continue
                worker_session_url_dict[rank] = info.session_url
                worker_session_urls_status[info.session_url] = info.is_active()

            return {
                "engine_rank_mesh_array": [list(engine_ranks) for engine_ranks in self._engine_rank_mesh_array],
                "server_url_dict": worker_server_urls_map,
                "rollout_config": self._rollout_config,
                "worker_server_urls_status": worker_server_urls_status,
                "worker_session_url_dict": worker_session_url_dict,
                "worker_session_urls_status": worker_session_urls_status,
            }
