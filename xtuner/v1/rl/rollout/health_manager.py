import asyncio
import os
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import ray

from xtuner.v1.utils import get_logger

from .utils import WorkerLifecycleState


if TYPE_CHECKING:
    from .controller import WorkerInfo
    from .worker import RolloutConfig, RolloutWorker

ROLLOUT_RAY_GET_TIMEOUT = int(os.getenv("XTUNER_ROLLOUT_RAY_GET_TIMEOUT", str(5 * 3600)))  # default 5 hours
logger = get_logger()

__all__ = [
    "ROLLOUT_RAY_GET_TIMEOUT",
    "RolloutHealthManager",
    "WorkerSnapshot",
    "check_worker_health",
    "shutdown_worker",
]


@dataclass(frozen=True)
class WorkerSnapshot:
    rank: int
    actor: "RolloutWorker"
    url: str
    session_url: str | None
    lifecycle_state: WorkerLifecycleState
    active: bool
    lifecycle_group_ranks: tuple[int, ...]
    is_request_entrypoint: bool


@dataclass(frozen=True)
class _WorkerGroupSnapshot:
    ranks: tuple[int, ...]
    workers: tuple[WorkerSnapshot, ...]


class RolloutHealthManager:
    """Own worker health state and recovery after controller startup.

    RolloutController creates workers, launches them the first time, and routes requests. RolloutHealthManager only
    reads that WorkerInfo table, updates lifecycle_state, runs health checks, and restarts failed lifecycle groups.
    Worker actors still own backend-specific server start/stop/probe/generate.
    """

    def __init__(
        self,
        config: "RolloutConfig",
        workers_info: dict[int, "WorkerInfo"],
        worker_infos_lock: Optional[threading.RLock] = None,
    ):
        self._workers_info = workers_info
        self._worker_infos_lock = worker_infos_lock or threading.RLock()
        self._check_interval = config.health_check_interval_seconds
        self._check_failure_threshold = config.health_check_failure_threshold
        self._stop_event: Optional[threading.Event] = None
        self._pause_event: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None
        self._operation_lock = threading.Lock()
        self._stopped = False

    def start(self) -> None:
        health_thread_alive = self._thread is not None and self._thread.is_alive()
        if health_thread_alive:
            return

        self._stopped = False
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("RolloutHealthManager started.")

    def stop(self) -> None:
        if not self._thread:
            return

        assert self._stop_event is not None
        self._stopped = True
        self._stop_event.set()
        if self._pause_event:
            self._pause_event.clear()

        if self._thread:
            self._thread.join()

        self._thread = None
        self._stop_event = None
        self._pause_event = None
        logger.info("RolloutHealthManager stopped.")

    def pause(self) -> None:
        if self._pause_event is None:
            return
        self._pause_event.set()
        logger.info("RolloutHealthManager paused.")

    def is_paused(self) -> bool:
        return self._pause_event is None or self._pause_event.is_set()

    def resume(self) -> None:
        if self._pause_event is None:
            return
        self._pause_event.clear()
        logger.info("RolloutHealthManager resumed.")

    def snapshot_workers(self) -> dict[int, WorkerSnapshot]:
        """Return immutable worker state for callers that only need to query
        rollout workers."""
        with self._worker_infos_lock:
            return {
                rank: WorkerSnapshot(
                    rank=rank,
                    actor=info.actor,
                    url=info.url,
                    session_url=getattr(info, "session_url", None),
                    lifecycle_state=info.lifecycle_state,
                    active=info.is_active(),
                    lifecycle_group_ranks=tuple(getattr(info, "lifecycle_group_ranks", ()) or ()),
                    is_request_entrypoint=bool(getattr(info, "is_request_entrypoint", True)),
                )
                for rank, info in self._workers_info.items()
            }

    def recover_failed_workers(self, reason: str) -> dict[int, bool]:
        """Synchronously restart inactive or incomplete worker groups and
        return per-rank recovery results."""
        if self._is_stopping():
            message = f"RolloutHealthManager is stopped; reject recovery: reason={reason}."
            logger.warning(message)
            raise RuntimeError(message)

        with self._operation_lock:
            worker_groups = self._snapshot_worker_groups()
            failed_groups = []
            for group in worker_groups.values():
                is_incomplete = len(group.workers) != len(group.ranks)
                if is_incomplete or any(not worker.active for worker in group.workers):
                    failed_groups.append(group)
            if not failed_groups:
                logger.info(f"No failed rollout workers detected during recovery: reason={reason}.")
                return {}

            for group in sorted(failed_groups, key=lambda group: group.ranks):
                failed_ranks = sorted(worker.rank for worker in group.workers if not worker.active)
                existing_ranks = {worker.rank for worker in group.workers}
                missing_ranks = sorted(rank for rank in group.ranks if rank not in existing_ranks)
                if missing_ranks:
                    logger.error(
                        f"Detected incomplete rollout worker group: group_ranks={group.ranks}, "
                        f"missing_ranks={missing_ranks}, failed_ranks={failed_ranks}, reason={reason}."
                    )
                elif len(group.ranks) > 1:
                    related_restart_ranks = [rank for rank in group.ranks if rank not in failed_ranks]
                    logger.warning(
                        f"Detected failed rollout worker ranks={failed_ranks}; "
                        f"restart_group_ranks={group.ranks}, related_restart_ranks={related_restart_ranks}, "
                        f"reason={reason}."
                    )
                else:
                    logger.warning(
                        f"Detected failed rollout worker rank={failed_ranks[0]}. Initiating recovery: reason={reason}."
                    )

            for group in failed_groups:
                self._set_group_lifecycle_state(group.ranks, WorkerLifecycleState.RECOVERING)

            results: dict[int, bool] = {}
            sorted_failed_groups = sorted(failed_groups, key=lambda group: group.ranks)
            for group_index, group in enumerate(sorted_failed_groups):
                if self._is_stopping():
                    for remaining_group in sorted_failed_groups[group_index:]:
                        self._set_group_lifecycle_state(remaining_group.ranks, WorkerLifecycleState.INACTIVE)
                        for rank in remaining_group.ranks:
                            results[rank] = False
                    return results

                is_recovered = self._restart_worker_group(group)
                for rank in group.ranks:
                    results[rank] = is_recovered
                if self._is_stopping():
                    if is_recovered:
                        self._shutdown_worker_group(group, wait_server_down=False, best_effort=True)
                    self._set_group_lifecycle_state(group.ranks, WorkerLifecycleState.INACTIVE)
                    results.update({rank: False for rank in group.ranks})
                    for remaining_group in sorted_failed_groups[group_index + 1 :]:
                        self._set_group_lifecycle_state(remaining_group.ranks, WorkerLifecycleState.INACTIVE)
                        for rank in remaining_group.ranks:
                            results[rank] = False
                    return results

                next_lifecycle_state = WorkerLifecycleState.ACTIVE if is_recovered else WorkerLifecycleState.INACTIVE
                self._set_group_lifecycle_state(group.ranks, next_lifecycle_state)
            return results

    def ensure_workers_healthy_before_training(self) -> dict[int, bool]:
        """Run a training barrier health check and recover failed groups before
        training resumes."""
        self._check_active_workers_and_mark_failed_groups()
        return self.recover_failed_workers(reason="before_training")

    def run_once(self) -> None:
        logger.debug("RolloutHealthManager running health checks for all workers.")
        checked_active_count = self._check_active_workers_and_mark_failed_groups()
        has_active_workers = any(worker.active for worker in self.snapshot_workers().values())
        if has_active_workers or self._is_stopping():
            return

        if checked_active_count == 0:
            logger.warning("No active rollout workers before health check. Initiating recovery.")
            reason = "no_active_workers_before_health_check"
        else:
            logger.warning("All rollout workers failed after health check. Initiating recovery.")
            reason = "all_workers_failed_after_health_check"
        self.recover_failed_workers(reason=reason)

    def _is_stopping(self) -> bool:
        """Return whether the health manager is stopping or already stopped."""
        return self._stopped or (self._stop_event is not None and self._stop_event.is_set())

    def _check_active_workers_and_mark_failed_groups(self) -> int:
        """Health-check active workers and mark any failed lifecycle group
        inactive."""
        with self._operation_lock:
            workers_snapshot = self.snapshot_workers()
            workers_to_check = [worker for worker in workers_snapshot.values() if worker.active]
            if not workers_to_check:
                return 0

            tasks = [
                check_worker_health(
                    worker.actor,
                    worker.rank,
                    worker.url,
                    worker.active,
                    worker.is_request_entrypoint,
                    self._check_failure_threshold,
                )
                for worker in workers_to_check
            ]

            async def _run_checks() -> list[bool]:
                """Run active worker health checks concurrently."""
                return await asyncio.gather(*tasks)

            check_results = asyncio.run(_run_checks())
            failed_groups: set[tuple[int, ...]] = set()
            for worker, is_healthy in zip(workers_to_check, check_results):
                if not is_healthy:
                    logger.warning(f"Worker {worker.rank} failed health check. Marking as inactive.")
                    failed_groups.add(worker.lifecycle_group_ranks or (worker.rank,))
                else:
                    logger.debug(f"[RolloutHealthManager] Worker {worker.rank} passed health check.")

            for group_ranks in failed_groups:
                self._set_group_lifecycle_state(group_ranks, WorkerLifecycleState.INACTIVE)

            return len(workers_to_check)

    def _run_loop(self) -> None:
        assert self._stop_event is not None and self._pause_event is not None
        logger.info("RolloutHealthManager loop started.")

        while not self._stop_event.is_set():
            while self._pause_event.is_set() and not self._stop_event.is_set():
                self._stop_event.wait(timeout=0.5)

            if self._stop_event.is_set():
                break

            if not self._pause_event.is_set() and not self._stop_event.is_set():
                try:
                    self.run_once()
                except RuntimeError:
                    if self._is_stopping():
                        break
                    logger.exception("RolloutHealthManager run_once failed.")
                except Exception:
                    logger.exception("RolloutHealthManager run_once failed.")

            if self._stop_event.wait(self._check_interval):
                break

    def _snapshot_worker_groups(self) -> dict[tuple[int, ...], _WorkerGroupSnapshot]:
        """Group worker snapshots by lifecycle group so recovery operates on
        whole groups."""
        workers_snapshot = self.snapshot_workers()
        grouped_workers: dict[tuple[int, ...], list[WorkerSnapshot]] = {}
        for worker in workers_snapshot.values():
            group_ranks = worker.lifecycle_group_ranks or (worker.rank,)
            missing_ranks = [rank for rank in group_ranks if rank not in workers_snapshot]
            if missing_ranks:
                logger.error(
                    f"Rollout worker rank={worker.rank} has incomplete lifecycle group metadata: "
                    f"group_ranks={group_ranks}, missing_ranks={missing_ranks}."
                )
            grouped_workers.setdefault(group_ranks, []).append(worker)

        return {
            group_ranks: _WorkerGroupSnapshot(
                ranks=group_ranks,
                workers=tuple(sorted(workers, key=lambda worker: worker.rank)),
            )
            for group_ranks, workers in grouped_workers.items()
        }

    def _set_group_lifecycle_state(
        self,
        group_ranks: tuple[int, ...],
        lifecycle_state: WorkerLifecycleState,
    ) -> None:
        """Update rollout worker lifecycle for every known rank in one
        group."""
        with self._worker_infos_lock:
            for rank in group_ranks:
                if rank in self._workers_info:
                    self._workers_info[rank].lifecycle_state = lifecycle_state

    def _shutdown_worker_group(
        self,
        group: _WorkerGroupSnapshot,
        *,
        wait_server_down: bool,
        best_effort: bool,
    ) -> bool:
        """Shutdown every worker in one group and aggregate per-worker shutdown
        results."""
        shutdown_succeeded = True
        for worker in group.workers:
            if not shutdown_worker(worker, wait_server_down=wait_server_down, best_effort=best_effort):
                shutdown_succeeded = False
        return best_effort or shutdown_succeeded

    def _restart_worker_group(self, group: _WorkerGroupSnapshot) -> bool:
        """Shutdown, init, health-check, and inference-check one complete
        worker group."""
        if not group.workers or len(group.workers) != len(group.ranks):
            logger.error(f"Cannot restart incomplete rollout worker group: ranks={group.ranks}.")
            return False
        if self._is_stopping():
            return False

        if not self._shutdown_worker_group(group, wait_server_down=True, best_effort=False):
            return False
        if self._is_stopping():
            return False

        try:
            init_results = ray.get(
                [
                    # init() reuses the immutable launch spec cached on each actor
                    # during controller startup, including placement bundles and dist addr.
                    worker.actor.init.remote()  # type: ignore[attr-defined]
                    for worker in group.workers
                ],
                timeout=ROLLOUT_RAY_GET_TIMEOUT,
            )
            if self._is_stopping():
                self._shutdown_worker_group(group, wait_server_down=False, best_effort=True)
                return False
            if len(init_results) != len(group.workers):
                logger.error(
                    f"Restarted rollout worker group ranks={group.ranks} returned {len(init_results)} init results, "
                    f"expected {len(group.workers)}."
                )
                self._shutdown_worker_group(group, wait_server_down=False, best_effort=True)
                return False

            for worker, init_result in zip(group.workers, init_results):
                init_rank, init_url = init_result
                if init_rank != worker.rank or init_url != worker.url:
                    logger.error(
                        f"Rollout worker restart returned unexpected endpoint: rank={worker.rank}, "
                        f"init_rank={init_rank}, expected_url={worker.url}, init_url={init_url}."
                    )
                    self._shutdown_worker_group(group, wait_server_down=False, best_effort=True)
                    return False

            health_results = ray.get(
                [worker.actor.check_health.remote() for worker in group.workers],  # type: ignore[attr-defined]
                timeout=ROLLOUT_RAY_GET_TIMEOUT,
            )
            if self._is_stopping():
                self._shutdown_worker_group(group, wait_server_down=False, best_effort=True)
                return False
            unhealthy_ranks = [
                worker.rank for worker, is_healthy in zip(group.workers, health_results) if not is_healthy
            ]
            if unhealthy_ranks:
                logger.error(
                    f"Restarted rollout worker group ranks={group.ranks} has unhealthy ranks={unhealthy_ranks}."
                )
                self._shutdown_worker_group(group, wait_server_down=False, best_effort=True)
                return False

            generate_not_ready_urls = self._check_worker_group_generate_ready_after_restart(group)
            if generate_not_ready_urls:
                logger.error(
                    f"Restarted rollout worker group ranks={group.ranks} has request-serving endpoints not "
                    f"generate-ready: {generate_not_ready_urls}."
                )
                self._shutdown_worker_group(group, wait_server_down=False, best_effort=True)
                return False

            logger.info(f"Successfully restarted rollout worker group ranks={group.ranks}.")
            return True
        except Exception as e:
            logger.error(f"Failed to restart rollout worker group ranks={group.ranks}: {e}")
            self._shutdown_worker_group(group, wait_server_down=False, best_effort=True)
            return False

    def _check_worker_group_generate_ready_after_restart(
        self,
        group: _WorkerGroupSnapshot,
        *,
        max_attempts: int = 60,
        retry_interval_seconds: float = 5.0,
    ) -> list[str]:
        """Check generate readiness through request-serving backend
        endpoints."""
        request_workers = [worker for worker in group.workers if worker.is_request_entrypoint]
        pending_workers = {worker.rank: worker for worker in request_workers}
        for attempt in range(1, max_attempts + 1):
            for rank, worker in list(pending_workers.items()):
                health_ref = None
                try:
                    health_ref = worker.actor.check_health_generate.remote()  # type: ignore[attr-defined]
                    is_generate_ready = ray.get(health_ref, timeout=60)
                except Exception as e:
                    if health_ref is not None:
                        try:
                            ray.cancel(health_ref, force=True)
                        except Exception:
                            pass
                    logger.warning(
                        f"Restarted rollout worker rank={worker.rank} generate readiness check failed "
                        f"attempt={attempt}/{max_attempts}, url={worker.url}: {e}"
                    )
                    continue

                if is_generate_ready:
                    del pending_workers[rank]
                else:
                    logger.warning(
                        f"Restarted rollout worker rank={worker.rank} generate readiness check returned false "
                        f"attempt={attempt}/{max_attempts}, url={worker.url}."
                    )
            if not pending_workers:
                return []
            if attempt < max_attempts:
                time.sleep(retry_interval_seconds)
        return [worker.url for worker in pending_workers.values()]


def shutdown_worker(
    worker: WorkerSnapshot,
    *,
    wait_server_down: bool,
    best_effort: bool,
) -> bool:
    """Shutdown one worker, optionally waiting until it stops serving health
    checks."""
    max_wait_attempts = 60
    retry_interval_seconds = 5.0
    shutdown_succeeded = True
    try:
        ray.get(worker.actor.shutdown.remote(), timeout=60)  # type: ignore[attr-defined]
    except Exception as e:
        shutdown_succeeded = False
        log = logger.warning if best_effort else logger.error
        log(f"Shutdown failed for rollout worker rank={worker.rank}, url={worker.url}: {e}")
        return best_effort or shutdown_succeeded

    if wait_server_down:
        server_down = False
        for attempt in range(1, max_wait_attempts + 1):
            try:
                is_healthy = ray.get(worker.actor.check_health.remote(), timeout=60)  # type: ignore[attr-defined]
            except Exception:
                server_down = True
                break
            if not is_healthy:
                server_down = True
                break
            if attempt < max_wait_attempts:
                logger.warning(
                    f"Rollout worker rank={worker.rank} server still responds after shutdown "
                    f"attempt={attempt}/{max_wait_attempts}, url={worker.url}."
                )
                time.sleep(retry_interval_seconds)
        if not server_down:
            logger.error(f"Rollout worker rank={worker.rank} server did not stop after shutdown: url={worker.url}.")
            shutdown_succeeded = False
    return best_effort or shutdown_succeeded


async def check_worker_health(
    worker: "RolloutWorker",
    rank: int,
    url: str,
    is_active: bool,
    is_request_entrypoint: bool,
    failure_threshold: int = 3,
) -> bool:
    if worker is None or not is_active:
        logger.warning("Worker has no actor reference or is marked inactive.")
        return False
    health_method_name = "check_health_generate" if is_request_entrypoint else "check_health"
    failing_count = 0
    while failing_count < failure_threshold:
        try:
            health_method = getattr(worker, health_method_name)
            health_status = await health_method.remote()  # type: ignore[attr-defined]
            if health_status:
                return True
            failing_count += 1
            logger.warning(f"{health_method_name} failed for worker {rank} at {url}. Failure count: {failing_count}")
        except Exception as e:
            failing_count += 1
            logger.error(
                f"Exception during {health_method_name} for worker {rank} at {url}: {e}. "
                f"Failure count: {failing_count}"
            )
    return False
