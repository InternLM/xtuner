from __future__ import annotations

import asyncio
import os
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import ray

from xtuner.v1.utils import get_logger

from .worker_registry import RolloutWorkerRegistry, WorkerGroup, WorkerSnapshot


if TYPE_CHECKING:
    from .worker import RolloutConfig

ROLLOUT_RAY_GET_TIMEOUT = int(os.getenv("XTUNER_ROLLOUT_RAY_GET_TIMEOUT", str(5 * 3600)))  # default 5 hours
ROLLOUT_RECOVERY_MAX_PARALLEL_GROUPS = 4
HEALTH_MANAGER_STOP_JOIN_TIMEOUT = 30.0
SHUTDOWN_SERVER_DOWN_MAX_ATTEMPTS = 60
logger = get_logger()

__all__ = [
    "HEALTH_MANAGER_STOP_JOIN_TIMEOUT",
    "ROLLOUT_RAY_GET_TIMEOUT",
    "RolloutHealthManager",
    "RolloutWorkerLifecycleListener",
]


class RolloutWorkerLifecycleListener(Protocol):
    def on_worker_group_inactive(self, group: WorkerGroup) -> None: ...

    def on_worker_group_recovered(self, group: WorkerGroup) -> None: ...


class _HealthManagerStopping(InterruptedError):
    """Raised at lifecycle checkpoints when the health manager is stopping."""


@dataclass
class _WorkerHealthFailureTracker:
    """Track per-rank health-check failures and decide when a rank fails.

    Periodic checks call update_failed_ranks() to apply the configured threshold. Explicit shutdown barriers call
    mark_failed_ranks() to fail unhealthy ranks immediately while still keeping failure-count bookkeeping in one place.
    """

    threshold: int
    failure_counts: dict[int, int] = field(default_factory=dict)

    def clear(self, ranks: Iterable[int]) -> None:
        for rank in ranks:
            self.failure_counts.pop(rank, None)

    def _record_failure(self, rank: int) -> int:
        failure_count = self.failure_counts.get(rank, 0) + 1
        self.failure_counts[rank] = failure_count
        return failure_count

    def update_failed_ranks(self, worker_health_results: dict[int, bool]) -> set[int]:
        failed_ranks: set[int] = set()
        for rank, is_healthy in worker_health_results.items():
            if is_healthy:
                self.failure_counts.pop(rank, None)
                continue

            failure_count = self._record_failure(rank)
            if failure_count >= self.threshold:
                logger.warning(
                    f"Worker {rank} reached health check failure threshold: {failure_count}/{self.threshold}."
                )
                failed_ranks.add(rank)
            else:
                logger.warning(
                    f"Worker {rank} health check failed but remains active: {failure_count}/{self.threshold}."
                )

        return failed_ranks

    def mark_failed_ranks(self, worker_health_results: dict[int, bool]) -> set[int]:
        failed_ranks: set[int] = set()
        for rank, is_healthy in worker_health_results.items():
            if is_healthy:
                self.failure_counts.pop(rank, None)
                continue

            failure_count = self._record_failure(rank)
            logger.warning(
                f"Worker {rank} failed explicit health check and will be marked inactive "
                f"immediately: failure_count={failure_count}."
            )
            failed_ranks.add(rank)

        return failed_ranks


@dataclass(frozen=True)
class _ReadyRecoveryHF:
    model_path: str
    tokenizer_path: str | None = None


class RolloutHealthManager:
    """Own worker health state and recovery after controller startup.

    RolloutController creates workers, launches them the first time, and routes requests. RolloutHealthManager only
    reads registry snapshots, updates lifecycle_state, runs health checks, and restarts failed lifecycle groups. Worker
    actors still own backend-specific server start/stop/probe/generate.
    """

    def __init__(
        self,
        config: RolloutConfig,
        registry: RolloutWorkerRegistry,
        worker_lifecycle_listeners: Iterable[RolloutWorkerLifecycleListener] | None = None,
    ):
        self._registry = registry
        self._worker_lifecycle_listeners = tuple(worker_lifecycle_listeners or ())
        self._check_interval = config.health_check_interval_seconds
        self._check_timeout_seconds = config.health_check_timeout_seconds
        self._check_failure_threshold = config.health_check_failure_threshold
        self._periodic_health_checks_enabled = self._check_failure_threshold > 0
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._thread: threading.Thread | None = None
        self._lifecycle_operation_lock = threading.Lock()
        self._worker_health_failure_tracker = _WorkerHealthFailureTracker(threshold=self._check_failure_threshold)
        self._ready_recovery_hf: _ReadyRecoveryHF | None = None

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------
    def set_ready_recovery_hf(
        self,
        *,
        model_path: str,
        tokenizer_path: str | None = None,
    ) -> None:
        self._ready_recovery_hf = _ReadyRecoveryHF(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
        )
        logger.info(f"Ready rollout recovery HF updated: model_path={model_path}, tokenizer_path={tokenizer_path}.")

    def clear_ready_recovery_hf(self) -> None:
        self._ready_recovery_hf = None
        logger.info("Ready rollout recovery HF cleared.")

    def start(self) -> None:
        health_thread_alive = self._thread is not None and self._thread.is_alive()
        if health_thread_alive:
            return

        self._stop_event.clear()
        self._pause_event.set()
        if not self._periodic_health_checks_enabled:
            logger.info("Rollout worker periodic health check is disabled.")
            return

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("RolloutHealthManager started.")

    def stop(self) -> None:
        self._stop_event.set()
        self._pause_event.clear()
        thread = self._thread
        if not thread:
            self._pause_event.set()
            return

        thread.join(timeout=HEALTH_MANAGER_STOP_JOIN_TIMEOUT)
        if thread.is_alive():
            logger.warning(
                f"RolloutHealthManager stop timed out after {HEALTH_MANAGER_STOP_JOIN_TIMEOUT}s; "
                "health thread is still exiting."
            )
            return

        self._thread = None
        self._pause_event.set()
        logger.info("RolloutHealthManager stopped.")

    def pause(self) -> None:
        self._pause_event.set()
        logger.info("RolloutHealthManager paused.")

    def resume(self) -> None:
        self._pause_event.clear()
        logger.info("RolloutHealthManager resumed.")

    # ------------------------------------------------------------------
    # Public health and lifecycle workflows
    # ------------------------------------------------------------------

    def run_once(self) -> None:
        if not self._periodic_health_checks_enabled:
            logger.debug("Skipping rollout worker periodic health check because it is disabled.")
            return

        if not self._lifecycle_operation_lock.acquire(blocking=False):
            logger.debug("Skipping rollout worker health check because another lifecycle operation is running.")
            return

        failed_groups: tuple[WorkerGroup, ...] = ()
        logger.debug("RolloutHealthManager running health checks for active workers.")
        try:
            worker_health_results = self._check_active_workers_health()
            failed_ranks = self._worker_health_failure_tracker.update_failed_ranks(worker_health_results)
            if not failed_ranks:
                return
            try:
                self._checkpoint_not_stopping()
            except _HealthManagerStopping:
                return
            failed_groups = self._registry.mark_unhealthy_ranks(failed_ranks)
        finally:
            self._lifecycle_operation_lock.release()

        for group in failed_groups:
            logger.warning(f"Rollout worker group ranks={group.ranks} failed health check. Marking as inactive.")
        self._notify_worker_lifecycle_listeners(
            failed_groups,
            event_name="inactive",
            notify_listener=lambda listener, group: listener.on_worker_group_inactive(group),
        )

    def restart_inactive_workers(self) -> None:
        """Synchronously restart inactive groups before the next sync-step
        weight update."""
        recovered_groups: list[WorkerGroup] = []
        groups_to_recover: tuple[WorkerGroup, ...] = ()

        try:
            with self._paused_lifecycle_operation():
                groups_to_recover = self._registry.claim_inactive_groups_for_recovery()
                if groups_to_recover:
                    recovered_groups = self._restart_claimed_recovery_groups(groups_to_recover)
        except _HealthManagerStopping:
            return

        if not groups_to_recover:
            logger.info("No failed rollout workers detected during recovery.")
            return

        self._notify_worker_lifecycle_listeners(
            recovered_groups,
            event_name="recovered",
            notify_listener=lambda listener, group: listener.on_worker_group_recovered(group),
        )
        inactive_workers = [f"rank={worker.rank}, url={worker.url}" for worker in self._registry.inactive_workers()]
        if inactive_workers:
            logger.error("inactive rollout workers before sync-step weight update: " + ", ".join(inactive_workers))

    def check_and_shutdown_inactive_workers(self) -> None:
        """Fail-fast health-check active workers, mark failures inactive, and
        shut down every non-active group so shared resources can be reused by
        training."""
        groups_to_shutdown: tuple[WorkerGroup, ...] = ()

        try:
            with self._paused_lifecycle_operation():
                worker_health_results = self._check_active_workers_health()
                self._checkpoint_not_stopping()
                self._mark_unhealthy_worker_groups_inactive(worker_health_results)
                self._checkpoint_not_stopping()
                groups_to_shutdown = self._registry.inactive_worker_groups()
                for group in groups_to_shutdown:
                    self._shutdown_worker_group(group)
        except _HealthManagerStopping:
            return

        self._notify_worker_lifecycle_listeners(
            groups_to_shutdown,
            event_name="inactive",
            notify_listener=lambda listener, group: listener.on_worker_group_inactive(group),
        )
        if not groups_to_shutdown:
            logger.info("No failed rollout workers detected during shutdown barrier.")

    # ------------------------------------------------------------------
    # Background health loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        logger.info("RolloutHealthManager loop started.")

        while self._wait_until_next_check():
            try:
                self.run_once()
            except _HealthManagerStopping:
                break
            except RuntimeError:
                if self._stop_event.is_set():
                    break
                logger.exception("RolloutHealthManager run_once failed.")
            except Exception:
                logger.exception("RolloutHealthManager run_once failed.")

    def _wait_until_next_check(self) -> bool:
        while True:
            while self._pause_event.is_set() and not self._stop_event.is_set():
                self._stop_event.wait(timeout=0.5)

            if self._stop_event.is_set():
                return False

            if self._stop_event.wait(self._check_interval):
                return False

            if not self._pause_event.is_set() and not self._stop_event.is_set():
                return True

    # ------------------------------------------------------------------
    # Lifecycle operation gates
    # ------------------------------------------------------------------

    def _checkpoint_not_stopping(self) -> None:
        if self._stop_event.is_set():
            raise _HealthManagerStopping

    @contextmanager
    def _background_health_checks_paused(self):
        was_paused = self._pause_event.is_set()
        if not was_paused:
            self.pause()
        try:
            yield
        finally:
            if not was_paused:
                self.resume()

    @contextmanager
    def _paused_lifecycle_operation(self):
        with self._background_health_checks_paused():
            with self._lifecycle_operation_lock:
                self._checkpoint_not_stopping()
                yield

    # ------------------------------------------------------------------
    # Health checks and failure bookkeeping
    # ------------------------------------------------------------------

    def _check_active_workers_health(self) -> dict[int, bool]:
        workers_to_check = tuple(self._registry.active_workers())
        return self._check_workers_health(workers_to_check)

    def _check_workers_health(self, workers_to_check: Iterable[WorkerSnapshot]) -> dict[int, bool]:
        worker_health_results: dict[int, bool] = {}
        worker_health_checks = []
        for worker in list(workers_to_check):
            if worker.actor is None:
                logger.warning(f"Worker {worker.rank} has no actor reference.")
                worker_health_results[worker.rank] = False
                continue

            try:
                worker_health_checks.append(
                    (
                        worker,
                        asyncio.wait_for(
                            worker.actor.check_health.remote(),  # type: ignore[attr-defined]
                            timeout=self._check_timeout_seconds,
                        ),
                    )
                )
            except Exception as e:
                logger.error(f"Exception during check_health for worker {worker.rank} at {worker.url}: {e}.")
                worker_health_results[worker.rank] = False
                continue

        async def probe_workers():
            return await asyncio.gather(
                *(health_check for _, health_check in worker_health_checks),
                return_exceptions=True,
            )

        check_results = asyncio.run(probe_workers())

        for (worker, _), result in zip(worker_health_checks, check_results):
            if isinstance(result, Exception):
                logger.error(f"Exception during check_health for worker {worker.rank} at {worker.url}: {result}.")
                worker_health_results[worker.rank] = False
                continue
            if not result:
                logger.warning(f"check_health failed for worker {worker.rank} at {worker.url}.")
            worker_health_results[worker.rank] = bool(result)

        return worker_health_results

    def _mark_unhealthy_worker_groups_inactive(self, worker_health_results: dict[int, bool]) -> None:
        failed_ranks = self._worker_health_failure_tracker.mark_failed_ranks(worker_health_results)
        if not failed_ranks:
            return

        inactive_groups = self._registry.mark_unhealthy_ranks(failed_ranks)
        for group in inactive_groups:
            logger.warning(f"Rollout worker group ranks={group.ranks} failed health check. Marking as inactive.")

    # ------------------------------------------------------------------
    # Worker group recovery state
    # ------------------------------------------------------------------

    def _restart_claimed_recovery_groups(self, groups: tuple[WorkerGroup, ...]) -> list[WorkerGroup]:
        groups_needing_cleanup = {group.ranks: group for group in groups}

        try:
            group_recovery_results = self._restart_worker_groups(groups)
            self._checkpoint_not_stopping()

            recovered_groups: list[WorkerGroup] = []
            for group in groups:
                recovered = group_recovery_results.get(group.ranks, False)
                recorded_group = self._registry.set_group_recovery_result(group, recovered=recovered)
                if recovered:
                    self._worker_health_failure_tracker.clear(group.ranks)
                    groups_needing_cleanup.pop(group.ranks, None)
                    recovered_groups.append(recorded_group)
                else:
                    groups_needing_cleanup.pop(group.ranks, None)
                    logger.error(
                        "Failed to restart rollout worker group; training can continue with remaining active "
                        "rollout workers and skip this inactive group during rollout-side operations: "
                        f"ranks={recorded_group.ranks}, workers=["
                        + ", ".join(f"rank={worker.rank}, url={worker.url}" for worker in recorded_group.workers)
                        + "]"
                    )
            return recovered_groups
        except BaseException:
            self._cleanup_unfinalized_recovery_groups(tuple(groups_needing_cleanup.values()))
            raise

    def _cleanup_unfinalized_recovery_groups(self, groups: tuple[WorkerGroup, ...]) -> None:
        for group in groups:
            try:
                self._shutdown_worker_group(group, wait_server_down=False)
            except BaseException:
                logger.exception(f"Failed to clean up claimed rollout worker group ranks={group.ranks}.")
            try:
                self._registry.set_group_recovery_result(group, recovered=False)
            except BaseException:
                logger.exception(f"Failed to finalize claimed rollout worker group ranks={group.ranks} as inactive.")

    # ------------------------------------------------------------------
    # Worker group actor operations
    # ------------------------------------------------------------------

    def _restart_worker_groups(
        self,
        groups_to_recover: tuple[WorkerGroup, ...],
    ) -> dict[tuple[int, ...], bool]:
        logger.info(
            f"Restarting rollout worker groups in parallel: "
            f"group_ranks={[group.ranks for group in groups_to_recover]}, "
            f"max_parallel_groups={ROLLOUT_RECOVERY_MAX_PARALLEL_GROUPS}."
        )
        group_recovery_results: dict[tuple[int, ...], bool] = {}
        max_workers = min(len(groups_to_recover), max(1, ROLLOUT_RECOVERY_MAX_PARALLEL_GROUPS))
        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="rollout-recovery",
        ) as pool:
            future_to_group = {
                pool.submit(
                    self._restart_worker_group,
                    group,
                ): group
                for group in groups_to_recover
            }
            for future in as_completed(future_to_group):
                group = future_to_group[future]
                try:
                    group_recovery_results[group.ranks] = future.result()
                except Exception:
                    logger.exception(f"Failed to restart rollout worker group ranks={group.ranks}.")
                    group_recovery_results[group.ranks] = False
        return group_recovery_results

    def _restart_worker_group(
        self,
        group: WorkerGroup,
    ) -> bool:
        """Shutdown, restart, and health-check one complete worker group."""
        if not group.workers or len(group.workers) != len(group.ranks):
            logger.error(f"Cannot restart incomplete rollout worker group: ranks={group.ranks}.")
            return False

        restart_cleanup_needed = False

        try:
            self._checkpoint_not_stopping()
            if not self._shutdown_worker_group(group):
                return False
            restart_cleanup_needed = True

            self._checkpoint_not_stopping()
            ready_recovery_hf = self._ready_recovery_hf
            if ready_recovery_hf is None:
                reinit_kwargs: dict[str, object] = {"skip_load_weights": True}
            else:
                reinit_kwargs = {
                    "model_path": ready_recovery_hf.model_path,
                    "tokenizer_path": ready_recovery_hf.tokenizer_path,
                    "skip_load_weights": False,
                }

            ray.get(
                [
                    worker.actor.reinit.remote(**reinit_kwargs)  # type: ignore[attr-defined]
                    for worker in group.workers
                ],
                timeout=ROLLOUT_RAY_GET_TIMEOUT,
            )

            self._checkpoint_not_stopping()
            health_results = self._check_workers_health(group.workers)
            unhealthy_ranks = [worker.rank for worker in group.workers if not health_results.get(worker.rank, False)]
            if unhealthy_ranks:
                logger.error(
                    f"Restarted rollout worker group ranks={group.ranks} has unhealthy ranks={unhealthy_ranks}."
                )
                self._shutdown_worker_group(group, wait_server_down=False)
                return False

            if ready_recovery_hf is None:
                self._checkpoint_not_stopping()
                # Weight-update recovery returns to the offloaded baseline
                # before the sync path wakes weights and KV cache back up.
                ray.get(
                    [worker.actor.offload.remote() for worker in group.workers],  # type: ignore[attr-defined]
                    timeout=ROLLOUT_RAY_GET_TIMEOUT,
                )

            logger.info(f"Successfully restarted rollout worker group ranks={group.ranks}.")
            return True
        except _HealthManagerStopping:
            if restart_cleanup_needed:
                self._shutdown_worker_group(group, wait_server_down=False)
            return False
        except Exception as e:
            logger.error(f"Failed to restart rollout worker group ranks={group.ranks}: {e}")
            if restart_cleanup_needed:
                self._shutdown_worker_group(group, wait_server_down=False)
            return False

    def _shutdown_worker_group(
        self,
        group: WorkerGroup,
        *,
        wait_server_down: bool = True,
    ) -> bool:
        """Shutdown every worker in one group and aggregate per-worker shutdown
        results."""
        shutdown_succeeded = True
        for worker in group.workers:
            try:
                ray.get(worker.actor.shutdown.remote(), timeout=60)  # type: ignore[attr-defined]
            except Exception as e:
                logger.warning(f"Shutdown failed for rollout worker rank={worker.rank}, url={worker.url}: {e}")
                shutdown_succeeded = False
                continue

            if not wait_server_down:
                continue
            if not self._wait_worker_server_down(worker, max_wait_attempts=SHUTDOWN_SERVER_DOWN_MAX_ATTEMPTS):
                logger.error(
                    f"Shutdown failed for rollout worker rank={worker.rank} because server did not stop: "
                    f"url={worker.url}"
                )
                shutdown_succeeded = False
        return shutdown_succeeded

    def _wait_worker_server_down(self, worker: WorkerSnapshot, *, max_wait_attempts: int) -> bool:
        retry_interval_seconds = 5.0
        for attempt in range(1, max_wait_attempts + 1):
            try:
                is_healthy = ray.get(worker.actor.check_health.remote(), timeout=self._check_timeout_seconds)  # type: ignore[attr-defined]
            except Exception:
                return True
            if not is_healthy:
                return True
            if attempt < max_wait_attempts:
                logger.warning(
                    f"Rollout worker rank={worker.rank} server still responds after shutdown "
                    f"attempt={attempt}/{max_wait_attempts}, url={worker.url}."
                )
                time.sleep(retry_interval_seconds)

        return False

    # ------------------------------------------------------------------
    # Worker lifecycle notifications
    # ------------------------------------------------------------------

    def _notify_worker_lifecycle_listeners(
        self,
        groups: Iterable[WorkerGroup],
        *,
        event_name: str,
        notify_listener: Callable[[RolloutWorkerLifecycleListener, WorkerGroup], None],
    ) -> None:
        for group in groups:
            for listener in self._worker_lifecycle_listeners:
                try:
                    notify_listener(listener, group)
                except Exception:
                    logger.exception(
                        f"Rollout worker {event_name} listener failed: "
                        f"listener={type(listener).__name__}, group_ranks={group.ranks}"
                    )
