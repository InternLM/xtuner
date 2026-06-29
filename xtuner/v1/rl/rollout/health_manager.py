from __future__ import annotations

import asyncio
import os
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import TYPE_CHECKING, Protocol

import ray

from xtuner.v1.utils import get_logger

from .worker_registry import RolloutWorkerRegistry, WorkerGroup, WorkerSnapshot


if TYPE_CHECKING:
    from .worker import RolloutConfig

ROLLOUT_RAY_GET_TIMEOUT = int(os.getenv("XTUNER_ROLLOUT_RAY_GET_TIMEOUT", str(5 * 3600)))  # default 5 hours
ROLLOUT_RECOVERY_MAX_PARALLEL_GROUPS = 4
HEALTH_MANAGER_STOP_JOIN_TIMEOUT = 30.0
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
        self._stop_event: threading.Event | None = None
        self._pause_event: threading.Event | None = None
        self._thread: threading.Thread | None = None
        self._operation_lock = threading.Lock()
        self._worker_health_failure_counts: dict[int, int] = {}
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
        thread = self._thread
        if not thread:
            return

        assert self._stop_event is not None
        self._stopped = True
        self._stop_event.set()
        if self._pause_event:
            self._pause_event.clear()

        thread.join(timeout=HEALTH_MANAGER_STOP_JOIN_TIMEOUT)
        if thread.is_alive():
            logger.warning(
                f"RolloutHealthManager stop timed out after {HEALTH_MANAGER_STOP_JOIN_TIMEOUT}s; "
                "health thread is still exiting."
            )
            return

        self._thread = None
        self._stop_event = None
        self._pause_event = None
        logger.info("RolloutHealthManager stopped.")

    def pause(self) -> None:
        if self._pause_event is None:
            return
        self._pause_event.set()
        logger.info("RolloutHealthManager paused.")

    def resume(self) -> None:
        if self._pause_event is None:
            return
        self._pause_event.clear()
        logger.info("RolloutHealthManager resumed.")

    def _is_paused(self) -> bool:
        return self._pause_event is None or self._pause_event.is_set()

    def _is_stopping(self) -> bool:
        """Return whether the health manager is stopping or already stopped."""
        return self._stopped or (self._stop_event is not None and self._stop_event.is_set())

    @contextmanager
    def _background_health_checks_paused(self):
        was_paused = self._is_paused()
        if not was_paused:
            self.pause()
        try:
            yield
        finally:
            if not was_paused:
                self.resume()

    def restart_inactive_workers(self) -> None:
        """Synchronously restart inactive groups before the next sync-step
        weight update."""
        with self._background_health_checks_paused():
            with self._operation_lock:
                failed_groups = list(self._registry.claim_inactive_groups_for_recovery())
                if not failed_groups:
                    logger.info("No failed rollout workers detected during recovery.")
                    return

                sorted_failed_groups = sorted(failed_groups, key=lambda group: group.ranks)
                for group in sorted_failed_groups:
                    failed_ranks = sorted(worker.rank for worker in group.workers if not worker.is_active())
                    logger.warning(
                        f"Detected failed rollout worker ranks={failed_ranks}; restart_group_ranks={group.ranks}."
                    )

                if self._abort_restart_recovery_if_stopping(sorted_failed_groups):
                    return

                logger.info(
                    f"Restarting rollout worker groups in parallel: "
                    f"group_ranks={[group.ranks for group in sorted_failed_groups]}, "
                    f"max_parallel_groups={ROLLOUT_RECOVERY_MAX_PARALLEL_GROUPS}."
                )
                group_recovery_results: dict[tuple[int, ...], bool] = {}
                max_workers = min(len(sorted_failed_groups), max(1, ROLLOUT_RECOVERY_MAX_PARALLEL_GROUPS))
                with ThreadPoolExecutor(
                    max_workers=max_workers,
                    thread_name_prefix="rollout-recovery",
                ) as pool:
                    future_to_group = {
                        pool.submit(
                            self._restart_worker_group,
                            group,
                        ): group
                        for group in sorted_failed_groups
                    }
                    for future in as_completed(future_to_group):
                        group = future_to_group[future]
                        try:
                            group_recovery_results[group.ranks] = future.result()
                        except Exception:
                            logger.exception(f"Failed to restart rollout worker group ranks={group.ranks}.")
                            group_recovery_results[group.ranks] = False

                if self._abort_restart_recovery_if_stopping(
                    sorted_failed_groups,
                    group_recovery_results=group_recovery_results,
                ):
                    return

                failed_recovery_groups: list[WorkerGroup] = []
                recovered_groups: list[WorkerGroup] = []
                for group in sorted_failed_groups:
                    is_recovered = group_recovery_results.get(group.ranks, False)
                    updated_group = self._registry.set_group_recovery_result(group, recovered=is_recovered)
                    if is_recovered:
                        for rank in group.ranks:
                            self._worker_health_failure_counts.pop(rank, None)
                        recovered_groups.append(updated_group or group)
                    if not is_recovered:
                        failed_recovery_groups.append(group)
                for group in recovered_groups:
                    self._notify_worker_group_recovered(group)
            inactive_workers = [
                f"rank={worker.rank}, url={worker.url}" for worker in self._registry.inactive_workers()
            ]
            if inactive_workers:
                logger.error("inactive rollout workers before sync-step weight update: " + ", ".join(inactive_workers))
            if failed_recovery_groups:
                logger.error(
                    "Failed to restart rollout worker groups; training can continue with remaining active rollout "
                    "workers and skip inactive groups during rollout-side operations: "
                    + "; ".join(
                        f"ranks={group.ranks}, workers=["
                        + ", ".join(f"rank={worker.rank}, url={worker.url}" for worker in group.workers)
                        + "]"
                        for group in failed_recovery_groups
                    )
                )

    def check_and_shutdown_inactive_workers(self) -> None:
        """Fail-fast health-check active workers, mark failures inactive, and
        shut down every non-active group so shared resources can be reused by
        training."""
        with self._background_health_checks_paused():
            self._check_and_deactivate_failed_worker_groups(fail_fast=True)
            with self._operation_lock:
                inactive_groups = list(self._registry.claim_inactive_groups_for_recovery())

                if not inactive_groups:
                    logger.info("No failed rollout workers detected during shutdown barrier.")
                    return

                failed_shutdown_groups: list[WorkerGroup] = []
                for group in sorted(inactive_groups, key=lambda group: group.ranks):
                    is_shutdown = self._shutdown_worker_group(group, wait_server_down=True, best_effort=False)
                    self._registry.set_group_recovery_result(group, recovered=False)
                    if not is_shutdown:
                        failed_shutdown_groups.append(group)
                        logger.error(
                            "failed to shut down inactive rollout workers before training: "
                            + ", ".join(f"rank={worker.rank}, url={worker.url}" for worker in group.workers)
                        )
                if failed_shutdown_groups:
                    logger.error(
                        "Failed to shut down inactive rollout worker groups; training can continue with remaining "
                        "active rollout workers and failed groups stay inactive for rollout-side operations: "
                        + "; ".join(
                            f"ranks={group.ranks}, workers=["
                            + ", ".join(f"rank={worker.rank}, url={worker.url}" for worker in group.workers)
                            + "]"
                            for group in failed_shutdown_groups
                        )
                    )

    def run_once(self) -> None:
        logger.debug("RolloutHealthManager running health checks for all workers.")
        checked_active_count = self._check_and_deactivate_failed_worker_groups()
        if self._registry.active_workers() or self._is_stopping():
            return

        if checked_active_count == 0:
            logger.error("No active rollout workers before health check. All rollout workers are inactive.")
        else:
            logger.error("All rollout workers failed after health check. All rollout workers are inactive.")
        # TODO(duanyanhui): Propagate this fatal rollout-dead state to the
        # trainer and abort training immediately instead of only logging here.

    def _check_and_deactivate_failed_worker_groups(self, *, fail_fast: bool = False) -> int:
        """Health-check active workers and mark any failed lifecycle group
        inactive."""
        if self._check_failure_threshold <= 0 and not fail_fast:
            logger.debug("Rollout worker periodic health check is disabled.")
            return 0

        with self._operation_lock:
            workers_to_check = list(self._registry.active_workers())

        if not workers_to_check:
            return 0

        check_results = self._check_workers_health(workers_to_check, fail_fast=fail_fast)

        failed_ranks = {worker.rank for worker, is_healthy in zip(workers_to_check, check_results) if not is_healthy}
        failed_groups: tuple[WorkerGroup, ...] = ()

        if failed_ranks:
            with self._operation_lock:
                if not self._is_stopping():
                    failed_groups = self._registry.mark_unhealthy_ranks(failed_ranks)
                    for group in failed_groups:
                        logger.warning(
                            f"Rollout worker group ranks={group.ranks} failed health check. Marking as inactive."
                        )
                        self._notify_worker_group_inactive(group)

        return len(workers_to_check)

    def _notify_worker_group_inactive(self, group: WorkerGroup) -> None:
        for listener in self._worker_lifecycle_listeners:
            try:
                listener.on_worker_group_inactive(group)
            except Exception:
                logger.exception(
                    f"Rollout worker inactive listener failed: "
                    f"listener={type(listener).__name__}, group_ranks={group.ranks}"
                )

    def _notify_worker_group_recovered(self, group: WorkerGroup) -> None:
        for listener in self._worker_lifecycle_listeners:
            try:
                listener.on_worker_group_recovered(group)
            except Exception:
                logger.exception(
                    f"Rollout worker recovered listener failed: "
                    f"listener={type(listener).__name__}, group_ranks={group.ranks}"
                )

    def _check_workers_health(self, workers_to_check: list[WorkerSnapshot], *, fail_fast: bool = False) -> list[bool]:
        """Run periodic check_health probes concurrently."""
        if self._check_failure_threshold <= 0 and not fail_fast:
            return [True for _ in workers_to_check]

        async def check_one_worker(worker: WorkerSnapshot) -> bool:
            if worker.actor is None or not worker.is_active():
                logger.warning("Worker has no actor reference or is marked inactive.")
                return False
            try:
                is_healthy = await asyncio.wait_for(
                    worker.actor.check_health.remote(),  # type: ignore[attr-defined]
                    timeout=self._check_timeout_seconds,
                )
            except Exception as e:
                logger.error(f"Exception during check_health for worker {worker.rank} at {worker.url}: {e}.")
                return False
            if not is_healthy:
                logger.warning(f"check_health failed for worker {worker.rank} at {worker.url}.")
            return bool(is_healthy)

        async def check_workers(workers: list[WorkerSnapshot]) -> list[bool]:
            return await asyncio.gather(*(check_one_worker(worker) for worker in workers))

        check_results = asyncio.run(check_workers(workers_to_check))
        keep_active_by_rank: dict[int, bool] = {}
        with self._operation_lock:
            for worker, is_healthy in zip(workers_to_check, check_results):
                if is_healthy:
                    self._worker_health_failure_counts.pop(worker.rank, None)
                    keep_active_by_rank[worker.rank] = True
                else:
                    failure_count = self._worker_health_failure_counts.get(worker.rank, 0) + 1
                    self._worker_health_failure_counts[worker.rank] = failure_count
                    if fail_fast:
                        logger.warning(
                            f"Worker {worker.rank} failed explicit health check and will be marked inactive "
                            f"immediately: failure_count={failure_count}."
                        )
                        keep_active_by_rank[worker.rank] = False
                        continue
                    if failure_count >= self._check_failure_threshold:
                        logger.warning(
                            f"Worker {worker.rank} reached health check failure threshold: "
                            f"{failure_count}/{self._check_failure_threshold}."
                        )
                        keep_active_by_rank[worker.rank] = False
                    else:
                        logger.warning(
                            f"Worker {worker.rank} health check failed but remains active: "
                            f"{failure_count}/{self._check_failure_threshold}."
                        )
                        keep_active_by_rank[worker.rank] = True

        return [keep_active_by_rank[worker.rank] for worker in workers_to_check]

    def _run_loop(self) -> None:
        assert self._stop_event is not None and self._pause_event is not None
        logger.info("RolloutHealthManager loop started.")

        while not self._stop_event.is_set():
            while self._pause_event.is_set() and not self._stop_event.is_set():
                self._stop_event.wait(timeout=0.5)

            if self._stop_event.is_set():
                break

            if self._stop_event.wait(self._check_interval):
                break

            if self._pause_event.is_set() or self._stop_event.is_set():
                continue

            try:
                self.run_once()
            except RuntimeError:
                if self._is_stopping():
                    break
                logger.exception("RolloutHealthManager run_once failed.")
            except Exception:
                logger.exception("RolloutHealthManager run_once failed.")

    def _shutdown_worker_group(
        self,
        group: WorkerGroup,
        *,
        wait_server_down: bool,
        best_effort: bool,
    ) -> bool:
        """Shutdown every worker in one group and aggregate per-worker shutdown
        results."""
        max_wait_attempts = 60
        retry_interval_seconds = 5.0
        shutdown_succeeded = True
        for worker in group.workers:
            worker_shutdown_succeeded = True
            try:
                ray.get(worker.actor.shutdown.remote(), timeout=60)  # type: ignore[attr-defined]
            except Exception as e:
                worker_shutdown_succeeded = False
                log = logger.warning if best_effort else logger.error
                log(f"Shutdown failed for rollout worker rank={worker.rank}, url={worker.url}: {e}")

            if worker_shutdown_succeeded and wait_server_down:
                server_down = False
                for attempt in range(1, max_wait_attempts + 1):
                    try:
                        is_healthy = ray.get(worker.actor.check_health.remote(), timeout=self._check_timeout_seconds)  # type: ignore[attr-defined]
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
                    logger.error(
                        f"Rollout worker rank={worker.rank} server did not stop after shutdown: url={worker.url}."
                    )
                    worker_shutdown_succeeded = False

            if not worker_shutdown_succeeded:
                shutdown_succeeded = False
        return best_effort or shutdown_succeeded

    def _abort_restart_recovery_if_stopping(
        self,
        sorted_failed_groups: list[WorkerGroup],
        *,
        group_recovery_results: dict[tuple[int, ...], bool] | None = None,
    ) -> bool:
        if not self._is_stopping():
            return False

        for group in sorted_failed_groups:
            is_recovered = False
            if group_recovery_results is not None:
                is_recovered = group_recovery_results.get(group.ranks, False)
            if is_recovered:
                self._shutdown_worker_group(group, wait_server_down=False, best_effort=True)
            self._registry.set_group_recovery_result(group, recovered=False)
        return True

    def _restart_worker_group(
        self,
        group: WorkerGroup,
    ) -> bool:
        """Shutdown, restart with empty-init, and health-check one complete
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
            ray.get(
                [
                    worker.actor.set_skip_load_weights.remote(True)  # type: ignore[attr-defined]
                    for worker in group.workers
                ],
                timeout=ROLLOUT_RAY_GET_TIMEOUT,
            )
            init_results = ray.get(
                [
                    # init() reuses the server launch spec bound during
                    # controller startup.
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
                if init_result.rank != worker.rank or init_result.server_url != worker.url:
                    logger.error(
                        f"Rollout worker restart returned unexpected endpoint: rank={worker.rank}, "
                        f"init_rank={init_result.rank}, expected_url={worker.url}, "
                        f"init_url={init_result.server_url}."
                    )
                    self._shutdown_worker_group(group, wait_server_down=False, best_effort=True)
                    return False

            health_results = ray.get(
                [worker.actor.check_health.remote() for worker in group.workers],  # type: ignore[attr-defined]
                timeout=self._check_timeout_seconds,
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

            # Newly restarted workers should return to the same offloaded/sleep
            # baseline as the other colocated rollout workers before the sync
            # path wakes weights/KV back up.
            ray.get(
                [worker.actor.offload.remote() for worker in group.workers],  # type: ignore[attr-defined]
                timeout=ROLLOUT_RAY_GET_TIMEOUT,
            )

            logger.info(f"Successfully restarted rollout worker group ranks={group.ranks}.")
            return True
        except Exception as e:
            logger.error(f"Failed to restart rollout worker group ranks={group.ranks}: {e}")
            self._shutdown_worker_group(group, wait_server_down=False, best_effort=True)
            return False
        finally:
            try:
                ray.get(
                    [
                        worker.actor.restore_skip_load_weights.remote()  # type: ignore[attr-defined]
                        for worker in group.workers
                    ],
                    timeout=ROLLOUT_RAY_GET_TIMEOUT,
                )
            except Exception:
                logger.exception(
                    f"Failed to restore rollout worker skip_load_weights after restart: group_ranks={group.ranks}."
                )
