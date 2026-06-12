import asyncio
import threading
from contextlib import nullcontext
from typing import TYPE_CHECKING, Callable, Optional

import ray

from xtuner.v1.utils import get_logger

from .utils import ROLLOUT_RAY_GET_TIMEOUT


if TYPE_CHECKING:
    from .controller import WorkerInfo
    from .worker import RolloutConfig, RolloutWorker


logger = get_logger()


class RolloutHealthManager:
    def __init__(
        self,
        config: "RolloutConfig",
        workers_info: dict[int, "WorkerInfo"],
        worker_infos_lock: Optional[threading.RLock] = None,
        on_worker_inactive: Callable[[int], None] | None = None,
        on_worker_recovered: Callable[[int], None] | None = None,
    ):
        self._workers_info = workers_info
        self._worker_infos_lock = worker_infos_lock
        self._on_worker_inactive = on_worker_inactive
        self._on_worker_recovered = on_worker_recovered
        self._check_interval = config.health_check_interval_seconds
        self._check_failure_threshold = config.health_check_failure_threshold
        self._stop_event: Optional[threading.Event] = None
        self._pause_event: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # 启动时暂停，开始 generation 后再 resume。
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("RolloutHealthManager started.")

    def stop(self) -> None:
        if not self._thread:
            return

        assert self._stop_event is not None
        self._stop_event.set()
        if self._pause_event:
            self._pause_event.clear()
        self._thread.join(timeout=5)
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

    def run_once(self) -> None:
        logger.debug("RolloutHealthManager running health checks for active workers.")
        with self._maybe_lock():
            workers_snapshot = {
                rank: (info.actor, info.url, info.is_active) for rank, info in self._workers_info.items()
            }

        workers_to_check = [
            (rank, actor, url, is_active) for rank, (actor, url, is_active) in workers_snapshot.items() if is_active
        ]
        if not workers_to_check:
            return

        tasks = [
            check_worker_health(actor, rank, url, is_active, self._check_failure_threshold)
            for rank, actor, url, is_active in workers_to_check
        ]

        async def _run_checks() -> list[bool]:
            return await asyncio.gather(*tasks)

        check_results = asyncio.run(_run_checks())
        inactive_workers = []
        for (rank, _, _, _), is_healthy in zip(workers_to_check, check_results):
            if not is_healthy:
                logger.warning(f"Worker {rank} failed health check. Marking as inactive.")
                with self._maybe_lock():
                    self._workers_info[rank].is_active = False
                    inactive_worker = self._workers_info[rank].actor
                self._notify_worker_inactive(rank)
                if inactive_worker is None:
                    logger.error(f"[RolloutHealthManager] Worker {rank} has no actor reference. Skipping shutdown.")
                    continue
                inactive_workers.append((rank, inactive_worker))
            else:
                logger.debug(f"[RolloutHealthManager] Worker {rank} passed health check.")

        for rank, inactive_worker in inactive_workers:
            try:
                ray.get(inactive_worker.offload.remote(), timeout=ROLLOUT_RAY_GET_TIMEOUT)  # type: ignore[attr-defined]
            except Exception as e:
                logger.error(f"Exception while offloading worker {rank}: {e}")

            try:
                ray.get(inactive_worker.shutdown.remote(), timeout=ROLLOUT_RAY_GET_TIMEOUT)  # type: ignore[attr-defined]
            except Exception as e:
                logger.error(f"Exception while shutting down worker {rank}: {e}")

    def recover_unhealthy_workers(self) -> None:
        health_manager_was_paused = self.is_paused()
        if not health_manager_was_paused:
            self.pause()
        try:
            self._final_check_all_workers()
            self._recover_inactive_workers()
            with self._maybe_lock():
                inactive_workers = [
                    f"rank={rank}, url={info.url}" for rank, info in self._workers_info.items() if not info.is_active
                ]
            if inactive_workers:
                raise RuntimeError(
                    "inactive rollout workers after recovery: "
                    + ", ".join(inactive_workers)
                    + ". Rollout worker recovery did not restore all workers."
                )
        finally:
            if not health_manager_was_paused:
                self.resume()

    def _run_loop(self) -> None:
        assert self._stop_event is not None and self._pause_event is not None
        logger.info("RolloutHealthManager loop started.")

        while not self._stop_event.is_set():
            while self._pause_event.is_set() and not self._stop_event.is_set():
                self._stop_event.wait(timeout=0.5)

            if self._stop_event.is_set():
                break

            if not self._pause_event.is_set() and not self._stop_event.is_set():
                self.run_once()

            if self._stop_event.wait(self._check_interval):
                break

    def _final_check_all_workers(self) -> None:
        with self._maybe_lock():
            workers_snapshot = {
                rank: (info.actor, info.url, info.is_active) for rank, info in self._workers_info.items()
            }
        for rank, (actor, url, was_active) in workers_snapshot.items():
            try:
                is_healthy = ray.get(actor.check_health.remote(), timeout=ROLLOUT_RAY_GET_TIMEOUT)  # type: ignore[attr-defined]
            except Exception as e:
                is_healthy = False
                logger.warning(f"Final health check raised for rollout worker {rank} at {url}: {e}.")

            if not is_healthy:
                logger.warning(f"Final health check failed for rollout worker {rank} at {url}.")

            with self._maybe_lock():
                self._workers_info[rank].is_active = bool(is_healthy)

            if is_healthy and not was_active:
                logger.info(f"Mark rollout worker {rank} active after final health check: url={url}")
                self._notify_worker_recovered(rank)
            elif not is_healthy and was_active:
                logger.warning(
                    f"Mark rollout worker {rank} inactive because final health check failed before training: url={url}"
                )
                self._notify_worker_inactive(rank)

    def _recover_inactive_workers(self) -> None:
        with self._maybe_lock():
            failed_workers = [
                (rank, info.actor, info.url) for rank, info in self._workers_info.items() if not info.is_active
            ]

        if not failed_workers:
            logger.info("No failed workers detected during recovery.")
            return

        logger.warning(f"Detected {len(failed_workers)} failed workers. Initiating recovery process.")
        for rank, actor, expected_url in failed_workers:
            if self._restart_failed_worker(rank, actor, expected_url):
                with self._maybe_lock():
                    self._workers_info[rank].is_active = True
                self._notify_worker_recovered(rank)

    def _restart_failed_worker(self, rank: int, worker: "RolloutWorker", expected_url: str) -> bool:
        try:
            ray.get(worker.shutdown.remote(), timeout=ROLLOUT_RAY_GET_TIMEOUT)  # type: ignore[attr-defined]
            _, url = ray.get(worker.init.remote(), timeout=ROLLOUT_RAY_GET_TIMEOUT)  # type: ignore[attr-defined]
            if url != expected_url:
                raise AssertionError(f"Worker restarted with unexpected URL: expected {expected_url}, got {url}.")
            _, session_url = ray.get(worker.get_session_server_info.remote(), timeout=ROLLOUT_RAY_GET_TIMEOUT)  # type: ignore[attr-defined]
            is_healthy = ray.get(worker.check_health.remote(), timeout=ROLLOUT_RAY_GET_TIMEOUT)  # type: ignore[attr-defined]

            if is_healthy:
                with self._maybe_lock():
                    self._workers_info[rank].url = url
                    self._workers_info[rank].session_url = session_url
                logger.info(f"Successfully restarted rollout worker rank={rank} with URL {url}.")
                return True
            logger.error(f"Rollout worker rank={rank} is still unhealthy after restart.")
            return False
        except AssertionError:
            raise
        except Exception as e:
            logger.error(f"Failed to restart rollout worker rank={rank}: {e}")
            return False

    def _maybe_lock(self):
        if self._worker_infos_lock is None:
            return nullcontext()
        return self._worker_infos_lock

    def _notify_worker_inactive(self, rank: int) -> None:
        if self._on_worker_inactive is None:
            return
        try:
            self._on_worker_inactive(rank)
        except Exception:
            logger.exception(f"Rollout worker inactive callback failed: rank={rank}")

    def _notify_worker_recovered(self, rank: int) -> None:
        if self._on_worker_recovered is None:
            return
        try:
            self._on_worker_recovered(rank)
        except Exception:
            logger.exception(f"Rollout worker recovered callback failed: rank={rank}")


async def check_worker_health(
    worker: "RolloutWorker", rank: int, url: str, is_active: bool, failure_threshold: int = 3
) -> bool:
    if worker is None or not is_active:
        logger.warning("Worker has no actor reference or is marked inactive.")
        return False
    failing_count = 0
    while failing_count < failure_threshold:
        try:
            health_status = await worker.check_health.remote()  # type: ignore[attr-defined]
            if health_status:
                return True
            failing_count += 1
            logger.warning(f"Health check failed for worker {rank} at {url}. Failure count: {failing_count}")
        except Exception as e:
            failing_count += 1
            logger.error(
                f"Exception during health check for worker {rank} at {url}: {e}. Failure count: {failing_count}"
            )
    return False
