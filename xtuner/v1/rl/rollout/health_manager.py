from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, TypeAlias

import ray
from ray.actor import ActorProxy

from xtuner.v1.utils import get_logger

from .utils import ROLLOUT_RAY_GET_TIMEOUT, RolloutHealthChecker, SessionRouter
from .worker import RolloutConfig


@dataclass
class RolloutWorkerRouteInfo:
    rank: int
    actor: Any
    url: str
    is_active: bool = True


class RolloutHealthManager:
    def __init__(self, config: RolloutConfig, route_infos: list[RolloutWorkerRouteInfo]) -> None:
        self.config = config
        self.logger = get_logger(log_dir=config.worker_log_dir, tag="RolloutHealthManager")
        self.worker_info_lock = threading.RLock()
        self.rank2info: dict[int, RolloutWorkerRouteInfo] = {info.rank: info for info in route_infos}
        self.router = SessionRouter(self.rank2info, worker_infos_lock=self.worker_info_lock)
        self.health_checker = RolloutHealthChecker(
            config=self.config,
            workers_info=self.rank2info,
            worker_infos_lock=self.worker_info_lock,
        )
        self.health_checker.start()

    def get_worker_route_infos(self) -> list[RolloutWorkerRouteInfo]:
        with self.worker_info_lock:
            return [
                RolloutWorkerRouteInfo(
                    rank=rank,
                    actor=info.actor,
                    url=info.url,
                    is_active=info.is_active,
                )
                for rank, info in self.rank2info.items()
            ]

    def get_worker_server_urls_status(self) -> dict[str, bool]:
        with self.worker_info_lock:
            return {info.url: info.is_active for info in self.rank2info.values()}

    def get_ready_status(self) -> tuple[bool, dict[str, Any]]:
        with self.worker_info_lock:
            active_workers = sum(1 for info in self.rank2info.values() if info.is_active)
            total_workers = len(self.rank2info)
        return active_workers > 0, {
            "active_workers": active_workers,
            "total_workers": total_workers,
        }

    async def get_worker(self, session_id: int) -> Any | None:
        return await self.router.get_worker(session_id)

    async def get_worker_route_info(self, session_id: int) -> RolloutWorkerRouteInfo | None:
        actor = await self.router.get_worker(session_id)
        if actor is None:
            return None
        with self.worker_info_lock:
            for rank, info in self.rank2info.items():
                if info.actor == actor:
                    return RolloutWorkerRouteInfo(
                        rank=rank,
                        actor=info.actor,
                        url=info.url,
                        is_active=info.is_active,
                    )
        return None

    def get_active_actors(self) -> list[Any]:
        with self.worker_info_lock:
            return [info.actor for info in self.rank2info.values() if info.is_active]

    def pause(self) -> None:
        self.health_checker.pause()

    def resume(self) -> None:
        self.health_checker.resume()

    def stop(self) -> None:
        self.health_checker.stop()

    def report_worker_failure(self, rank: int, error_msg: str | None = None) -> None:
        worker = self._mark_worker_inactive(rank, "failure_report", error_msg)
        if worker is None:
            return
        self.logger.warning(f"Rollout worker {rank} marked inactive from failure report: {error_msg}")

    def recover_failed_workers(self) -> None:
        self.health_checker.pause()
        with self.worker_info_lock:
            failed_workers = [info for info in self.rank2info.values() if not info.is_active]
        if not failed_workers:
            self.logger.info("No failed workers detected during recovery.")
            self.health_checker.resume()
            return

        self.logger.warning(f"Detected {len(failed_workers)} failed workers. Initiating recovery process.")
        for info in failed_workers:
            self._shutdown_worker(info.rank, info.actor)
            url = self._restart_failed_worker(info.actor)
            if url is None:
                continue
            with self.worker_info_lock:
                current = self.rank2info.get(info.rank)
                if current is None:
                    continue
                current.url = url
                current.is_active = True
        self.health_checker.resume()

    def _mark_worker_inactive(self, rank: int, reason: str, error_msg: str | None = None) -> Any | None:
        with self.worker_info_lock:
            info = self.rank2info.get(rank)
            if info is None:
                self.logger.warning(f"Received inactive update for unknown rollout worker rank {rank}: {error_msg}")
                return None

            worker = info.actor
            if info.is_active:
                info.is_active = False
            return worker

    def _restart_failed_worker(self, worker: Any) -> str | None:
        try:
            dist_init_addr = ray.get(worker.init_dist_port.remote(), timeout=ROLLOUT_RAY_GET_TIMEOUT)
            _, url = ray.get(worker.init.remote(dist_init_addr), timeout=ROLLOUT_RAY_GET_TIMEOUT)
            is_healthy = ray.get(worker.check_health.remote(), timeout=ROLLOUT_RAY_GET_TIMEOUT)
            if is_healthy:
                self.logger.info(f"Successfully restarted worker {worker} with URL {url}.")
                return url
            self.logger.error(f"Worker {worker} is still unhealthy after restart.")
            return None
        except Exception as e:
            self.logger.error(f"Failed to restart worker: {e}")
            return None

    def _shutdown_worker(self, rank: int, worker: Any) -> None:
        try:
            ray.get(worker.offload.remote(), timeout=ROLLOUT_RAY_GET_TIMEOUT)
            ray.get(worker.shutdown.remote(), timeout=ROLLOUT_RAY_GET_TIMEOUT)
        except Exception as e:
            self.logger.error(f"Exception while shutting down worker {rank}: {e}")


RayRolloutHealthManager = ray.remote(RolloutHealthManager)
RolloutHealthManagerProxy: TypeAlias = ActorProxy[RayRolloutHealthManager]
