from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from xtuner.v1.utils import get_logger

from .health_manager import RolloutHealthManagerProxy, RolloutWorkerRouteInfo


@dataclass
class WorkerExternRouterConfig:
    """Control-plane config for a known external HTTP router.

    This class does not proxy user requests. It checks rollout worker URLs and
    keeps an external high-throughput router's worker table updated.
    """

    base_url: str
    worker_health_path: str = "/health"
    register_path: str = "/admin/rollout_workers"
    remove_path: str = "/admin/rollout_workers/remove"
    register_method: Literal["POST", "PUT"] = "POST"
    remove_method: Literal["POST", "DELETE"] = "POST"
    request_timeout: float = 10.0
    poll_interval_seconds: float = 1.0
    api_key: str | None = None
    worker_api_key: str | None = None
    headers: dict[str, str] = field(default_factory=dict)


class WorkerExternRouter:
    def __init__(
        self,
        health_manager: RolloutHealthManagerProxy,
        config: WorkerExternRouterConfig,
        *,
        log_dir: str | None = None,
    ) -> None:
        self.health_manager = health_manager
        self.config = config
        self.logger = get_logger(log_dir=log_dir, tag="WorkerExternRouter")
        self._registered_urls: dict[int, str] = {}
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="worker-extern-router")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._thread = None

    async def sync_once(self) -> None:
        route_infos = await self.health_manager.get_worker_route_infos.remote()
        current_ranks = {info.rank for info in route_infos}
        async with self._client() as client:
            for rank, url in list(self._registered_urls.items()):
                if rank not in current_ranks:
                    await self._remove_worker(client, rank, url)

            for info in route_infos:
                is_healthy = info.is_active and await self._check_worker_url(client, info.url)
                registered_url = self._registered_urls.get(info.rank)
                if is_healthy:
                    if registered_url != info.url:
                        if registered_url is not None:
                            await self._remove_worker(client, info.rank, registered_url)
                        await self._register_worker(client, info)
                elif registered_url is not None:
                    await self._remove_worker(client, info.rank, registered_url)

    async def run_forever(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self.sync_once()
            except Exception as exc:
                self.logger.error(f"Worker external router sync failed: {exc}")
            await asyncio.sleep(self.config.poll_interval_seconds)

    async def _check_worker_url(self, client: httpx.AsyncClient, url: str) -> bool:
        try:
            headers = {}
            if self.config.worker_api_key is not None:
                headers["authorization"] = f"Bearer {self.config.worker_api_key}"
            response = await client.get(self._worker_health_url(url), headers=headers)
            return response.status_code == 200
        except Exception as exc:
            self.logger.warning(f"Worker URL health check failed for {url}: {exc}")
            return False

    async def _register_worker(self, client: httpx.AsyncClient, info: RolloutWorkerRouteInfo) -> None:
        payload = self._worker_payload(info)
        response = await client.request(self.config.register_method, self._url(self.config.register_path), json=payload)
        response.raise_for_status()
        self._registered_urls[info.rank] = info.url
        self.logger.info(f"Registered rollout worker {info.rank} to external router: {info.url}")

    async def _remove_worker(self, client: httpx.AsyncClient, rank: int, url: str) -> None:
        payload = {"rank": rank, "url": url}
        response = await client.request(self.config.remove_method, self._url(self.config.remove_path), json=payload)
        response.raise_for_status()
        self._registered_urls.pop(rank, None)
        self.logger.warning(f"Removed rollout worker {rank} from external router: {url}")

    def _worker_payload(self, info: RolloutWorkerRouteInfo) -> dict[str, Any]:
        return {
            "rank": info.rank,
            "url": info.url,
            "is_active": info.is_active,
        }

    def _client(self) -> httpx.AsyncClient:
        headers = dict(self.config.headers)
        if self.config.api_key is not None:
            headers.setdefault("authorization", f"Bearer {self.config.api_key}")
        return httpx.AsyncClient(timeout=self.config.request_timeout, headers=headers)

    def _url(self, path: str) -> str:
        return f"{self.config.base_url.rstrip('/')}/{path.lstrip('/')}"

    def _run(self) -> None:
        try:
            asyncio.run(self.run_forever())
        except Exception as exc:
            self.logger.error(f"Worker external router stopped unexpectedly: {exc}")

    def _worker_health_url(self, url: str) -> str:
        return f"{url.rstrip('/')}/{self.config.worker_health_path.lstrip('/')}"
