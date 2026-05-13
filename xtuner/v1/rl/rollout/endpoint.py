from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import ray
from pydantic import BaseModel, ConfigDict, Field

from .worker_extern_router import WorkerExternRouterConfig
from .worker_http_router import WorkerHttpRouterConfig
from .worker_local_router import WorkerLocalRouter, WorkerLocalRouterConfig


RolloutEndpointKind = Literal["worker_local", "worker_http", "worker_extern"]


@dataclass
class RolloutEndpoint:
    kind: RolloutEndpointKind
    worker_local_router: WorkerLocalRouter | None = None
    base_url: str | None = None
    rollout_controller: Any | None = None

    def require_worker_local_router(self) -> WorkerLocalRouter:
        if self.worker_local_router is None:
            raise RuntimeError(f"Rollout endpoint {self.kind!r} does not provide a WorkerLocalRouter.")
        return self.worker_local_router

    def require_base_url(self) -> str:
        if self.base_url is None:
            raise RuntimeError(f"Rollout endpoint {self.kind!r} does not provide a base URL.")
        return self.base_url


# 对外用户配置
class RolloutEndpointConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    kind: RolloutEndpointKind = "worker_local"
    worker_local_router_config: WorkerLocalRouterConfig = Field(default_factory=WorkerLocalRouterConfig)

    # URL consumed by AgentLoop for worker_extern. For worker_http this is filled
    # at runtime by RolloutController after WorkerHttpRouter starts.
    base_url: str | None = None

    worker_http_router_host: str = "0.0.0.0"
    worker_http_router_port: int = 8081
    worker_http_router_title: str = "XTuner Worker Router"
    worker_http_router_version: str = "0.1.0"
    worker_http_router_log_level: str = "warning"
    worker_http_router_request_timeout: float | None = None
    worker_http_router_stream_timeout: float | None = None

    worker_extern_router_health_path: str = "/health"
    worker_extern_router_register_path: str = "/admin/rollout_workers"
    worker_extern_router_remove_path: str = "/admin/rollout_workers/remove"
    worker_extern_router_register_method: str = "POST"
    worker_extern_router_remove_method: str = "POST"
    worker_extern_router_request_timeout: float = 10.0
    worker_extern_router_poll_interval_seconds: float = 1.0
    worker_extern_router_api_key: str | None = None
    worker_extern_router_worker_api_key: str | None = None
    worker_extern_router_headers: dict[str, str] = Field(default_factory=dict)

    def build(self, rollout_controller) -> RolloutEndpoint:
        if self.kind == "worker_local":
            return RolloutEndpoint(
                kind=self.kind,
                worker_local_router=self.worker_local_router_config.build(rollout_controller),
                rollout_controller=rollout_controller,
            )

        base_url = self.base_url
        if base_url is None and self.kind == "worker_http":
            rollout_metadata = ray.get(rollout_controller.get_rollout_metadata.remote())
            base_url = rollout_metadata.get("worker_http_router_url")
            if base_url is None:
                raise ValueError(
                    "Rollout endpoint 'worker_http' requires WorkerHttpRouter to be started before building "
                    "AgentLoop, or base_url to be provided as a runtime override."
                )
        if base_url is None:
            raise ValueError(f"Rollout endpoint {self.kind!r} requires base_url.")
        return RolloutEndpoint(kind=self.kind, base_url=base_url, rollout_controller=rollout_controller)

    def build_worker_http_router_config(self) -> WorkerHttpRouterConfig | None:
        if self.kind != "worker_http":
            return None
        return WorkerHttpRouterConfig(
            host=self.worker_http_router_host,
            port=self.worker_http_router_port,
            title=self.worker_http_router_title,
            version=self.worker_http_router_version,
            log_level=self.worker_http_router_log_level,
            request_timeout=self.worker_http_router_request_timeout,
            stream_timeout=self.worker_http_router_stream_timeout,
        )

    def build_worker_extern_router_config(self) -> WorkerExternRouterConfig | None:
        if self.kind != "worker_extern":
            return None
        if self.base_url is None:
            raise ValueError("Rollout endpoint 'worker_extern' requires base_url.")
        return WorkerExternRouterConfig(
            base_url=self.base_url,
            worker_health_path=self.worker_extern_router_health_path,
            register_path=self.worker_extern_router_register_path,
            remove_path=self.worker_extern_router_remove_path,
            register_method=self.worker_extern_router_register_method,
            remove_method=self.worker_extern_router_remove_method,
            request_timeout=self.worker_extern_router_request_timeout,
            poll_interval_seconds=self.worker_extern_router_poll_interval_seconds,
            api_key=self.worker_extern_router_api_key,
            worker_api_key=self.worker_extern_router_worker_api_key,
            headers=self.worker_extern_router_headers,
        )
