from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypeAlias, TypedDict

import ray
from ray.actor import ActorProxy

from xtuner.v1.utils import get_logger

from ._generation.session_worker_selector import RolloutWorkerHandle
from .rollout_worker_build import RolloutRuntime
from .utils import ROLLOUT_RAY_GET_TIMEOUT
from .worker import RolloutConfig


if TYPE_CHECKING:
    from xtuner.v1.rl.gateway.config import GatewayConfig
    from xtuner.v1.rl.rollout._generation.external_http_entry import ExternalRolloutHttpEntryConfig
    from xtuner.v1.rl.rollout._generation.internal_http_entry import InternalRolloutHttpEntryConfig


class RolloutWorkerMetadata(TypedDict):
    """Metadata for rollout workers and their configuration.

    This data structure encapsulates all necessary information about the rollout worker infrastructure, including
    engine topology, server addresses, and worker status. Used for communication between training processes and rollout
    workers.
    """

    # 推理引擎的拓扑结构，每个子列表代表一个推理引擎包含的所有 worker ranks
    # 例如：[[0, 1, 2, 3], [4, 5, 6, 7]] 表示有 2 个推理引擎，每个引擎包含 4 个 workers
    # 用于确定分布式推理的并行组划分
    engine_rank_mesh_array: List[List[int]]

    # worker rank 到服务器 URL 的映射字典，用于训练进程与 rollout workers 通信
    # 键：worker 的 rank ID（字符串形式的整数）
    # 值：对应的服务器地址列表（通常每个 rank 对应一个 URL）
    server_url_dict: Dict[int, str]

    # Rollout 配置对象，包含推理引擎的所有配置参数
    # 包括：并行策略（TP/EP）、超时设置、后端类型（LMDeploy/vLLM/SGLang）等
    rollout_config: RolloutConfig

    # Worker server URL map used by trainer-side control paths. First-version
    # rollout refactor assumes all workers remain available.
    worker_server_urls_status: Dict[str, bool]

    # Gateway HTTP server URL (e.g. "http://1.2.3.4:8080").
    # Set after start_gateway() is called; None if the gateway has not been started.
    api_server_url: Optional[str]

    # Internal rollout HTTP entry URL. Set after start_internal_http_entry() is called.
    internal_http_entry_url: Optional[str]

    # worker rank -> SessionServer proxy URL. server_url_dict keeps the
    # original worker URLs for trainer-side weight update / backend control paths.
    worker_session_url_dict: Dict[int, str]

    # SessionServer URL -> availability status. First-version rollout refactor
    # does not deactivate workers, so these values are always True.
    worker_session_urls_status: Dict[str, bool]

    # Runtime worker handles consumed by local/http/external generation paths.
    worker_handles: List[RolloutWorkerHandle]


# Keep this as a Ray actor because Ray AgentLoop actors need a shared, cross-process handle to the same controller
# state; passing a normal Python object would serialize a separate copy into each actor.
class RolloutController:
    """Controller for managing and coordinating multiple RolloutWorker
    actors."""

    def __init__(
        self,
        infer_config: RolloutConfig,
        runtime: RolloutRuntime,
    ):
        """Initialize the RolloutController.

        Args:
            infer_config (RolloutConfig): The configuration for the rollout.
            runtime: Pre-built rollout runtime, including worker handles and
                backend server URLs.
        """
        self.config = infer_config
        self.num_gpus_per_engine = self.config.num_gpus_per_engine
        self.logger = get_logger(log_dir=infer_config.worker_log_dir, tag="RolloutController")
        self.engine_rank_mesh_array = runtime.engine_rank_mesh_array
        self.worker_server_urls_map = runtime.worker_server_urls_map
        self.rank2worker = runtime.rank2worker
        self.worker_handles = runtime.worker_handles
        self.num_rollout_workers = len(self.rank2worker)
        # The timeout for the environment to wait for the rollout controller's response.
        # This should be longer than the controller's internal timeout (`rollout_timeout`)
        # to account for potential queuing delays and other overheads.
        self.timeout_multiplier = 2.0
        self._gateway_url: str | None = None
        self._internal_http_entry_url: str | None = None
        self._external_http_entries = []

    def start_gateway(self, config: "GatewayConfig") -> str | None:
        """Start the gateway HTTP server in a daemon thread and return its URL.

        The gateway exposes OpenAI-compatible endpoints that forward requests to
        this controller via :class:`~xtuner.v1.rl.gateway.backend.local_backend.LocalRolloutBackend`.
        Agent loops (e.g. CamelAgentLoop) discover the URL via :meth:`get_rollout_metadata`.

        Args:
            config: Gateway configuration.  ``port`` and ``host`` control where
                the server binds; ``capture_folder`` enables per-request trace files.

        Returns:
            The base URL of the gateway, e.g. ``"http://1.2.3.4:8080"``, or
            ``None`` when the configured rollout backend does not support the
            gateway.
        """
        if self.config.rollout_backend == "sglang":
            self.logger.error("XTuner gateway is not supported for SGLang rollout backend yet; skip starting gateway.")
            return None

        from xtuner.v1.rl.gateway import build_local_gateway_app, serve_gateway_in_thread

        config.capture_folder = str(Path(self.config.worker_log_dir) / config._CAPTURE_PATH_FOLDER)
        app = build_local_gateway_app(
            ray.get_runtime_context().current_actor, config=config, rollout_config=self.config
        )
        serve_gateway_in_thread(app, config)
        host = ray.util.get_node_ip_address() if config.host in ("", "0.0.0.0") else config.host
        url = f"http://{host}:{config.port}"
        self._gateway_url = url
        self.logger.info(f"Gateway server started at {url}, capture_folder: {config.capture_folder}")
        return url

    def start_internal_http_entry(self, config: "InternalRolloutHttpEntryConfig") -> str:
        from xtuner.v1.rl.rollout._generation.internal_http_entry import (
            build_internal_rollout_http_entry_app,
            serve_internal_rollout_http_entry_in_thread,
        )

        app = build_internal_rollout_http_entry_app(
            worker_handles=self.worker_handles,
            rollout_config=self.config,
            config=config,
        )
        serve_internal_rollout_http_entry_in_thread(app, config)
        host = ray.util.get_node_ip_address() if config.host in ("", "0.0.0.0") else config.host
        url = f"http://{host}:{config.port}"
        self._internal_http_entry_url = url
        self.logger.info(f"Internal rollout HTTP entry started at {url}")
        return url

    def start_external_http_entry(self, config: "ExternalRolloutHttpEntryConfig") -> None:
        from xtuner.v1.rl.rollout._generation.external_http_entry import ExternalRolloutHttpEntry

        entry = ExternalRolloutHttpEntry(
            worker_handles=self.worker_handles,
            rollout_config=self.config,
            config=config,
            log_dir=str(self.config.worker_log_dir),
        )
        entry.start()
        self._external_http_entries.append(entry)

    def get_rollout_metadata(self) -> RolloutWorkerMetadata:
        """Get information about the current rollout setup.

        Returns:
            dict: A dictionary containing the engine mesh list, server URL
                dictionary, and the rollout configuration.
        """
        worker_server_urls_map = {worker.rank: worker.backend_url for worker in self.worker_handles}
        worker_server_urls_status = {worker.backend_url: True for worker in self.worker_handles}
        worker_session_url_dict = {
            worker.rank: worker.session_server_url for worker in self.worker_handles if worker.session_server_url is not None
        }
        worker_session_urls_status = {
            worker.session_server_url: True for worker in self.worker_handles if worker.session_server_url is not None
        }
        rollout_metadata: RolloutWorkerMetadata = {
            "engine_rank_mesh_array": self.engine_rank_mesh_array,
            "server_url_dict": worker_server_urls_map,
            "rollout_config": self.config,
            "worker_server_urls_status": worker_server_urls_status,
            "api_server_url": self._internal_http_entry_url or self._gateway_url,
            "worker_session_url_dict": worker_session_url_dict,
            "worker_session_urls_status": worker_session_urls_status,
            "internal_http_entry_url": self._internal_http_entry_url,
            "worker_handles": list(self.worker_handles),
        }
        return rollout_metadata

    def get_worker_handles(self) -> list[RolloutWorkerHandle]:
        return list(self.worker_handles)

    def get_runtime_status(self) -> tuple[bool, dict[str, Any]]:
        return bool(self.worker_handles), {
            "rollout_workers": len(self.worker_handles),
            "total_workers": len(self.worker_handles),
            "workers": {
                worker.rank: {
                    "url": worker.backend_url,
                    "session_url": worker.session_server_url,
                    "has_rollout_worker_generator": True,
                }
                for worker in self.worker_handles
            },
        }

    def pause_generation(self):
        self._broadcast_to_rollout_worker_generators("pause_generation")
        self._broadcast_to_workers("pause_generation")

    def cleanup_after_pause(self):
        self._broadcast_to_workers("cleanup_after_pause")

    def continue_generation(self):
        self._broadcast_to_rollout_worker_generators("continue_generation")
        self._broadcast_to_workers("continue_generation")

    def offload(self):
        self._broadcast_to_workers("offload")

    def onload(self):
        self._broadcast_to_workers("onload_weights")
        self._broadcast_to_workers("onload_kvcache")

    def onload_weights(self):
        self._broadcast_to_workers("onload_weights")

    def onload_kvcache(self):
        self._broadcast_to_workers("onload_kvcache")

    def shutdown(self):
        """Shuts down all active rollout workers.

        Args:
            block (bool): Whether to block until the operation completes.
        """
        for entry in self._external_http_entries:
            entry.stop()
        self._external_http_entries.clear()
        self._broadcast_to_workers("shutdown")

    def _broadcast_to_workers(self, method_name: str):
        """Helper function to call a method on all rollout workers.

        Args:
            method_name (str): The name of the method to call.
            block (bool): Whether to block until the call completes.

        Returns:
            A list of futures if `block` is False, otherwise a list of results.
        """
        worker_actors = [worker.worker_actor for worker in self.worker_handles]
        futures = [getattr(actor, method_name).remote() for actor in worker_actors]
        results = ray.get(futures, timeout=ROLLOUT_RAY_GET_TIMEOUT)
        return results

    def _broadcast_to_rollout_worker_generators(self, method_name: str):
        generators = [worker.generator_actor for worker in self.worker_handles]
        futures = [getattr(actor, method_name).remote() for actor in generators]
        if not futures:
            return []
        return ray.get(futures, timeout=ROLLOUT_RAY_GET_TIMEOUT)

RayRolloutController = ray.remote(RolloutController)
RolloutControllerProxy: TypeAlias = ActorProxy[RayRolloutController]
