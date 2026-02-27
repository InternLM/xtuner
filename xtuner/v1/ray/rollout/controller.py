import asyncio
import os
import socket
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from itertools import cycle
from typing import Any, Dict, List, Optional, TypedDict
from uuid import uuid4

import ray
from ray.actor import ActorProxy
import uvicorn
from fastapi import FastAPI
from ray.util.placement_group import PlacementGroup

from transformers import AutoTokenizer
from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.ray.base import AutoAcceleratorWorkers
from xtuner.v1.ray.config.worker import RolloutConfig
from xtuner.v1.utils import get_logger

from .worker import RolloutWorker


ROLLOUT_RAY_GET_TIMEOUT = os.getenv("XTUNER_ROLLOUT_RAY_GET_TIMEOUT", 5 * 3600)  # default 5 hours


@dataclass
class WorkerInfo:
    """A data class to hold all state information for a single worker."""

    actor: RolloutWorker
    rank: int = -1
    is_active: bool = True
    failure_count: int = 0
    running_count: int = 0
    success_count: int = 0


class RolloutWorkerMetadata(TypedDict):
    engine_rank_mesh_array: List[List[int]]
    server_url_dict: Dict[str, List[str]]  # rank -> url
    rollout_config: RolloutConfig
    worker_server_urls_status: Dict[str, bool]


class SessionRouter:
    def __init__(
        self,
        worker_status: Dict[Any, bool],  # worker: worker_status
        max_sessions: int = 10000,
        max_idle_seconds: Optional[float] = 3600.0,
    ):
        assert len(worker_status) > 0
        self._workers = list(worker_status.items())
        self._max_sessions = max_sessions
        self._max_idle = max_idle_seconds

        # OrderedDict: key=session_id -> value=(worker, last_used_ts)
        self._map: OrderedDict[int, tuple[Any, float]] = OrderedDict()
        self._worker_cycler = cycle(self._workers)
        self._lock = asyncio.Lock()
        self.logger = get_logger()

    def _now(self) -> float:
        return time.time()

    def _evict_expired(self):
        if self._max_idle is None:
            return
        now = self._now()

        to_delete = []
        for sid, (_, last_used) in self._map.items():
            if now - last_used > self._max_idle:
                to_delete.append(sid)
            else:
                break
        for sid in to_delete:
            self._map.pop(sid, None)

    def _evict_lru_to_capacity(self):
        while len(self._map) > self._max_sessions:
            self._map.popitem(last=False)

    def update_active_workers(self, worker_status: Dict[Any, bool]):
        self._workers = list(worker_status.items())
        self.logger.debug(f"SessionRouter update active workers: {self._workers}")
        self._worker_cycler = cycle(self._workers)

    async def get_worker(self, session_id: int) -> Any:
        async with self._lock:
            self._evict_expired()

            if session_id in self._map:
                worker, _ = self._map.pop(session_id)
                self._map[session_id] = (worker, self._now())
                if worker[1]:  # worker is healthy
                    return worker[0]

            worker = next(self._worker_cycler)
            while worker[1] is False:
                worker = next(self._worker_cycler)
            self._map[session_id] = (worker, self._now())

            self._evict_lru_to_capacity()
            return worker[0]


class RolloutController:
    """Controller for managing and coordinating multiple RolloutWorker
    actors."""

    def __init__(
        self,
        infer_config: RolloutConfig,
        placement_group: PlacementGroup,
    ):
        """Initialize the RolloutController.

        Args:
            infer_config (RolloutConfig): The configuration for the rollout.
            placement_group (PlacementGroup): The placement group for the
                RolloutWorker actors.
        """
        self.config = infer_config
        self.num_gpus_per_engine = (
            self.config.expert_parallel_size
            if self.config.expert_parallel_size > 1
            else self.config.tensor_parallel_size
        )
        self.logger = get_logger(log_dir=infer_config.worker_log_dir, tag="RolloutController")
        self.num_workers = 0
        self.workers_info: Dict[str, WorkerInfo] = {}  # url -> WorkerInfo
        self.active_rollout_workers: List[RolloutWorker] = []
        self.tokenizer = AutoTokenizer.from_pretrained(infer_config.tokenizer_path, trust_remote_code=True)
        self.workers, self.rank_bundle_idx_list = AutoAcceleratorWorkers.from_placement_group(
            self._get_worker_cls(), infer_config, placement_group
        )
        self.engine_rank_mesh_array, self.worker_server_urls_map = self._init_workers()
        self._start_api_server()
        # todo(@duanyanhui): add router to replace native round robin
        self.router = SessionRouter(self._get_worker_status_for_router())
        # The timeout for the environment to wait for the rollout controller's response.
        # This should be longer than the controller's internal timeout (`rollout_timeout`)
        # to account for potential queuing delays and other overheads.
        self.timeout_multiplier = 2.0

    def check_health(self) -> None:
        """Check the health of all active rollout workers.

        Returns:
            List[bool]: A list of booleans indicating the health status of
                each active rollout worker.
        """
        active_workers = [(url, info) for url, info in self.workers_info.items() if info.is_active]
        if not active_workers:
            return

        urls, infos = zip(*active_workers)
        actors = [info.actor for info in infos]

        health_statuses = ray.get([actor.check_health.remote() for actor in actors], timeout=ROLLOUT_RAY_GET_TIMEOUT)

        for url, is_healthy in zip(urls, health_statuses):
            if not is_healthy:
                self.logger.warning(f"Rollout worker {url} is unhealthy.")
                self._deactivate_worker(url)

    def get_rollout_metadata(self) -> RolloutWorkerMetadata:
        """Get information about the current rollout setup.

        Returns:
            dict: A dictionary containing the engine mesh list, server URL
                dictionary, and the rollout configuration.
        """
        worker_server_urls_status = {url: info.is_active for url, info in self.workers_info.items()}
        rollout_metadata: RolloutWorkerMetadata = {
            "engine_rank_mesh_array": self.engine_rank_mesh_array,
            "server_url_dict": self.worker_server_urls_map,
            "rollout_config": self.config,
            "worker_server_urls_status": worker_server_urls_status,
        }
        return rollout_metadata

    async def generate(self, rollout_state: RolloutState) -> RolloutState:
        session_id = rollout_state.session_uid if rollout_state.session_uid else uuid4().int
        worker = await self.router.get_worker(session_id)
        # TODO(#@duanyanhui):
        # 1. 现在的worker_info的管理写的不好，由于打印和计数都不太合理，实际用处不大，仅在debug时才会打开，先去掉这部分逻辑，后面加上
        # 2. 目前先去掉根据失败的样本的数量来判断worker是否可用的逻辑，实际上这部分逻辑也用不到，需要更好的check_health的机制来判断worker是否可用
        active_worker_to_url_map = self._get_active_worker_to_url_map()
        server_url = active_worker_to_url_map.get(worker)
        self.workers_info[server_url].running_count += 1
        response_ref = worker.generate.remote(rollout_state=rollout_state)  # type: ignore[attr-defined]
        try:
            response_rollout_state = await asyncio.wait_for(
                response_ref, timeout=self.config.rollout_timeout * self.timeout_multiplier
            )
            return response_rollout_state
        except asyncio.TimeoutError:
            self.logger.error(f"Rollout timeout for worker {worker}. Skipping sample.")
            rollout_state.status = Status.FAILED
            rollout_state.error_msg = (
                f"Rollout request timed out after {self.config.rollout_timeout * self.timeout_multiplier} seconds."
            )
            return rollout_state

    def pause_generation(self):
        self._broadcast_to_active_workers("pause_generation")

    def continue_generation(self):
        self._broadcast_to_active_workers("continue_generation")

    def offload(self):
        self._broadcast_to_active_workers("offload")

    def onload(self):
        self._broadcast_to_active_workers("onload_weights")
        self._broadcast_to_active_workers("onload_kvcache")

    def onload_weights(self):
        self._broadcast_to_active_workers("onload_weights")

    def onload_kvcache(self):
        self._broadcast_to_active_workers("onload_kvcache")

    def shutdown(self):
        """Shuts down all active rollout workers.

        Args:
            block (bool): Whether to block until the operation completes.
        """
        self._broadcast_to_active_workers("shutdown")

    def _update_dist_init_addr(self, nodes_per_engine, server_urls_per_engine, dist_init_addrs, tp_size):
        """Update the distributed initialization addresses for workers.

        This is used to group workers that belong to the same inference engine.

        Args:
            nodes_per_engine (int): The number of nodes per inference engine.
            server_urls_per_engine (int): The number of server urls per inference engine.
            dist_init_addrs (list): The list of initial addresses.
            tp_size (int): The tensor parallel size.

        Returns:
            list: The updated list of distributed initialization addresses.
        """
        # lmdeploy pytorch ep: server_urls_per_engine > 1
        # sglang cross node engine: nodes_per_engine > 1
        assert server_urls_per_engine == 1 or nodes_per_engine == 1
        if nodes_per_engine > 1:
            index = list(range(0, self.num_workers + 1, tp_size)) + [self.num_workers]
            for i in range(1, len(index)):
                dist_init_addrs[index[i - 1] : index[i]] = [dist_init_addrs[index[i - 1]]] * (index[i] - index[i - 1])
        if server_urls_per_engine > 1:
            activate_servers = len(dist_init_addrs)
            for i in range(0, activate_servers, server_urls_per_engine):
                dist_init_addrs[i : i + server_urls_per_engine] = [dist_init_addrs[i]] * server_urls_per_engine
        return dist_init_addrs

    def _get_active_servers_count(self, infer_config: RolloutConfig, gpu_nums: int):
        """Calculate the number of active servers and nodes per engine.

        This calculation depends on the inference backend and parallelism settings.

        Args:
            infer_config (RolloutConfig): The rollout configuration.
            gpu_nums (int): The total number of GPUs available.

        Returns:
            Tuple[int, int]: A tuple containing the number of active servers
                and the number of nodes per engine.
        """
        # NOTE：Since different inference engines have different launch methods,
        # the number of nodes contained in each engine is not consistent.
        # For example: sglang requires starting an inference engine for each node,
        # while lmdeploy and vllm does not. Therefore, we calculate the number
        # of active servers based on the configuration.
        support_cross_node_comm = infer_config.rollout_cross_node_comm
        gpus_per_node = infer_config.gpus_per_node
        nodes_per_engine = (
            1
            if support_cross_node_comm or self.num_gpus_per_engine < gpus_per_node
            else self.num_gpus_per_engine // gpus_per_node
        )

        active_servers_count = int(
            (gpu_nums // self.num_gpus_per_engine) * nodes_per_engine * infer_config.server_urls_per_engine
        )
        return active_servers_count, nodes_per_engine

    def _broadcast_to_active_workers(self, method_name: str):
        """Helper function to call a method on all active workers.

        Args:
            method_name (str): The name of the method to call.
            block (bool): Whether to block until the call completes.

        Returns:
            A list of futures if `block` is False, otherwise a list of results.
        """
        futures = []
        for info in self.workers_info.values():
            if info.is_active:
                futures.append(getattr(info.actor, method_name).remote())
            else:
                self.logger.warning(f"Skipping {method_name} for inactive worker {info.actor}.")

        results = ray.get(futures, timeout=ROLLOUT_RAY_GET_TIMEOUT)
        return results

    def _deactivate_worker(self, url: str):
        """A helper function to deactivate a worker, update all related states,
        and shut it down."""
        worker_info = self.workers_info.get(url)
        if not worker_info or not worker_info.is_active:
            return

        self.logger.warning(f"Deactivating rollout worker {worker_info.actor} with URL {url} due to failures.")
        worker_info.is_active = False
        self.router.update_active_workers(self._get_worker_status_for_router())

        ray.get(worker_info.actor.offload.remote(), timeout=ROLLOUT_RAY_GET_TIMEOUT)  # type: ignore[attr-defined]
        ray.get(worker_info.actor.shutdown.remote(), timeout=ROLLOUT_RAY_GET_TIMEOUT)  # type: ignore[attr-defined]

    def _get_worker_status_for_router(self) -> Dict[RolloutWorker, bool]:
        """Helper to generate the status dict required by the SessionRouter."""
        return {info.actor: info.is_active for info in self.workers_info.values()}

    def _get_worker_cls(self):
        if os.environ.get("XTUNER_USE_LMDEPLOY") == "1":
            from .lmdeploy import LMDeployWorker

            return ray.remote(LMDeployWorker)
        elif os.environ.get("XTUNER_USE_VLLM") == "1":
            from .vllm import vLLMWorker

            return ray.remote(vLLMWorker)
        elif os.environ.get("XTUNER_USE_SGLANG") == "1":
            from .sglang import SGLangWorker

            return ray.remote(SGLangWorker)
        else:
            raise NotImplementedError(
                "Rollout backend is not supported."
                "Please set XTUNER_USE_LMDEPLOY or XTUNER_USE_VLLM"
                " or XTUNER_USE_SGLANG environment variable."
            )

    def _get_active_worker_to_url_map(self):
        """Get a mapping of active workers to their server URLs."""
        return {info.actor: url for url, info in self.workers_info.items()}

    def _is_port_in_use(self, host: str, port: int) -> bool:
        """Check if a port is in use on the given host."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return False
            except OSError:
                return True

    def _update_active_workers_and_urls_map(self, active_rollout_workers, worker_server_urls_map):
        """Update the list of active rollout workers and their server URLs.

        When the inference engine is launched across nodes (rollout_cross_node_comm=True), only the worker with
        tp_rank=0 in each engine is responsible for receiving input data. Other tp_ranks do not accept input.
        Therefore, this function updates active_rollout_workers and worker_server_urls_map to keep only the tp_rank=0
        workers and their corresponding URLs.
        """
        if self.config.rollout_cross_node_comm or self.num_gpus_per_engine < self.config.gpus_per_node:
            return active_rollout_workers, worker_server_urls_map
        else:
            active_worker_interval = self.num_gpus_per_engine // self.config.gpus_per_node
            active_rank = list(worker_server_urls_map.keys())[::active_worker_interval]
            active_worker_server_urls = list(worker_server_urls_map.values())[::active_worker_interval]
            return active_rollout_workers[::active_worker_interval], dict(zip(active_rank, active_worker_server_urls))

    def _start_api_server(self, host: str = "0.0.0.0", port: int = 8000):
        """Starts the API server to expose the rollout functionality."""
        app = FastAPI()
        port = self.config.api_port if self.config.api_port else port

        original_port = port
        while self._is_port_in_use(host, port):
            self.logger.warning(f"Port {port} is in use, trying port {port + 1}")
            port += 1

        if original_port != port:
            self.logger.info(f"API server will use port {port} instead of the originally configured {original_port}.")

        # TODO(@duanyanhui): add more API for other functionalities like pause, restart, shutdown, get_metadata, etc.
        @app.post("/generate")
        async def generate(request: RolloutState) -> RolloutState:
            try:
                response = await self.generate(rollout_state=request)
                return response
            except Exception as e:
                self.logger.error(f"Generate failed in API server: {e}")
                request.status = Status.FAILED
                request.error_msg = f"Generate failed in API server with error: {str(e)}"
                return request

        config = uvicorn.Config(app, host=host, port=port)
        server = uvicorn.Server(config)
        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()

    def _init_workers(self):
        """Initializes and configures the pool of RolloutWorker actors.

        This method configures distributed inference engines by grouping
        workers, where each group forms a tensor-parallel inference engine. It
        determines the `active_workers` to act as the head of each engine,
        constructs the `engine_rank_mesh_array` to define engine topology, acquires
        necessary distributed communication ports, and finally launches servers
        on the `active_workers` to get their addresses.

        Returns:
            Tuple[List, Dict]: A tuple where the first element is
            `engine_rank_mesh_array`, a list of lists containing the ranks of workers
            in each engine, and the second element is `worker_server_urls_map`,
            a dictionary mapping the ID of each active worker to its
            corresponding server URL.
        """
        active_servers_count, nodes_per_engine = self._get_active_servers_count(self.config, len(self.workers))
        interval = len(self.workers) // active_servers_count
        active_rollout_workers = self.workers[::interval]
        self.num_workers = len(active_rollout_workers)
        server_urls_per_engine = self.config.server_urls_per_engine

        set_bundle_idxs_objectref = []
        engine_rank_mesh_array = []
        activate_worker_idx = 0
        for active_worker in active_rollout_workers:
            head_rank, _ = self.rank_bundle_idx_list[activate_worker_idx]
            engine_workers_meta = self.rank_bundle_idx_list[head_rank : head_rank + interval]
            engine_bundle_idxs = [meta[1] for meta in engine_workers_meta]  # meta: (rank, bundle_idx)
            set_bundle_idxs_objectref.append(active_worker._set_engine_bundle_idxs.remote(engine_bundle_idxs))  # type: ignore[attr-defined]
            engine_rank_mesh_array.append([meta[0] for meta in engine_workers_meta])
            activate_worker_idx += interval
        ray.get(set_bundle_idxs_objectref)
        # set engine mesh list for each worker
        ray.get(
            [worker._set_engine_rank_mesh_array.remote(engine_rank_mesh_array) for worker in active_rollout_workers]
        )  # type: ignore[attr-defined]
        # init dist_init_addr for each worker according to parallel settings
        init_dist_init_addrs = ray.get([worker.init_dist_port.remote() for worker in active_rollout_workers])  # type: ignore[attr-defined]
        dist_init_addrs = self._update_dist_init_addr(
            nodes_per_engine, server_urls_per_engine, init_dist_init_addrs, self.num_gpus_per_engine
        )
        # launch rollout servers
        worker_server_urls_map = dict(  # rank -> url
            ray.get([worker.init.remote(dist_init_addrs[i]) for i, worker in enumerate(active_rollout_workers)])
        )
        active_rollout_workers, worker_server_urls_map = self._update_active_workers_and_urls_map(
            active_rollout_workers, worker_server_urls_map
        )
        self.workers_info = {}
        for i in range(len(active_rollout_workers)):
            rank = list(worker_server_urls_map.keys())[i]
            url = worker_server_urls_map[rank]
            self.workers_info[url] = WorkerInfo(rank=rank, actor=active_rollout_workers[i])
        self.logger.info(f"Rollout worker server URLs: {list(self.workers_info.keys())}")
        return engine_rank_mesh_array, worker_server_urls_map

RayRolloutController = ray.remote(RolloutController)
RolloutControllerProxy = ActorProxy[RayRolloutController]