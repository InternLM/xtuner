import asyncio
import math
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypeAlias, TypedDict
from uuid import uuid4

import ray
from ray.actor import ActorProxy
from ray.util.placement_group import PlacementGroup

from transformers import AutoTokenizer
from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.rl.trace import trace_function
from xtuner.v1.rl.utils import AutoAcceleratorWorkers
from xtuner.v1.utils import XTUNER_DETERMINISTIC, get_logger

from .parser.factory import build_reasoning_parser, build_tool_call_parser
from .parser.reasoning_parser import ReasoningParser
from .parser.tool_parser import ToolCallParser
from .utils import ROLLOUT_RAY_GET_TIMEOUT, RolloutHealthChecker, SessionRouter
from .worker import (
    ROLLOUT_CONCURRENCY_GROUP_GENERATE,
    RolloutConfig,
    RolloutWorker,
)


@dataclass
class WorkerInfo:
    """A data class to hold all state information for a single worker."""

    actor: RolloutWorker
    url: str
    session_url: str | None = None
    is_active: bool = True


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

    # 每个 worker 服务器 URL 的当前活跃状态
    # 键：服务器 URL 字符串
    # 值：布尔值，True 表示该 worker 处于活跃状态，False 表示已失效或停用
    worker_server_urls_status: Dict[str, bool]

    # worker rank -> SessionServer proxy URL. These are the externally
    # registered URLs for routedapiproxy; server_url_dict keeps the original
    # worker URLs for trainer-side weight update / backend control paths.
    worker_session_url_dict: Dict[int, str]

    # SessionServer URL -> active status. This mirrors worker_server_urls_status
    # but is keyed by the proxy URL that external traffic uses.
    worker_session_urls_status: Dict[str, bool]


# Keep this as a Ray actor because Ray AgentLoop actors need a shared, cross-process handle to the same controller
# state; passing a normal Python object would serialize a separate copy into each actor.
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
        self.num_gpus_per_engine = self.config.num_gpus_per_engine
        self.logger = get_logger(log_dir=infer_config.worker_log_dir, tag="RolloutController")
        self.engine_rank_mesh_array: List[List[int]] = []
        self.worker_server_urls_map: dict[int, str] = {}
        self.rank2info: dict[int, WorkerInfo] = {}
        self.engine_rank_mesh_array, self.worker_server_urls_map, self.rank2info = self._init_workers(placement_group)
        self.num_active_workers = len(self.rank2info)
        self.worker_info_lock = threading.RLock()
        # The timeout for the environment to wait for the rollout controller's response.
        # This should be longer than the controller's internal timeout (`rollout_timeout`)
        # to account for potential queuing delays and other overheads.
        self.timeout_multiplier = 2.0
        self.router = SessionRouter(self.rank2info, worker_infos_lock=self.worker_info_lock)
        self.health_checker = RolloutHealthChecker(
            config=self.config,
            workers_info=self.rank2info,
            worker_infos_lock=self.worker_info_lock,
        )
        self.health_checker.start()
        self._tool_call_parser, self._reasoning_parser = self._build_output_parsers()

    def get_rollout_metadata(self) -> RolloutWorkerMetadata:
        """Get information about the current rollout setup.

        Returns:
            dict: A dictionary containing the engine mesh list, server URL
                dictionary, and the rollout configuration.
        """
        with self.worker_info_lock:
            worker_server_urls_status = {info.url: info.is_active for info in self.rank2info.values()}
            worker_session_url_dict = {
                rank: info.session_url for rank, info in self.rank2info.items() if info.session_url is not None
            }
            worker_session_urls_status = {
                info.session_url: info.is_active for info in self.rank2info.values() if info.session_url is not None
            }
        rollout_metadata: RolloutWorkerMetadata = {
            "engine_rank_mesh_array": self.engine_rank_mesh_array,
            "server_url_dict": self.worker_server_urls_map,
            "rollout_config": self.config,
            "worker_server_urls_status": worker_server_urls_status,
            "worker_session_url_dict": worker_session_url_dict,
            "worker_session_urls_status": worker_session_urls_status,
        }
        return rollout_metadata

    def _build_output_parsers(self) -> tuple[ToolCallParser | None, ReasoningParser | None]:
        tool_call_parser = None
        reasoning_parser = None

        if self.config.tool_call_parser != "none":
            tool_call_parser = build_tool_call_parser(self.config.tool_call_parser)

        if self.config.reasoning_parser != "none":
            tokenizer_path = self.config.tokenizer_path or self.config.model_path
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            reasoning_parser = build_reasoning_parser(self.config.reasoning_parser, tokenizer)

        return tool_call_parser, reasoning_parser

    def get_ready_status(self) -> tuple[bool, dict[str, Any]]:
        with self.worker_info_lock:
            active_workers = sum(1 for info in self.rank2info.values() if info.is_active)
            total_workers = len(self.rank2info)
        return active_workers > 0, {
            "active_workers": active_workers,
            "total_workers": total_workers,
        }

    def get_generate_concurrency(self) -> int:
        assert self.config.rollout_max_batch_size_per_instance is not None, (
            "rollout_max_batch_size_per_instance must be set before building AgentLoop."
        )
        concurrency_per_worker = math.ceil(
            self.config.rollout_max_batch_size_per_instance * self.config.allow_over_concurrency_ratio
        )
        with self.worker_info_lock:
            active_workers = sum(1 for info in self.rank2info.values() if info.is_active)
        return active_workers * concurrency_per_worker

    @ray.method(concurrency_group=ROLLOUT_CONCURRENCY_GROUP_GENERATE)
    @trace_function("xtuner.rollout_controller.generate")
    async def generate(self, rollout_state: RolloutState) -> RolloutState:
        if XTUNER_DETERMINISTIC:
            sample_params = rollout_state.sample_params.model_copy(deep=True)
            sample_params.sampling_seed = self.config.random_seed + (
                (rollout_state.uid or 0) - (rollout_state.message_uid or 0)
            )
            rollout_state.sample_params = sample_params

        session_id = rollout_state.session_uid if rollout_state.session_uid is not None else uuid4().int
        worker = await self.router.get_worker(session_id)
        if worker is None:
            rollout_state.status = Status.FAILED
            rollout_state.error_msg = "No active rollout worker available."
            return rollout_state

        response_ref = worker.generate.remote(rollout_state=rollout_state)  # type: ignore[attr-defined]
        try:
            response_rollout_state = await asyncio.wait_for(
                response_ref,
                timeout=self.config.rollout_timeout * self.timeout_multiplier,
            )
            self._apply_output_parsers(response_rollout_state)
            return response_rollout_state
        except asyncio.TimeoutError:
            self.logger.error(
                f"RolloutController.generate timed out waiting for worker: session_id={session_id}, "
                f"timeout={self.config.rollout_timeout * self.timeout_multiplier}"
            )
            rollout_state.status = Status.FAILED
            rollout_state.error_msg = (
                f"Rollout request timed out after {self.config.rollout_timeout * self.timeout_multiplier} seconds."
            )
            return rollout_state

    def _apply_output_parsers(self, rollout_state: RolloutState) -> None:
        """Apply tool-call and reasoning parsers to the rollout state in-
        place."""
        if self._tool_call_parser is not None:
            parsed = self._tool_call_parser.parse(rollout_state)
            rollout_state.tool_calls = parsed.tool_calls
            rollout_state.response = parsed.remaining_text or None
        if self._reasoning_parser is not None:
            parsed_reasoning = self._reasoning_parser.parse(rollout_state)
            rollout_state.response = parsed_reasoning.remaining_text
            if parsed_reasoning.reasoning_text:
                rollout_state.extra_fields["reasoning_text"] = parsed_reasoning.reasoning_text
            else:
                rollout_state.extra_fields.pop("reasoning_text", None)

    def set_enable_partial_rollout(self, enable: bool) -> None:
        """Propagate enable_partial_rollout flag to all active workers."""
        with self.worker_info_lock:
            active_actors = [info.actor for info in self.rank2info.values() if info.is_active]
            ray.get([actor.set_enable_partial_rollout.remote(enable) for actor in active_actors])  # type: ignore[attr-defined]

    def pause_generation(self):
        self.health_checker.pause()
        with self.worker_info_lock:
            active_workers = [info for info in self.rank2info.values() if info.is_active]
        futures = [info.actor.pause_generation.remote() for info in active_workers]  # type: ignore[attr-defined]
        try:
            results = ray.get(futures, timeout=ROLLOUT_RAY_GET_TIMEOUT)
        except Exception:
            self.logger.exception(
                f"RolloutController pause_generation failed for {len(active_workers)} active workers."
            )
            raise
        succeeded_worker_urls = [info.url for info, result in zip(active_workers, results) if result is not False]
        failed_worker_urls = [info.url for info, result in zip(active_workers, results) if result is False]
        if succeeded_worker_urls:
            self.logger.info(f"Abort request sent successfully: count={len(succeeded_worker_urls)}")
        if failed_worker_urls:
            self.logger.warning(f"Abort request failed: worker_urls={failed_worker_urls}")

    def ensure_workers_healthy_before_training(self):
        """Ensure rollout workers are healthy before colocated training
        onloads."""
        health_checker_was_paused = self.health_checker.is_paused()
        if not health_checker_was_paused:
            self.health_checker.pause()
        try:
            with self.worker_info_lock:
                workers = {rank: (info.actor, info.url, info.is_active) for rank, info in self.rank2info.items()}

            for rank, (actor, url, was_active) in workers.items():
                try:
                    is_healthy = ray.get(actor.check_health.remote(), timeout=ROLLOUT_RAY_GET_TIMEOUT)  # type: ignore[attr-defined]
                except Exception as e:
                    is_healthy = False
                    self.logger.warning(f"Final health check raised for rollout worker {rank} at {url}: {e}.")

                if not is_healthy:
                    self.logger.warning(f"Final health check failed for rollout worker {rank} at {url}.")

                with self.worker_info_lock:
                    info = self.rank2info[rank]
                    info.is_active = bool(is_healthy)

                if is_healthy and not was_active:
                    self.logger.info(f"Mark rollout worker {rank} active after final health check: url={url}")
                elif not is_healthy and was_active:
                    self.logger.warning(
                        f"Mark rollout worker {rank} inactive because final health check failed before training: url={url}"
                    )

            self._recover_failed_workers()
            with self.worker_info_lock:
                inactive_workers = [
                    f"rank={rank}, url={info.url}" for rank, info in self.rank2info.items() if not info.is_active
                ]
            if inactive_workers:
                raise RuntimeError(
                    "inactive rollout workers before training: "
                    + ", ".join(inactive_workers)
                    + ". Refusing to onload training workers because rollout GPU memory may still be held."
                )
        finally:
            if not health_checker_was_paused:
                self.health_checker.resume()

    def continue_generation(self):
        self.health_checker.resume()
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
        self.health_checker.stop()
        self._broadcast_to_active_workers("shutdown", stop_session_server=True)

    def _recover_failed_workers(self) -> None:
        """Recover inactive workers before training while keeping health checks
        paused."""
        with self.worker_info_lock:
            failed_workers = [info for info in self.rank2info.values() if not info.is_active]

        if not failed_workers:
            self.logger.info("No failed workers detected during recovery.")
            return

        self.logger.warning(f"Detected {len(failed_workers)} failed workers. Initiating recovery process.")
        for worker in failed_workers:
            if self._restart_failed_workers(worker.actor, expected_url=worker.url):
                with self.worker_info_lock:
                    rank = self._get_rank_by_actor(worker.actor)
                    if rank is not None:
                        self.rank2info[rank].is_active = True

    def _restart_failed_workers(self, worker: RolloutWorker, expected_url: str) -> bool:
        try:
            # 先保证把老的worker关掉
            ray.get(worker.shutdown.remote(), timeout=ROLLOUT_RAY_GET_TIMEOUT)  # type: ignore[attr-defined]
            # 保证新的worker启动在之前的端口上，否则权重更新会出错
            _, url = ray.get(worker.init.remote(), timeout=ROLLOUT_RAY_GET_TIMEOUT)  # type: ignore[attr-defined]
            assert url == expected_url, f"Worker restarted with unexpected URL: expected {expected_url}, got {url}."
            _, session_url = ray.get(worker.get_session_server_info.remote(), timeout=ROLLOUT_RAY_GET_TIMEOUT)  # type: ignore[attr-defined]
            is_healthy = ray.get(worker.check_health.remote(), timeout=ROLLOUT_RAY_GET_TIMEOUT)  # type: ignore[attr-defined]

            if is_healthy:
                self.logger.info(f"Successfully restarted worker {worker} with URL {url}.")
                with self.worker_info_lock:
                    rank = self._get_rank_by_actor(worker)
                    if rank is not None:
                        self.rank2info[rank].url = url
                        self.rank2info[rank].session_url = session_url
                        self.worker_server_urls_map[rank] = url
                return True
            else:
                self.logger.error(f"Worker {worker} is still unhealthy after restart.")
                return False
        except AssertionError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to restart worker: {e}")
            return False

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
            index = list(range(0, self.num_active_workers + 1, tp_size)) + [self.num_active_workers]
            for i in range(1, len(index)):
                dist_init_addrs[index[i - 1] : index[i]] = [dist_init_addrs[index[i - 1]]] * (index[i] - index[i - 1])
        if server_urls_per_engine > 1:
            activate_servers = len(dist_init_addrs)
            for i in range(0, activate_servers, server_urls_per_engine):
                dist_init_addrs[i : i + server_urls_per_engine] = [dist_init_addrs[i]] * server_urls_per_engine
        return dist_init_addrs

    def _broadcast_to_active_workers(self, method_name: str, **kwargs):
        """Helper function to call a method on all active workers.

        Args:
            method_name (str): The name of the method to call.
            block (bool): Whether to block until the call completes.

        Returns:
            A list of futures if `block` is False, otherwise a list of results.
        """
        futures = []
        with self.worker_info_lock:
            active_actors = [info.actor for info in self.rank2info.values() if info.is_active]
        futures = [getattr(actor, method_name).remote(**kwargs) for actor in active_actors]
        results = ray.get(futures, timeout=ROLLOUT_RAY_GET_TIMEOUT)
        return results

    def _get_worker_cls(self):
        if os.environ.get("XTUNER_USE_LMDEPLOY") == "1":
            from .lmdeploy import LMDeployWorker

            worker_cls = LMDeployWorker
        elif os.environ.get("XTUNER_USE_VLLM") == "1":
            from .vllm import vLLMWorker

            worker_cls = vLLMWorker
        elif os.environ.get("XTUNER_USE_SGLANG") == "1":
            from .sglang import SGLangWorker

            worker_cls = SGLangWorker
        else:
            raise NotImplementedError(
                "Rollout backend is not supported."
                "Please set XTUNER_USE_LMDEPLOY or XTUNER_USE_VLLM"
                " or XTUNER_USE_SGLANG environment variable."
            )
        assert self.config.rollout_max_batch_size_per_instance is not None, (
            "rollout_max_batch_size_per_instance must be set before building RolloutWorker."
        )
        worker_generate_max_concurrency = max(
            1000,  # Ray async actor default max_concurrency.
            math.ceil(self.config.rollout_max_batch_size_per_instance * self.config.allow_over_concurrency_ratio),
        )
        return ray.remote(
            concurrency_groups={
                ROLLOUT_CONCURRENCY_GROUP_GENERATE: worker_generate_max_concurrency,
            },
        )(worker_cls)

    def _get_rank_by_actor(self, actor: RolloutWorker) -> Optional[int]:
        """Get rank by actor object.

        Args:
            actor: The RolloutWorker actor object.

        Returns:
            The rank of the worker, or None if not found.
        """
        for rank, info in self.rank2info.items():
            if info.actor == actor:
                return rank
        return None

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

    def _init_workers(self, placement_group: PlacementGroup):
        """Initializes and configures the pool of RolloutWorker actors.

        This method creates workers from the placement group, configures distributed
        inference engines by grouping workers, where each group forms a tensor-parallel
        inference engine. It determines the `active_workers` to act as the head of each
        engine, constructs the `engine_rank_mesh_array` to define engine topology,
        acquires necessary distributed communication ports, and finally launches servers
        on the `active_workers` to get their addresses.

        Returns:
            Tuple[List, Dict]: A tuple where the first element is
            `engine_rank_mesh_array`, a list of lists containing the ranks of workers
            in each engine, and the second element is `worker_server_urls_map`,
            a dictionary mapping the rank of each active worker to its
            corresponding server URL.
        """
        # Create workers from placement group
        workers, rank_bundle_idx_list = AutoAcceleratorWorkers.from_placement_group(
            self._get_worker_cls(), self.config, placement_group
        )
        active_servers_count, nodes_per_engine = self.config.get_active_servers_count(len(workers))
        interval = len(workers) // active_servers_count
        active_rollout_workers = workers[::interval]
        server_urls_per_engine = self.config.server_urls_per_engine

        set_bundle_idxs_objectref = []
        engine_rank_mesh_array = []
        activate_worker_idx = 0
        for active_worker in active_rollout_workers:
            head_rank, _ = rank_bundle_idx_list[activate_worker_idx]
            engine_workers_meta = rank_bundle_idx_list[head_rank : head_rank + interval]
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
        init_results = ray.get(
            [worker.init.remote(dist_init_addrs[i]) for i, worker in enumerate(active_rollout_workers)]
        )
        worker_server_urls_map = dict(init_results)  # rank -> url
        worker_session_url_dict = dict(
            ray.get([worker.get_session_server_info.remote() for worker in active_rollout_workers])
        )
        active_rollout_workers, worker_server_urls_map = self._update_active_workers_and_urls_map(
            active_rollout_workers, worker_server_urls_map
        )
        active_ranks = list(worker_server_urls_map.keys())
        worker_session_url_dict = {rank: worker_session_url_dict[rank] for rank in active_ranks}
        workers_info = {}
        for i in range(len(active_rollout_workers)):
            rank = list(worker_server_urls_map.keys())[i]
            url = worker_server_urls_map[rank]
            workers_info[rank] = WorkerInfo(
                actor=active_rollout_workers[i],
                url=url,
                session_url=worker_session_url_dict[rank],
            )
        self.logger.info(f"Rollout worker server URLs: {[info.url for info in workers_info.values()]}")
        self.logger.info(f"Rollout worker session server URLs: {[info.session_url for info in workers_info.values()]}")
        return engine_rank_mesh_array, worker_server_urls_map, workers_info


RayRolloutController = ray.remote(RolloutController)
RolloutControllerProxy: TypeAlias = ActorProxy[RayRolloutController]
