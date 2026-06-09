import asyncio
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypeAlias, TypedDict
from uuid import uuid4

import ray
import requests
from ray.actor import ActorProxy
from ray.util.placement_group import PlacementGroup

from transformers import AutoTokenizer
from xtuner.v1.data_proto.rl_data import RolloutState, Status
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
    dist_init_addr: str = ""
    ep_group_ranks: tuple[int, ...] = ()


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
                if not was_active:
                    continue

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

                if not is_healthy:
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
            failed_group_to_ranks: dict[tuple[int, ...], list[int]] = {}
            for rank, info in self.rank2info.items():
                if info.is_active:
                    continue
                group_ranks = info.ep_group_ranks or (rank,)
                failed_group_to_ranks.setdefault(group_ranks, []).append(rank)

        if not failed_group_to_ranks:
            self.logger.info("No failed workers detected during recovery.")
            return

        failed_groups = set(failed_group_to_ranks)
        for group_ranks in sorted(failed_groups):
            failed_ranks = sorted(failed_group_to_ranks[group_ranks])
            if len(group_ranks) > 1:
                related_restart_ranks = [rank for rank in group_ranks if rank not in failed_ranks]
                self.logger.warning(
                    f"Detected failed rollout worker ranks={failed_ranks}; "
                    f"restart_group_ranks={group_ranks}, related_restart_ranks={related_restart_ranks}."
                )
            else:
                self.logger.warning(f"Detected failed rollout worker rank={failed_ranks[0]}. Initiating recovery.")

        with self.worker_info_lock:
            for group_ranks in failed_groups:
                for rank in group_ranks:
                    if rank in self.rank2info:
                        self.rank2info[rank].is_active = False

        for group_ranks in sorted(failed_groups):
            if self._restart_worker_group(group_ranks):
                with self.worker_info_lock:
                    for rank in group_ranks:
                        self.rank2info[rank].is_active = True

    def _restart_worker_group(self, group_ranks: tuple[int, ...]) -> bool:
        workers: list[tuple[int, RolloutWorker, str, str]] = []
        try:
            with self.worker_info_lock:
                workers = [
                    (
                        rank,
                        self.rank2info[rank].actor,
                        self.rank2info[rank].url,
                        self.rank2info[rank].dist_init_addr,
                    )
                    for rank in group_ranks
                ]

            if not self._cleanup_inactive_worker_group(workers):
                return False
            init_refs = [
                actor.init.remote(dist_init_addr=dist_init_addr)  # type: ignore[attr-defined]
                for _, actor, _, dist_init_addr in workers
            ]
            init_results = ray.get(init_refs, timeout=ROLLOUT_RAY_GET_TIMEOUT)
            if len(init_results) != len(workers):
                raise RuntimeError(
                    f"Restarted rollout worker group ranks={group_ranks} returned "
                    f"{len(init_results)} init results, expected {len(workers)}."
                )

            for worker_info, init_result in zip(workers, init_results):
                rank, _, expected_url, _ = worker_info
                actual_rank, actual_url = init_result
                if actual_rank != rank or actual_url != expected_url:
                    raise RuntimeError(
                        f"Restarted rollout worker {rank} returned rank={actual_rank}, url={actual_url}; "
                        f"expected rank={rank}, url={expected_url}."
                    )

            infer_not_ready_ranks = self._check_worker_group_infer_ready_after_restart(workers)
            if infer_not_ready_ranks:
                raise RuntimeError(
                    f"Restarted rollout worker group ranks={group_ranks} has ranks not ready for inference: "
                    f"{infer_not_ready_ranks}."
                )

            # SessionServer is not stopped during worker restart, so its URL in rank2info remains valid.
            self.logger.info(f"Successfully restarted rollout worker group ranks={group_ranks}.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to restart worker group ranks={group_ranks}: {e}")
            if workers:
                self._cleanup_inactive_worker_group(workers)
            return False

    def _cleanup_inactive_worker_group(self, workers: list[tuple[int, RolloutWorker, str, str]]) -> bool:
        shutdown_succeeded = True
        for rank, actor, url, _ in workers:
            try:
                ray.get(actor.shutdown.remote(), timeout=60)  # type: ignore[attr-defined]
            except Exception as e:
                shutdown_succeeded = False
                self.logger.warning(f"Cleanup shutdown failed for rollout worker {rank} at {url}: {e}")
                continue
            if not self._wait_worker_server_down_after_shutdown(rank, url):
                shutdown_succeeded = False
        return shutdown_succeeded

    def _wait_worker_server_down_after_shutdown(
        self,
        rank: int,
        url: str,
        *,
        max_attempts: int = 60,
        retry_interval_seconds: float = 5.0,
    ) -> bool:
        endpoint_url = f"{url}/health"
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        for attempt in range(1, max_attempts + 1):
            try:
                response = requests.get(endpoint_url, headers=headers, timeout=1.0)
            except requests.RequestException:
                return True
            if attempt < max_attempts:
                self.logger.warning(
                    f"Rollout worker rank={rank} server still responds after shutdown "
                    f"attempt={attempt}/{max_attempts}, url={url}, status={response.status_code}."
                )
                time.sleep(retry_interval_seconds)
        self.logger.error(f"Rollout worker rank={rank} server did not stop after shutdown: url={url}.")
        return False

    def _check_worker_group_infer_ready_after_restart(
        self,
        workers: list[tuple[int, RolloutWorker, str, str]],
        *,
        max_attempts: int = 60,
        retry_interval_seconds: float = 5.0,
    ) -> list[int]:
        # NOTE: 推理引擎现在仅依靠 /health 接口来判断是否 ready, 有可能会出现server已经ready了，但是推理引擎还未ready的情况，所以重启后需要判断推理引擎是否能够推理成功才认为启动成功
        pending = {rank: url for rank, _, url, _ in workers}
        for attempt in range(1, max_attempts + 1):
            for rank, url in list(pending.items()):
                if self._check_chat_completion_ready_once(rank, url, attempt, max_attempts):
                    del pending[rank]
            if not pending:
                return []
            if attempt < max_attempts:
                time.sleep(retry_interval_seconds)
        return sorted(pending)

    def _check_chat_completion_ready_once(
        self,
        rank: int,
        url: str,
        attempt: int,
        max_attempts: int,
    ) -> bool:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        payload = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.0,
            "top_p": 1.0,
            "n": 1,
            "stream": False,
            "max_tokens": 1,
            "repetition_penalty": 1.0,
            "top_k": 1,
            "skip_special_tokens": True,
        }
        endpoint_url = f"{url}/v1/chat/completions"
        try:
            response = requests.post(endpoint_url, headers=headers, json=payload, timeout=60)
        except requests.RequestException as e:
            self.logger.warning(
                f"Restarted rollout worker rank={rank} inference readiness request failed "
                f"attempt={attempt}/{max_attempts}: {e}"
            )
            return False

        if response.status_code != 200:
            self.logger.warning(
                f"Restarted rollout worker rank={rank} inference readiness request returned "
                f"status={response.status_code}, attempt={attempt}/{max_attempts}, response={response.text}"
            )
            return False
        return True

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

    @staticmethod
    def _build_rank_to_ep_group(active_ranks: list[int], server_urls_per_engine: int) -> dict[int, tuple[int, ...]]:
        if server_urls_per_engine <= 1:
            return {rank: (rank,) for rank in active_ranks}

        assert len(active_ranks) % server_urls_per_engine == 0, (
            f"active rollout worker count {len(active_ranks)} must be divisible by "
            f"server_urls_per_engine={server_urls_per_engine}"
        )
        rank_to_ep_group: dict[int, tuple[int, ...]] = {}
        for start in range(0, len(active_ranks), server_urls_per_engine):
            ep_group = tuple(active_ranks[start : start + server_urls_per_engine])
            for rank in ep_group:
                rank_to_ep_group[rank] = ep_group
        return rank_to_ep_group

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
        worker_dist_init_addr_map = {rank: dist_init_addrs[i] for i, (rank, _) in enumerate(init_results)}
        active_rollout_workers, worker_server_urls_map = self._update_active_workers_and_urls_map(
            active_rollout_workers, worker_server_urls_map
        )
        active_ranks = list(worker_server_urls_map.keys())
        rank_to_ep_group = self._build_rank_to_ep_group(active_ranks, server_urls_per_engine)
        worker_session_url_dict = dict(
            ray.get(
                [worker.get_session_server_info.remote() for worker in active_rollout_workers],  # type: ignore[attr-defined]
                timeout=ROLLOUT_RAY_GET_TIMEOUT,
            )
        )
        workers_info = {}
        for i, rank in enumerate(active_ranks):
            url = worker_server_urls_map[rank]
            workers_info[rank] = WorkerInfo(
                actor=active_rollout_workers[i],
                url=url,
                session_url=worker_session_url_dict.get(rank),
                dist_init_addr=worker_dist_init_addr_map[rank],
                ep_group_ranks=rank_to_ep_group[rank],
            )
        self.logger.info(f"Rollout worker server URLs: {[info.url for info in workers_info.values()]}")
        self.logger.info(
            f"Rollout worker session URLs: {[info.session_url for info in workers_info.values() if info.session_url]}"
        )
        self.logger.info(f"Rollout worker EP groups: {sorted(set(rank_to_ep_group.values()))}")
        return engine_rank_mesh_array, worker_server_urls_map, workers_info


RayRolloutController = ray.remote(RolloutController)
RolloutControllerProxy: TypeAlias = ActorProxy[RayRolloutController]
