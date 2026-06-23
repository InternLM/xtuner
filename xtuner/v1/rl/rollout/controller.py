import asyncio
from typing import TypeAlias
from uuid import uuid4

import ray
from ray.actor import ActorProxy
from ray.util.placement_group import PlacementGroup

from transformers import AutoTokenizer
from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.rl.utils import AutoAcceleratorWorkers
from xtuner.v1.utils import XTUNER_DETERMINISTIC, get_logger

from .constants import ROLLOUT_RAY_GENERATE_MAX_CONCURRENCY
from .health_manager import ROLLOUT_RAY_GET_TIMEOUT, RolloutHealthManager
from .parser.factory import build_reasoning_parser, build_tool_call_parser
from .parser.reasoning_parser import ReasoningParser
from .parser.tool_parser import ToolCallParser
from .proxy_manager import RolloutProxyManager
from .utils import SessionRouter
from .worker import (
    ROLLOUT_CONCURRENCY_GROUP_GENERATE,
    RolloutConfig,
    get_rollout_worker_base_cls,
)
from .worker_registry import RolloutWorkerMetadata, RolloutWorkerRegistry


# Keep this as a Ray actor because Ray AgentLoop actors need a shared, cross-process handle to the same controller
# state; passing a normal Python object would serialize a separate copy into each actor.
class RolloutController:
    """Control-plane entrypoint for rollout traffic and worker startup.

    The controller creates workers, routes generate requests, and broadcasts training lifecycle commands. Health state
    transitions and worker recovery belong to RolloutHealthManager.
    """

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
        self.registry = self._init_workers(placement_group)
        # The timeout for the environment to wait for the rollout controller's response.
        # This should be longer than the controller's internal timeout (`rollout_timeout`)
        # to account for potential queuing delays and other overheads.
        self.timeout_multiplier = 2.0
        self.router = SessionRouter(self.registry)
        self.proxy_manager: RolloutProxyManager | None = None
        if self.config.enable_proxy:
            self.proxy_manager = RolloutProxyManager(self.config)
            self.register_active_workers_to_proxy()
        self.health_manager = RolloutHealthManager(
            config=self.config,
            registry=self.registry,
            worker_lifecycle_listeners=[self.proxy_manager] if self.proxy_manager is not None else None,
        )
        self.health_manager.start()
        self._tool_call_parser, self._reasoning_parser = self._build_output_parsers()

    def get_rollout_metadata(self) -> RolloutWorkerMetadata:
        """Get information about the current rollout setup.

        Returns:
            dict: A dictionary containing the engine mesh list, server URL
                dictionary, and the rollout configuration.
        """
        rollout_metadata = self.registry.training_metadata_snapshot()
        self.logger.info(f"Rollout worker server URLs: {rollout_metadata['server_url_dict']}")
        self.logger.info(f"Rollout worker session server URLs: {rollout_metadata['worker_session_url_dict']}")
        return rollout_metadata

    def register_active_workers_to_proxy(self) -> None:
        assert self.proxy_manager is not None, "Proxy manager must be initialized"
        session_urls = sorted(
            worker.session_url for worker in self.registry.active_entrypoints() if worker.session_url is not None
        )
        self.proxy_manager.replace_registered_session_urls(session_urls)

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

    @ray.method(concurrency_group=ROLLOUT_CONCURRENCY_GROUP_GENERATE)
    async def generate(self, rollout_state: RolloutState) -> RolloutState:
        if XTUNER_DETERMINISTIC:
            sample_params = rollout_state.sample_params.model_copy(deep=True)
            sample_params.sampling_seed = self.config.random_seed + (
                (rollout_state.rollout_id or 0) - (rollout_state.group_id or 0)
            )
            rollout_state.sample_params = sample_params

        session_id = rollout_state.session_id if rollout_state.session_id is not None else uuid4().int
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
        active_workers = self.registry.active_workers()
        ray.get(
            [
                worker.actor.set_enable_partial_rollout.remote(enable)  # type: ignore[attr-defined]
                for worker in active_workers
            ]
        )

    def pause_generation(self):
        self.health_manager.pause()
        active_workers = self.registry.active_workers()
        futures = [
            worker.actor.pause_generation.remote()  # type: ignore[attr-defined]
            for worker in active_workers
        ]
        try:
            results = ray.get(futures, timeout=ROLLOUT_RAY_GET_TIMEOUT)
        except Exception:
            self.logger.exception(
                f"RolloutController pause_generation failed for {len(active_workers)} active workers."
            )
            raise
        succeeded_worker_urls = [worker.url for worker, result in zip(active_workers, results) if result is not False]
        failed_worker_urls = [worker.url for worker, result in zip(active_workers, results) if result is False]
        if succeeded_worker_urls:
            self.logger.info(f"Abort request sent successfully: count={len(succeeded_worker_urls)}")
        if failed_worker_urls:
            self.logger.warning(f"Abort request failed: worker_urls={failed_worker_urls}")

    async def check_and_shutdown_inactive_workers(self):
        """Run a fail-fast health barrier and shut down failed groups so
        training can reuse shared rollout resources."""
        await asyncio.to_thread(self.health_manager.check_and_shutdown_inactive_workers)

    async def restart_inactive_workers(self):
        """Restart inactive groups before a sync-step weight update."""
        await asyncio.to_thread(self.health_manager.restart_inactive_workers)

    def continue_generation(self):
        self._broadcast_to_active_workers("continue_generation")
        self.health_manager.resume()

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
        """Shut down all rollout workers tracked by the controller."""
        self.health_manager.stop()
        actors = self.registry.all_actors()
        ray.get(
            [actor.shutdown.remote(stop_session_server=True) for actor in actors],  # type: ignore[attr-defined]
            timeout=ROLLOUT_RAY_GET_TIMEOUT,
        )

    def _broadcast_to_active_workers(self, method_name: str, **kwargs):
        workers = self.registry.active_workers()
        futures = [getattr(worker.actor, method_name).remote(**kwargs) for worker in workers]
        return ray.get(futures, timeout=ROLLOUT_RAY_GET_TIMEOUT)

    def _build_remote_worker_cls(self, worker_base_cls):
        assert self.config.rollout_max_batch_size_per_instance is not None, (
            "rollout_max_batch_size_per_instance must be set before building RolloutWorker."
        )
        return ray.remote(
            concurrency_groups={
                ROLLOUT_CONCURRENCY_GROUP_GENERATE: ROLLOUT_RAY_GENERATE_MAX_CONCURRENCY,
            },
        )(worker_base_cls)

    def _init_workers(self, placement_group: PlacementGroup) -> RolloutWorkerRegistry:
        """Initializes and configures the pool of RolloutWorker actors.

        This method follows the same high-level flow as the legacy implementation:
        create workers, initialize worker-local ports, build engine groups,
        select workers that launch rollout servers, launch servers, and
        expose request-entrypoint server URLs to rollout traffic.

        Returns:
            A registry containing all server-process workers and the public
            training metadata mesh.
        """
        worker_base_cls = get_rollout_worker_base_cls(self.config)
        worker_cls = self._build_remote_worker_cls(worker_base_cls)

        # Create workers from placement group.
        workers, rank_bundle_idx_list = AutoAcceleratorWorkers.from_placement_group(
            worker_cls, self.config, placement_group
        )
        rank_to_actor = {rank: worker for (rank, _), worker in zip(rank_bundle_idx_list, workers)}

        # Reserve worker-local ports for all actors first. build_engine_launch_specs
        # uses the returned addresses to bind each ServerProcessSpec to its
        # logical engine rendezvous address; only server-process owners call init().
        rank_to_dist_init_addr = {
            rank: dist_init_addr
            for (rank, _), dist_init_addr in zip(
                rank_bundle_idx_list,
                ray.get([worker.init_dist_port.remote() for worker in workers]),  # type: ignore[attr-defined]
            )
        }

        # Build engine groups and server-process specs from the rank/bundle mapping.
        engine_launch_specs = worker_base_cls.build_engine_launch_specs(
            self.config,
            rank_bundle_idx_list,
            rank_to_dist_init_addr,
        )
        # Keep the public metadata mesh compatible with origin/main. Backends
        # may expose a different update-weight mesh than their internal launch
        # topology, e.g. LMDeploy EP has one logical engine but one public entry
        # per request-serving EP rank.
        engine_rank_mesh_array = worker_base_cls.build_metadata_engine_rank_mesh_array(engine_launch_specs)

        # Launch every server process described by the backend-specific specs.
        server_rank_to_url = dict(
            ray.get(
                [
                    rank_to_actor[server_process.worker_rank].init.remote(  # type: ignore[attr-defined]
                        engine_launch_spec=engine_spec,
                    )
                    for engine_spec in engine_launch_specs
                    for server_process in engine_spec.server_processes
                ]
            )
        )
        session_url_by_rank = dict(
            ray.get(
                [
                    (
                        rank_to_actor[server_process.worker_rank].get_session_server_info.remote()  # type: ignore[attr-defined]
                    )
                    for engine_spec in engine_launch_specs
                    for server_process in engine_spec.server_processes
                ]
            )
        )

        registry = RolloutWorkerRegistry(
            engine_rank_mesh_array=engine_rank_mesh_array,
            rollout_config=self.config,
        )
        for engine_spec in engine_launch_specs:
            for server_process in engine_spec.server_processes:
                rank = server_process.worker_rank
                url = server_rank_to_url[rank]
                session_url = session_url_by_rank.get(rank)
                if server_process.accepts_rollout_requests and session_url is None:
                    raise RuntimeError(f"Rollout worker rank={rank} did not return session server URL during init.")
                registry.register_started_server(
                    rank=rank,
                    actor=rank_to_actor[rank],
                    server_url=url,
                    session_url=session_url,
                    lifecycle_group_ranks=engine_spec.server_worker_ranks,
                    is_request_entrypoint=server_process.accepts_rollout_requests,
                )

        server_process_workers_info = registry.all_workers()
        self.logger.info(f"Rollout server-process worker URLs: {[info.url for info in server_process_workers_info]}")
        lifecycle_groups = sorted({info.lifecycle_group_ranks for info in server_process_workers_info})
        self.logger.info(f"Rollout worker lifecycle groups: {lifecycle_groups}")
        return registry


RayRolloutController = ray.remote(RolloutController)
RolloutControllerProxy: TypeAlias = ActorProxy[RayRolloutController]
