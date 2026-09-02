import asyncio
from typing import Any, TypeAlias
from uuid import uuid4

import ray
from ray.actor import ActorProxy
from ray.util.placement_group import PlacementGroup

from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.rl.trace.rollout_api import trace_rollout_endpoint, trace_rollout_remote
from xtuner.v1.rl.utils import AutoAcceleratorWorkers
from xtuner.v1.rl.weight_update.data import RolloutWeightUpdateTarget
from xtuner.v1.utils import XTUNER_DETERMINISTIC, get_logger

from .constants import ROLLOUT_RAY_GENERATE_MAX_CONCURRENCY
from .health_manager import ROLLOUT_RAY_GET_TIMEOUT, RolloutHealthManager
from .proxy_manager import RolloutProxyManager
from .rollout_topology import RolloutTopology
from .utils import SessionRouter
from .worker import (
    ROLLOUT_CONCURRENCY_GROUP_GENERATE,
    RolloutConfig,
    get_rollout_worker_base_cls,
)
from .worker_registry import RolloutWorkerRegistry, WorkerLifecycleState


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

    def get_weight_update_targets(
        self, target_state: WorkerLifecycleState | None = None, return_group_ranks: bool = False
    ) -> (
        list[RolloutWeightUpdateTarget]
        | tuple[
            list[RolloutWeightUpdateTarget],
            list[tuple[int, ...]],
        ]
    ):
        """Return rollout weight-update targets and their lifecycle groups."""

        target_states: tuple[WorkerLifecycleState, ...]
        if target_state is None:
            target_states = (
                WorkerLifecycleState.PENDING_WEIGHTS,
                WorkerLifecycleState.ACTIVE,
                WorkerLifecycleState.INACTIVE,
            )
        else:
            target_states = (target_state,)
        target_state_values = {state.value for state in target_states}
        targets, group_ranks = self.registry.weight_update_targets()

        filtered_targets = [target for target in targets if target.lifecycle_state in target_state_values]

        if not return_group_ranks:
            return filtered_targets

        endpoint_ranks = {target.endpoint_rank for target in filtered_targets}
        filtered_group_ranks = [ranks for ranks in group_ranks if endpoint_ranks.intersection(ranks)]

        return filtered_targets, filtered_group_ranks

    def inject_backend_crash_for_test(self, *, rank: int = 0) -> None:
        """Crash one active rollout backend for the immediate-recovery test."""
        worker = self.registry.active_entrypoint_by_rank(rank)
        if worker is None:
            raise RuntimeError(f"No active rollout request entrypoint found for test fault injection: rank={rank}.")

        accepted = ray.get(
            worker.actor.inject_backend_crash_for_test.remote(),  # type: ignore[attr-defined]
            timeout=ROLLOUT_RAY_GET_TIMEOUT,
        )
        if not accepted:
            raise RuntimeError(f"Rollout worker rejected test fault injection: rank={rank}, url={worker.url}.")
        self.logger.warning(f"[ImmediateRecoveryExperiment] backend_crash_injected rank={rank} url={worker.url}")

    def register_active_workers_to_proxy(self) -> None:
        if self.proxy_manager is None:
            return
        session_urls = sorted(
            worker.session_url for worker in self.registry.active_entrypoints() if worker.session_url is not None
        )
        self.proxy_manager.replace_registered_session_urls(session_urls)

    def validate_registered_workers_to_proxy(self) -> None:
        if self.proxy_manager is None:
            return
        self.proxy_manager.validate_registered_session_urls()

    @trace_rollout_endpoint("rollout.controller.generate")
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

        response_ref = trace_rollout_remote(
            worker.generate,  # type: ignore[attr-defined]
            rollout_state=rollout_state,
        )
        try:
            response_rollout_state = await asyncio.wait_for(
                response_ref,
                timeout=self.config.rollout_timeout * self.timeout_multiplier,
            )
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
        # Wait for the health manager to finish recovery before pausing generation.
        if not self.health_manager.wait_recovery_done(timeout=600.0):
            raise TimeoutError("Timed out waiting for rollout worker recovery before training.")
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

    async def shutdown_inactive_workers(self):
        """Shut down failed groups so training can reuse shared rollout
        resources."""
        await asyncio.to_thread(self.health_manager.shutdown_inactive_workers)

    async def restart_inactive_workers(self):
        """Restart inactive groups before a sync-step weight update."""
        groups = await asyncio.to_thread(self.health_manager.restart_inactive_workers)
        return tuple(group.ranks for group in groups)

    def mark_worker_groups_lifecycle_state(
        self,
        group_ranks: list[tuple[int, ...]] | None = None,
        *,
        source_state: WorkerLifecycleState,
        target_state: WorkerLifecycleState,
    ) -> None:
        """Move selected worker groups from source_state to target_state.

        When group_ranks is omitted, every complete worker group currently in source_state is moved. When it is
        provided, only exact matching groups are considered. Transitions to ACTIVE or INACTIVE notify the health
        manager so routing and lifecycle listeners stay in sync.
        """
        source_groups = self.registry.get_target_state_worker_groups(source_state)
        # 只对目标group中命中状态的worker进行状态更新，若不提供目标group，则对所有source_state状态的worker进行状态更新
        if group_ranks is None:
            groups = source_groups
        else:
            groups_by_ranks = {group.ranks: group for group in source_groups}
            groups = tuple(groups_by_ranks[ranks] for ranks in group_ranks if ranks in groups_by_ranks)
        updated_groups = self.registry.set_groups_state(
            groups,
            target_state,
            source_state=source_state,
        )
        if target_state is WorkerLifecycleState.ACTIVE:
            self.health_manager.notify_worker_group_active(updated_groups)
        elif target_state is WorkerLifecycleState.INACTIVE:
            self.health_manager.notify_worker_group_inactive(updated_groups)

    def continue_generation(self):
        self._broadcast_to_workers("continue_generation", WorkerLifecycleState.ACTIVE)
        self.health_manager.resume()

    def offload(self):
        self._broadcast_to_workers("offload", WorkerLifecycleState.ACTIVE)

    def flush_cache(self):
        self._broadcast_to_active_workers("flush_cache")

    def onload(self):
        self._broadcast_to_workers("onload_weights", WorkerLifecycleState.ACTIVE)
        self._broadcast_to_workers("onload_kvcache", WorkerLifecycleState.ACTIVE)

    def onload_weights(self, target_state: WorkerLifecycleState = WorkerLifecycleState.ACTIVE):
        self._broadcast_to_workers("onload_weights", target_state)

    def onload_kvcache(self, target_state: WorkerLifecycleState = WorkerLifecycleState.ACTIVE):
        self._broadcast_to_workers("onload_kvcache", target_state)

    def shutdown(self):
        """Shut down all rollout workers tracked by the controller."""
        self.health_manager.stop()
        actors = self.registry.all_actors()
        ray.get(
            [actor.shutdown.remote(stop_session_server=True) for actor in actors],  # type: ignore[attr-defined]
            timeout=ROLLOUT_RAY_GET_TIMEOUT,
        )

    def _broadcast_to_workers(self, method_name: str, target_state: WorkerLifecycleState, **kwargs):
        workers = self.registry.get_target_state_workers(target_state)
        futures = [getattr(worker.actor, method_name).remote(**kwargs) for worker in workers]
        return ray.get(futures, timeout=ROLLOUT_RAY_GET_TIMEOUT)

    def _build_remote_worker_cls(self, worker_base_cls):
        assert self.config.rollout_max_batch_size_per_instance is not None, (
            "rollout_max_batch_size_per_instance must be set before building RolloutWorker."
        )
        from xtuner.v1.rl.trace import get_trace_env_vars

        trace_env_vars = get_trace_env_vars()
        ray_kwargs = {}
        if trace_env_vars:
            ray_kwargs["runtime_env"] = {"env_vars": trace_env_vars}
        return ray.remote(
            concurrency_groups={
                ROLLOUT_CONCURRENCY_GROUP_GENERATE: ROLLOUT_RAY_GENERATE_MAX_CONCURRENCY,
            },
            **ray_kwargs,
        )(worker_base_cls)

    def _create_worker_actors(
        self,
        placement_group: PlacementGroup,
    ) -> tuple[tuple[Any, ...], tuple[tuple[int, int], ...]]:
        """Create rollout worker actors.

        Returns workers_by_rank, which is indexed by rollout worker rank, and rank_bundle_indices, which maps worker
        ranks to placement-group bundles.
        """
        worker_base_cls = get_rollout_worker_base_cls(self.config)
        worker_cls = self._build_remote_worker_cls(worker_base_cls)
        workers, rank_bundle_indices = AutoAcceleratorWorkers.from_placement_group(
            worker_cls, self.config, placement_group
        )
        workers_by_rank = tuple(workers)
        return workers_by_rank, tuple(rank_bundle_indices)

    def _initialize_worker_ports_and_build_rollout_topology(
        self,
        workers_by_rank: tuple[Any, ...],
        rank_bundle_indices: tuple[tuple[int, int], ...],
    ) -> RolloutTopology:
        """Initialize worker-local dist ports and build rollout topology.

        This performs the Ray init_dist_port handshake before building the topology, so the returned layout is bound to
        runtime worker addresses.
        """
        dist_init_results = ray.get(
            [
                worker.init_dist_port.remote()  # type: ignore[attr-defined]
                for worker in workers_by_rank
            ]
        )
        worker_base_cls = get_rollout_worker_base_cls(self.config)
        return worker_base_cls.build_rollout_topology(
            self.config,
            list(rank_bundle_indices),
            dict(dist_init_results),
        )

    def _init_workers(
        self,
        placement_group: PlacementGroup,
    ) -> RolloutWorkerRegistry:
        """Initializes and configures the pool of RolloutWorker actors.

        This method follows the same high-level flow as the legacy implementation:
        create workers, initialize worker-local ports, build the bound rollout
        topology, launch rollout servers, and expose request-entrypoint server
        URLs to rollout traffic.

        Returns:
            A registry containing all server-process workers and runtime state.
        """
        workers_by_rank, rank_bundle_indices = self._create_worker_actors(placement_group)
        rollout_topology = self._initialize_worker_ports_and_build_rollout_topology(
            workers_by_rank,
            rank_bundle_indices,
        )
        init_results = tuple(
            ray.get(
                [
                    workers_by_rank[launch_spec.worker_rank].init.remote(launch_spec)  # type: ignore[attr-defined]
                    for launch_spec in rollout_topology.server_launch_specs()
                ]
            )
        )

        registry = RolloutWorkerRegistry(rollout_topology=rollout_topology)
        registry.register_started_servers(
            init_results=init_results,
            workers_by_rank=workers_by_rank,
        )

        self.logger.info(
            "Rollout worker registry snapshot: "
            f"weight_update_targets={registry.weight_update_targets()}, "
            f"active_entrypoints={registry.active_entrypoints()}, "
            f"server_process_urls={[worker.url for worker in registry.all_workers()]}, "
            f"lifecycle_groups={registry.lifecycle_groups()}"
        )
        return registry


RayRolloutController = ray.remote(RolloutController)
RolloutControllerProxy: TypeAlias = ActorProxy[RayRolloutController]
