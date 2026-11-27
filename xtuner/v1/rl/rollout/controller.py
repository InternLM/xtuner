import asyncio
from dataclasses import dataclass
from time import monotonic
from typing import Any, Callable, TypeAlias, cast
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
from .cross_weight_update_rollout import CrossWeightUpdateRolloutConfig, RequestBarrier
from .health_manager import ROLLOUT_RAY_GET_TIMEOUT, RolloutHealthManager
from .proxy_manager import RolloutProxyManager
from .rollout_topology import RolloutTopology
from .utils import SessionRouter
from .worker import (
    ROLLOUT_CONCURRENCY_GROUP_GENERATE,
    RolloutConfig,
    get_rollout_worker_base_cls,
)
from .worker_registry import RolloutWorkerRegistry, WorkerLifecycleState, WorkerSnapshot


@dataclass
class RolloutRequestState:
    """Controller-owned state for requests spanning rollout weight updates."""

    epoch: int = 0
    barrier: RequestBarrier | None = None
    published_model_step: int | None = None


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
        self._request_state = RolloutRequestState()

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

    def assert_request_gates_healthy(self) -> None:
        """Fail fast if an admission gate or its settled-request ledger is
        unhealthy."""
        if self.config.cross_weight_update_rollout is None:
            raise RuntimeError("Cross-weight-update rollout request gates are not enabled")
        workers = self.registry.active_entrypoints()
        servers = ray.get(
            [worker.actor.get_session_server.remote() for worker in workers],  # type: ignore[attr-defined]
            timeout=ROLLOUT_RAY_GET_TIMEOUT,
        )
        if not servers or any(server is None for server in servers):
            raise RuntimeError("A fixed request entrypoint is missing")
        states = ray.get([server.request_status.remote() for server in servers], timeout=ROLLOUT_RAY_GET_TIMEOUT)
        failures = [state for state in states if state["failures"]]
        if failures:
            raise RuntimeError(f"Unsettled requests stop this run: {failures}")

    def pause_generation(self) -> None:
        """Pause rollout, settling admitted requests before the backend
        pauses."""
        settings = self.config.cross_weight_update_rollout
        if settings is not None and self._request_state.barrier is not None:
            raise RuntimeError("A request barrier already exists")
        self.health_manager.pause()
        # TODO(timeout-reconciliation): freeze_timeout also bounds health recovery, SessionServer lookup and seal;
        # every timeout currently aborts training. Distinguish a slow control RPC from a real failure: ray.get
        # timeout does not cancel the remote call, and a partial seal must not advance or rerun pause blindly. Keep
        # the barrier and pending RPCs, poll each entrypoint's state, and continue only after all of them meet the
        # current stage condition. (PR #2297 review.)
        # Wait for the health manager to finish recovery before pausing generation.
        timeout = settings.freeze_timeout if settings is not None else 600.0
        if not self.health_manager.wait_recovery_done(timeout=timeout):
            raise TimeoutError("Timed out waiting for rollout worker recovery before training.")
        if settings is not None:
            self._seal_and_drain_requests(settings)
        active_workers = self.registry.active_workers()
        try:
            results = self._broadcast_to_workers("pause_generation", WorkerLifecycleState.ACTIVE)
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
            if settings is not None:
                raise RuntimeError("Backend pause failed under request barrier")
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

    def continue_generation(self, model_step: int) -> None:
        """Resume rollout using the version of the weights actually installed.

        Args:
            model_step (int): Actual rollout model version, including zero for the initial weights.
        """
        if self.config.cross_weight_update_rollout is not None:
            self._resume_with_request_barrier(model_step)
            return
        self._broadcast_to_workers("continue_generation", WorkerLifecycleState.ACTIVE)
        self.health_manager.resume()

    def offload(self):
        self._broadcast_to_workers("offload", WorkerLifecycleState.ACTIVE)

    def onload(self):
        self._broadcast_to_workers("onload_weights", WorkerLifecycleState.ACTIVE)
        self._broadcast_to_workers("onload_kvcache", WorkerLifecycleState.ACTIVE)

    def onload_weights(self, target_state: WorkerLifecycleState = WorkerLifecycleState.ACTIVE):
        self._broadcast_to_workers("onload_weights", target_state)

    def onload_kvcache(self, target_state: WorkerLifecycleState = WorkerLifecycleState.ACTIVE):
        self._broadcast_to_workers("onload_kvcache", target_state)

    def retire_route_sessions(self, session_ids: list[str]) -> None:
        """Retire finished Agent sessions so a late retry cannot start a second
        attempt."""
        if self.proxy_manager is not None:
            self.proxy_manager.retire_sessions(session_ids)

    def shutdown(self):
        """Shut down all rollout workers tracked by the controller."""
        self.health_manager.stop()
        try:
            actors = self.registry.all_actors()
            ray.get(
                [actor.shutdown.remote(stop_session_server=True) for actor in actors],  # type: ignore[attr-defined]
                timeout=ROLLOUT_RAY_GET_TIMEOUT,
            )
        finally:
            if self.proxy_manager is not None:
                self.proxy_manager.close()

    def _seal_and_drain_requests(self, settings: CrossWeightUpdateRolloutConfig) -> None:
        workers = self.registry.active_entrypoints()
        servers = ray.get(
            [worker.actor.get_session_server.remote() for worker in workers],  # type: ignore[attr-defined]
            timeout=settings.freeze_timeout,
        )
        if not servers or any(server is None for server in servers):
            raise RuntimeError("Every fixed entrypoint must own a SessionServer")
        state = self._request_state
        state.epoch += 1
        barrier = RequestBarrier(state.epoch, tuple(str(worker.session_url) for worker in workers), tuple(servers))
        state.barrier = barrier
        # TODO(timeout-reconciliation): seal/drain ray.get timeouts abort the run even though the RPCs may still
        # complete, and a seal may already cover only part of the entrypoints. Record per-entrypoint stage state, keep
        # waiting or retry from that state within the same barrier, and reserve hard failure for confirmed
        # unsettled/backend errors. (PR #2297 review.)
        # 所有入口先 seal 再 drain；已准入请求结算完成后，才允许暂停 backend 和切换权重。
        ray.get(
            [server.seal_requests.remote(barrier.epoch) for server in barrier.servers],
            timeout=settings.freeze_timeout,
        )
        try:
            ray.get(
                [server.drain_requests.remote(settings.drain_timeout) for server in barrier.servers],
                timeout=settings.drain_timeout + settings.freeze_timeout,
            )
        except Exception:
            if not self._recover_http_timeouts(barrier, workers, settings):
                raise
            ray.get(
                [server.drain_requests.remote(settings.drain_timeout) for server in barrier.servers],
                timeout=settings.drain_timeout + settings.freeze_timeout,
            )

    def _recover_http_timeouts(
        self, barrier: RequestBarrier, workers: tuple[WorkerSnapshot, ...], settings: CrossWeightUpdateRolloutConfig
    ) -> bool:
        """Reconcile known HTTP failures under the existing fixed-topology barrier."""
        states = ray.get(
            [server.request_status.remote() for server in barrier.servers], timeout=settings.freeze_timeout
        )
        if not any(state["failures"] for state in states) or any(
            not set(state["failures"]).issubset(state.get("known_forward_timeouts", [])) for state in states
        ):
            return False
        if tuple(worker.session_url for worker in self.registry.active_entrypoints()) != barrier.endpoints:
            raise RuntimeError("Fixed rollout topology changed before timeout recovery")
        receipts = ray.get(
            [worker.actor.sleep_for_timeout_recovery.remote(barrier.epoch) for worker in workers],
            timeout=600 + settings.freeze_timeout,
        )
        if tuple(worker.session_url for worker in self.registry.active_entrypoints()) != barrier.endpoints:
            raise RuntimeError("Fixed rollout topology changed during timeout recovery")
        for worker, receipt in zip(workers, receipts):
            if receipt.get("backend_url") != worker.url or receipt.get("epoch") != barrier.epoch:
                raise RuntimeError("Backend timeout recovery receipt identity mismatch")
        ray.get(
            [
                server.settle_after_backend_sleep.remote(barrier.epoch, receipt)
                for server, receipt in zip(barrier.servers, receipts)
            ],
            timeout=60 + settings.freeze_timeout,
        )
        self.logger.warning(f"HTTP timeout recovery completed under request barrier epoch={barrier.epoch}")
        return True

    def _resume_with_request_barrier(self, model_step: int) -> None:
        if model_step is None:
            raise ValueError("Cross-weight-update rollout requires the actual rollout model_step")
        state = self._request_state
        barrier = state.barrier
        if barrier is None:
            # 重入已开放窗口仅允许同版本；更新版本必须先建立暂停屏障。
            if model_step != state.published_model_step:
                raise RuntimeError("A different model version requires a pause barrier")
            return
        endpoints = tuple(worker.session_url for worker in self.registry.active_entrypoints())
        if endpoints != barrier.endpoints:
            raise RuntimeError("Fixed rollout topology changed under barrier")
        settings = cast(CrossWeightUpdateRolloutConfig, self.config.cross_weight_update_rollout)
        # TODO(timeout-reconciliation): freeze_timeout is also the shared install→resume→reopen budget. A timeout may
        # leave some entrypoints installed/reopened and others not, while ray.get timeout never cancels the RPCs.
        # Preserve the barrier and stage, verify each entrypoint before the next stage, and distinguish slow RPCs
        # from confirmed backend/topology failures instead of aborting training on the first timeout. (PR #2297.)
        deadline = monotonic() + settings.freeze_timeout

        def wait_all(submit: Callable[[], list[Any]]) -> list[Any]:
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError("Request publication deadline exceeded")
            return ray.get(submit(), timeout=remaining)

        # install、backend resume、reopen 共用一个预算；任一阶段失败都保留 barrier 并停止训练。
        wait_all(
            lambda: [server.install_request_version.remote(barrier.epoch, model_step) for server in barrier.servers]
        )
        results = self._broadcast_to_workers(
            "continue_generation", WorkerLifecycleState.ACTIVE, timeout=deadline - monotonic()
        )
        if any(result is False for result in results):
            raise RuntimeError("Backend resume failed under request barrier")
        # 多入口 reopen 不是原子操作；部分失败不能声称能够全局回滚。
        wait_all(lambda: [server.reopen_requests.remote(barrier.epoch) for server in barrier.servers])
        state.published_model_step = model_step
        state.barrier = None
        self.health_manager.resume()

    def _broadcast_to_workers(
        self,
        method_name: str,
        target_state: WorkerLifecycleState,
        *,
        timeout: float = ROLLOUT_RAY_GET_TIMEOUT,
        **kwargs: Any,
    ) -> list[Any]:
        # timeout 是控制侧预算，不下传给 worker；预算耗尽后不能再发出下一阶段的 RPC。
        if timeout <= 0:
            raise TimeoutError("Rollout control deadline exceeded")
        workers = self.registry.get_target_state_workers(target_state)
        futures = [getattr(worker.actor, method_name).remote(**kwargs) for worker in workers]
        return ray.get(futures, timeout=timeout)

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
