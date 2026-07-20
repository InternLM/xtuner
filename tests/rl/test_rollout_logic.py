"""Rollout worker / utils 的 PR-fast 逻辑测试。

本文件合并旧的 test_rollout_worker.py 和 test_rollout_utils.py 中不依赖真实模型/后端的测试：
- SGLangWorker pause/continue 对 abort flag 和 server request 的控制。
- RolloutWorker abort、abort request timeout 和 in-flight request 取消语义。
- RolloutHealthManager 对 inactive/unhealthy worker 的生命周期标记逻辑。
- PartialRolloutHandler 拼接 routed_experts 后释放旧 Ray ObjectRef 的逻辑。

旧 test_rollout_utils.py 中的 TestRolloutControllerRecover 需要真实 Ray controller / lmdeploy backend，
不属于 PR-fast，后续应放到 PR-real smoke 或 nightly。
"""

import asyncio
import threading
import unittest
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

import httpx

from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams, Status
from xtuner.v1.rl.agent_loop import AgentLoopConfig
from xtuner.v1.rl.rollout.controller import RolloutController
from xtuner.v1.rl.rollout.health_manager import RolloutHealthManager
from xtuner.v1.rl.rollout.lmdeploy import LMDeployWorker
from xtuner.v1.rl.rollout.rollout_topology import RolloutEngine, RolloutTopology, RolloutServerProcess
from xtuner.v1.rl.rollout.proxy_manager import RolloutProxyManager
from xtuner.v1.rl.rollout.worker_registry import (
    RolloutWorkerRegistry,
    WorkerLifecycleState,
    WorkerSnapshot,
)
from xtuner.v1.rl.rollout.sglang import SGLangWorker
from xtuner.v1.rl.rollout.utils import PartialRolloutHandler, SessionRouter
from xtuner.v1.rl.rollout.worker import RolloutWorker, RolloutWorkerInitResult
from xtuner.v1.rl.utils.misc import delete_from_routedapiproxy
from xtuner.v1.rl.weight_update.data import RolloutWeightUpdateInfo
from xtuner.v1.train.rl_trainer import BaseRLTrainer, _agent_loop_manager_requires_rollout_proxy
from xtuner.v1.utils.httpx_utils import HttpRequestErrorType, HttpRequestResult


class _FakeAsyncRemoteMethod:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def remote(self, *args, **kwargs):
        if kwargs:
            self.calls.append((args, kwargs))
        else:
            self.calls.append(args)

        async def _result():
            if isinstance(self.result, Exception):
                raise self.result
            return self.result

        return _result()


def _register_started_servers(
    registry,
    entries,
    *,
    lifecycle_state=WorkerLifecycleState.ACTIVE,
):
    entries = tuple(entries)
    workers_by_rank = [None] * (max((rank for rank, _actor, _server_url, _session_url in entries), default=-1) + 1)
    init_results = []
    for rank, actor, server_url, session_url in entries:
        workers_by_rank[rank] = actor
        init_results.append(
            RolloutWorkerInitResult(
                rank=rank,
                server_url=server_url,
                session_url=session_url,
            )
        )
    registry.register_started_servers(
        init_results=tuple(init_results),
        workers_by_rank=tuple(workers_by_rank),
        lifecycle_state=lifecycle_state,
    )


class _FakeRolloutRouter:
    def __init__(self, worker):
        self.worker = worker
        self.session_ids = []

    async def get_worker(self, session_id):
        self.session_ids.append(session_id)
        return self.worker


class _FakeRolloutWorkerGenerate:
    def __init__(self, returned_state):
        self.returned_state = returned_state
        self.calls = []

    def remote(self, *, rollout_state):
        self.calls.append(rollout_state)
        result = asyncio.get_running_loop().create_future()
        result.set_result(self.returned_state)
        return result


class _FakeRolloutWorker:
    def __init__(self, returned_state):
        self.generate = _FakeRolloutWorkerGenerate(returned_state)


class _ProxyRequiredAgentLoopConfig(AgentLoopConfig):
    hf_checkpoint: str = "fake"
    requires_rollout_proxy: bool = True

    def build_local(self, rollout_controller, judger=None, logger=None):
        raise NotImplementedError


class _ProxyOptionalAgentLoopConfig(AgentLoopConfig):
    hf_checkpoint: str = "fake"

    def build_local(self, rollout_controller, judger=None, logger=None):
        raise NotImplementedError


class TestAgentLoopProxyRequirement(unittest.TestCase):
    def test_agent_loop_manager_proxy_requirement_uses_config_declaration(self):
        requires_proxy_manager = SimpleNamespace(
            tasks=[
                SimpleNamespace(agent_loop_config=_ProxyOptionalAgentLoopConfig()),
                SimpleNamespace(agent_loop_config=_ProxyRequiredAgentLoopConfig()),
            ]
        )
        optional_proxy_manager = SimpleNamespace(
            tasks=SimpleNamespace(agent_loop_config=_ProxyOptionalAgentLoopConfig())
        )

        self.assertTrue(_agent_loop_manager_requires_rollout_proxy(requires_proxy_manager))
        self.assertFalse(_agent_loop_manager_requires_rollout_proxy(optional_proxy_manager))

    def test_trainer_auto_enables_rollout_proxy_when_agent_loop_requires_it(self):
        trainer = BaseRLTrainer.__new__(BaseRLTrainer)
        trainer._rollout_config = SimpleNamespace(enable_proxy=False)
        trainer.logger = MagicMock()
        cfg = SimpleNamespace(
            agent_loop_manager_cfg=SimpleNamespace(
                tasks=SimpleNamespace(agent_loop_config=_ProxyRequiredAgentLoopConfig())
            ),
            eval_agent_loop_manager_cfg=None,
        )

        trainer._ensure_rollout_proxy_config(cfg)

        self.assertTrue(trainer._rollout_config.enable_proxy)


class TestRolloutTopologyAPI(unittest.TestCase):
    def _rollout_config(
        self,
        *,
        tp: int,
        ep: int,
        num_gpus_per_engine: int,
        gpus_per_node: int = 8,
    ):
        return SimpleNamespace(
            api_key="test-key",
            tensor_parallel_size=tp,
            expert_parallel_size=ep,
            num_gpus_per_engine=num_gpus_per_engine,
            gpus_per_node=gpus_per_node,
            extra_rollout_config={"lmdeploy_backend": "pytorch"},
        )

    def _rank_bundle_idx_list(self, num_workers: int):
        return [(rank, rank) for rank in range(num_workers)]

    def _rank_to_dist_init_addr(self, num_workers: int):
        return {rank: f"host{rank}:25{rank:03d}" for rank in range(num_workers)}

    def _weight_update_targets(self, topology: RolloutTopology):
        registry = RolloutWorkerRegistry(rollout_topology=topology)
        _register_started_servers(
            registry,
            (
                (
                    spec.worker_rank,
                    object(),
                    f"http://worker-{spec.worker_rank}",
                    f"http://session-{spec.worker_rank}",
                )
                for spec in topology.server_launch_specs()
            ),
        )
        return registry.weight_update_targets()

    def _rollout_info(self, *, config, targets, train_rank: int):
        return RolloutWeightUpdateInfo.from_targets(
            rollout_config=config,
            weight_update_targets=targets,
            train_rank=train_rank,
            weight_transport_type="ipc",
        )

    def test_rollout_topology_resolves_engine_dist_init_addr_when_created(self):
        rank_to_dist_init_addr = {0: "host0:25000", 1: "host1:25004"}
        dist_init_addr_owner_rank = 0
        engine = RolloutEngine(
            engine_ranks=(0, 1),
            dist_init_addr=rank_to_dist_init_addr[dist_init_addr_owner_rank],
            server_processes=(
                RolloutServerProcess(
                    worker_rank=0,
                    placement_group_bundle_idxs=(0,),
                    accepts_rollout_requests=True,
                    weight_update_ranks=(0, 1),
                ),
                RolloutServerProcess(
                    worker_rank=1,
                    placement_group_bundle_idxs=(1,),
                    weight_update_ranks=(),
                    accepts_rollout_requests=False,
                ),
            ),
        )

        topology = RolloutTopology(
            engines=(engine,),
        )

        launch_specs = topology.server_launch_specs()
        self.assertEqual(tuple(spec.worker_rank for spec in launch_specs), (0, 1))
        rank_0_launch_spec, rank_1_launch_spec = launch_specs
        self.assertEqual(rank_0_launch_spec.dist_init_addr, "host0:25000")
        self.assertEqual(rank_1_launch_spec.dist_init_addr, "host0:25000")
        self.assertEqual(rank_0_launch_spec.engine_rank, 0)
        self.assertEqual(rank_1_launch_spec.engine_rank, 1)
        self.assertEqual(rank_1_launch_spec.placement_group_bundle_idxs, (1,))
        self.assertTrue(topology.is_request_entrypoint_rank(0))
        self.assertFalse(topology.is_request_entrypoint_rank(1))
        self.assertEqual(topology.lifecycle_group_for_server_rank(1), (0, 1))
        self.assertEqual(
            tuple(
                (server.worker_rank, server.weight_update_ranks)
                for server in topology.weight_update_endpoint_processes()
            ),
            ((0, (0, 1)),),
        )

    def test_lmdeploy_tp16_weight_update_targets_match_legacy_mesh_and_url_semantics(self):
        config = self._rollout_config(tp=16, ep=1, num_gpus_per_engine=16)
        topology = LMDeployWorker.build_rollout_topology(
            config,
            self._rank_bundle_idx_list(16),
            self._rank_to_dist_init_addr(16),
        )
        targets = self._weight_update_targets(topology)

        self.assertEqual(
            tuple((target.endpoint_rank, target.update_ranks) for target in targets),
            ((0, tuple(range(16))),),
        )
        self.assertEqual(self._rollout_info(config=config, targets=targets, train_rank=0).rollout_url, "http://worker-0")
        self.assertIsNone(self._rollout_info(config=config, targets=targets, train_rank=1).rollout_url)
        self.assertEqual(
            self._rollout_info(config=config, targets=targets, train_rank=1).ipc_rank_mesh,
            (tuple(range(16)),),
        )

    def test_lmdeploy_ep16_weight_update_targets_match_legacy_mesh_and_url_semantics(self):
        config = self._rollout_config(tp=1, ep=16, num_gpus_per_engine=16)
        topology = LMDeployWorker.build_rollout_topology(
            config,
            self._rank_bundle_idx_list(16),
            self._rank_to_dist_init_addr(16),
        )
        targets = self._weight_update_targets(topology)

        self.assertEqual(
            tuple((target.endpoint_rank, target.update_ranks) for target in targets),
            tuple((rank, (rank,)) for rank in range(16)),
        )
        self.assertEqual(self._rollout_info(config=config, targets=targets, train_rank=0).rollout_url, "http://worker-0")
        self.assertEqual(
            self._rollout_info(config=config, targets=targets, train_rank=15).rollout_url,
            "http://worker-15",
        )
        self.assertEqual(
            self._rollout_info(config=config, targets=targets, train_rank=0).ipc_rank_mesh,
            tuple((rank,) for rank in range(16)),
        )

    def test_sglang_tp16_cross_node_weight_update_targets_match_legacy_mesh_and_url_semantics(self):
        config = self._rollout_config(tp=16, ep=1, num_gpus_per_engine=16, gpus_per_node=8)
        topology = SGLangWorker.build_rollout_topology(
            config,
            self._rank_bundle_idx_list(16),
            self._rank_to_dist_init_addr(16),
        )
        targets = self._weight_update_targets(topology)

        self.assertEqual(tuple(spec.worker_rank for spec in topology.server_launch_specs()), (0, 8))
        self.assertEqual(
            tuple((target.endpoint_rank, target.update_ranks) for target in targets),
            ((0, tuple(range(16))),),
        )
        self.assertEqual(self._rollout_info(config=config, targets=targets, train_rank=0).rollout_url, "http://worker-0")
        self.assertIsNone(self._rollout_info(config=config, targets=targets, train_rank=8).rollout_url)
        self.assertEqual(
            self._rollout_info(config=config, targets=targets, train_rank=8).ipc_rank_mesh,
            (tuple(range(16)),),
        )


class TestRolloutController(unittest.IsolatedAsyncioTestCase):
    def _state(self, uid: int, session_id: int) -> RolloutState:
        return RolloutState(
            rollout_id=uid,
            group_id=uid,
            session_id=session_id,
            message=[{"role": "user", "content": f"prompt {uid}"}],
            prompt_ids=[uid],
            status=Status.INIT,
            extra_fields={},
        )

    def _build_controller(self, router):
        controller = RolloutController.__new__(RolloutController)
        controller.config = SimpleNamespace(rollout_timeout=1.0, random_seed=0)
        controller.timeout_multiplier = 1.0
        controller.router = router
        controller.logger = MagicMock()
        return controller

    def _build_registry(self, ranks):
        rollout_topology = RolloutTopology(
            engines=tuple(
                RolloutEngine(
                    engine_ranks=(rank,),
                    dist_init_addr=f"addr{rank}",
                    server_processes=(
                        RolloutServerProcess(
                            worker_rank=rank,
                            placement_group_bundle_idxs=(rank,),
                            weight_update_ranks=(rank,),
                        ),
                    ),
                )
                for rank in ranks
            ),
        )
        return RolloutWorkerRegistry(
            rollout_topology=rollout_topology,
        )

    async def test_generate_fails_fast_when_no_active_worker(self):
        # router 找不到 active worker 时，controller 应直接把原样本标成 FAILED，避免请求悬挂。
        state = self._state(uid=1, session_id=123)
        router = _FakeRolloutRouter(worker=None)
        controller = self._build_controller(router)

        with patch("xtuner.v1.rl.rollout.controller.XTUNER_DETERMINISTIC", False):
            result = await controller.generate(state)

        self.assertIs(result, state)
        self.assertEqual(router.session_ids, [123])
        self.assertEqual(result.status, Status.FAILED)
        self.assertEqual(result.error_msg, "No active rollout worker available.")

    async def test_generate_routes_to_active_worker(self):
        # 有 active worker 时，controller 要按 session_id 路由，并返回 worker 的 rollout 结果。
        request_state = self._state(uid=1, session_id=456)
        returned_state = self._state(uid=1, session_id=456)
        returned_state.status = Status.COMPLETED
        worker = _FakeRolloutWorker(returned_state)
        router = _FakeRolloutRouter(worker=worker)
        controller = self._build_controller(router)

        with patch("xtuner.v1.rl.rollout.controller.XTUNER_DETERMINISTIC", False):
            result = await controller.generate(request_state)

        self.assertIs(result, returned_state)
        self.assertEqual(router.session_ids, [456])
        self.assertEqual(worker.generate.calls, [request_state])
        self.assertEqual(result.status, Status.COMPLETED)

    def test_register_active_workers_to_proxy_delegates_active_session_urls(self):
        controller = RolloutController.__new__(RolloutController)
        controller.registry = self._build_registry((0, 1))
        _register_started_servers(
            controller.registry,
            (
                (0, object(), "http://worker-0", "http://session-0"),
                (1, object(), "http://worker-1", "http://session-1"),
            ),
        )
        controller.registry.mark_unhealthy_ranks({1})
        controller.proxy_manager = MagicMock()

        controller.register_active_workers_to_proxy()

        controller.proxy_manager.replace_registered_session_urls.assert_called_once_with(["http://session-0"])

    def test_register_active_workers_to_proxy_noops_without_proxy_manager(self):
        controller = RolloutController.__new__(RolloutController)
        controller.registry = MagicMock()
        controller.proxy_manager = None

        controller.register_active_workers_to_proxy()

    def test_validate_registered_workers_to_proxy_delegates_proxy_validation(self):
        controller = RolloutController.__new__(RolloutController)
        controller.proxy_manager = MagicMock()

        controller.validate_registered_workers_to_proxy()

        controller.proxy_manager.validate_registered_session_urls.assert_called_once_with()


class TestRolloutProxyManager(unittest.TestCase):
    _ROUTED_PROXY_URL = "http://routed-proxy"
    _ROUTED_PROXY_ADMIN_URL = "http://routed-proxy-admin"

    def _build_manager(self):
        config = SimpleNamespace(
            model_name="test-model",
            routed_proxy_url=self._ROUTED_PROXY_URL,
            routed_proxy_admin_url=self._ROUTED_PROXY_ADMIN_URL,
            worker_log_dir=None,
        )
        return RolloutProxyManager(config)

    def test_replace_registered_session_urls_replaces_proxy_registrations(self):
        manager = self._build_manager()

        with (
            patch("xtuner.v1.rl.rollout.proxy_manager.delete_from_routedapiproxy") as delete_proxy,
            patch("xtuner.v1.rl.rollout.proxy_manager.register_to_routedapiproxy") as register_proxy,
            patch("xtuner.v1.rl.rollout.proxy_manager.check_chat_completions") as check_chat_completions,
        ):
            manager.replace_registered_session_urls(["http://session-1", "http://session-0", "http://session-1"])

        delete_proxy.assert_called_once_with(self._ROUTED_PROXY_ADMIN_URL, "test-model")
        check_chat_completions.assert_not_called()
        self.assertEqual(
            register_proxy.call_args_list,
            [
                call(self._ROUTED_PROXY_ADMIN_URL, "test-model", "http://session-0"),
                call(self._ROUTED_PROXY_ADMIN_URL, "test-model", "http://session-1"),
            ],
        )

    def test_validate_registered_session_urls_checks_routed_proxy_url(self):
        manager = self._build_manager()
        manager._registered_session_urls = {"http://session-0"}

        with (
            patch("xtuner.v1.rl.rollout.proxy_manager.check_chat_completions", return_value=True) as check_chat,
            patch("xtuner.v1.rl.rollout.proxy_manager.time.sleep"),
        ):
            manager.validate_registered_session_urls()

        check_chat.assert_called_once_with(self._ROUTED_PROXY_URL, "test-model")

    def test_validate_registered_session_urls_raises_when_proxy_validation_fails(self):
        manager = self._build_manager()
        manager._registered_session_urls = {"http://session-0"}

        with (
            patch("xtuner.v1.rl.rollout.proxy_manager.check_chat_completions", return_value=False) as check_chat,
            patch("xtuner.v1.rl.rollout.proxy_manager.time.sleep"),
            self.assertRaisesRegex(RuntimeError, "routed API proxy"),
        ):
            manager.validate_registered_session_urls()

        check_chat.assert_has_calls([call(self._ROUTED_PROXY_URL, "test-model")] * manager._CHECK_ATTEMPTS)

    def test_validate_registered_session_urls_noops_without_registrations(self):
        manager = self._build_manager()

        with patch("xtuner.v1.rl.rollout.proxy_manager.check_chat_completions") as check_chat:
            manager.validate_registered_session_urls()

        check_chat.assert_not_called()

    def test_delete_and_register_session_url_use_single_url_payload(self):
        manager = self._build_manager()
        manager._registered_session_urls = {"http://session-0"}

        with (
            patch("xtuner.v1.rl.rollout.proxy_manager.delete_from_routedapiproxy") as delete_proxy,
            patch("xtuner.v1.rl.rollout.proxy_manager.register_to_routedapiproxy") as register_proxy,
            patch("xtuner.v1.rl.rollout.proxy_manager.check_chat_completions", return_value=True),
        ):
            manager._delete_session_url("http://session-0")
            manager._register_session_url("http://session-0")

        delete_proxy.assert_called_once_with(self._ROUTED_PROXY_ADMIN_URL, "test-model", "http://session-0")
        register_proxy.assert_called_once_with(self._ROUTED_PROXY_ADMIN_URL, "test-model", "http://session-0")

    def test_inactive_lifecycle_listener_deletes_entrypoint_session_urls(self):
        manager = self._build_manager()
        manager._delete_session_url = MagicMock()
        manager._register_session_url = MagicMock()
        entrypoint = WorkerSnapshot(
            rank=0,
            actor=object(),
            url="http://worker-0",
            session_url="http://session-0",
            is_request_entrypoint=True,
        )
        non_entrypoint = WorkerSnapshot(
            rank=1,
            actor=object(),
            url="http://worker-1",
            session_url="http://session-1",
            is_request_entrypoint=False,
        )
        worker_group = SimpleNamespace(workers=(entrypoint, non_entrypoint))

        manager.on_worker_group_inactive(worker_group)

        manager._delete_session_url.assert_called_once_with("http://session-0")
        manager._register_session_url.assert_not_called()

    def test_recovered_lifecycle_listener_registers_entrypoint_session_urls_without_validation(self):
        manager = self._build_manager()
        manager._register_session_url = MagicMock()
        worker_group = SimpleNamespace(
            workers=(
                WorkerSnapshot(
                    rank=0,
                    actor=object(),
                    url="http://worker-0",
                    session_url="http://session-0",
                    is_request_entrypoint=True,
                ),
            )
        )

        manager.on_worker_group_recovered(worker_group)

        manager._register_session_url.assert_called_once_with("http://session-0")


class TestRoutedApiProxyUtils(unittest.TestCase):
    def test_delete_from_routedapiproxy_includes_api_base_when_provided(self):
        response = MagicMock()
        response.json.return_value = {"ok": True}

        with patch("xtuner.v1.rl.utils.misc.requests.post", return_value=response) as post:
            delete_from_routedapiproxy("http://proxy-admin", "test-model", "http://session-0")

        post.assert_called_once()
        self.assertEqual(post.call_args.args[0], "http://proxy-admin/v1/models/delete")
        self.assertEqual(
            post.call_args.kwargs["json"],
            {"model_name": "test-model", "api_base": "http://session-0"},
        )
        response.raise_for_status.assert_called_once_with()


class TestRolloutWorkerRegistry(unittest.TestCase):
    def _worker_by_rank(self, registry, rank):
        return next(worker for worker in registry.all_workers() if worker.rank == rank)

    def _runtime_layout(
        self,
        *,
        engine_ranks=(0,),
        server_processes=None,
    ):
        if server_processes is None:
            server_processes = (
                RolloutServerProcess(
                    worker_rank=engine_ranks[0],
                    placement_group_bundle_idxs=tuple(range(len(engine_ranks))),
                    accepts_rollout_requests=True,
                    weight_update_ranks=tuple(engine_ranks),
                ),
            )
        dist_init_addr_owner_rank = server_processes[0].worker_rank
        return RolloutTopology(
            engines=(
                RolloutEngine(
                    engine_ranks=tuple(engine_ranks),
                    dist_init_addr=f"addr{dist_init_addr_owner_rank}",
                    server_processes=tuple(server_processes),
                ),
            ),
        )

    def test_registry_filters_entrypoints_and_tracks_lifecycle(self):
        runtime_layout = self._runtime_layout(
            engine_ranks=(0, 1),
            server_processes=(
                RolloutServerProcess(
                    worker_rank=0,
                    placement_group_bundle_idxs=(0,),
                    accepts_rollout_requests=True,
                    weight_update_ranks=(0, 1),
                ),
                RolloutServerProcess(
                    worker_rank=1,
                    placement_group_bundle_idxs=(1,),
                    weight_update_ranks=(),
                    accepts_rollout_requests=False,
                ),
            ),
        )
        registry = RolloutWorkerRegistry(rollout_topology=runtime_layout)
        _register_started_servers(
            registry,
            (
                (0, object(), "http://worker-0", "http://session-0"),
                (1, object(), "http://worker-1", None),
            ),
        )

        active_entrypoint = registry.active_entrypoints()[0]
        self.assertIsInstance(active_entrypoint, WorkerSnapshot)
        self.assertEqual(active_entrypoint.rank, 0)
        with self.assertRaises(FrozenInstanceError):
            active_entrypoint.lifecycle_state = WorkerLifecycleState.INACTIVE

        unhealthy_groups = registry.mark_unhealthy_ranks({0})

        self.assertEqual(unhealthy_groups[0].ranks, (0, 1))
        self.assertEqual(tuple(worker.rank for worker in registry.inactive_workers()), (0, 1))
        self.assertEqual(registry.active_entrypoints(), ())
        inactive_groups = registry.inactive_worker_groups()
        self.assertEqual(inactive_groups[0].ranks, (0, 1))
        self.assertEqual(self._worker_by_rank(registry, 0).lifecycle_state, WorkerLifecycleState.INACTIVE)
        claimed_groups = registry.claim_inactive_groups_for_recovery()
        self.assertEqual(claimed_groups[0].ranks, (0, 1))
        self.assertEqual(self._worker_by_rank(registry, 0).lifecycle_state, WorkerLifecycleState.RECOVERING)
        registry.set_group_recovery_result(claimed_groups[0], recovered=False)
        self.assertEqual(self._worker_by_rank(registry, 0).lifecycle_state, WorkerLifecycleState.INACTIVE)

    def test_registry_projects_weight_update_targets_from_topology_and_runtime_state(self):
        runtime_layout = self._runtime_layout(engine_ranks=(0, 1))
        registry = RolloutWorkerRegistry(rollout_topology=runtime_layout)
        _register_started_servers(
            registry,
            ((0, object(), "http://worker-0", "http://session-0"),),
        )

        targets = registry.weight_update_targets()

        self.assertEqual(len(targets), 1)
        target = targets[0]
        self.assertEqual(target.endpoint_rank, 0)
        self.assertEqual(target.update_ranks, (0, 1))
        self.assertEqual(target.engine_size, 2)
        self.assertEqual(target.server_url, "http://worker-0")
        self.assertEqual(target.lifecycle_state, WorkerLifecycleState.ACTIVE.value)
        self.assertTrue(target.is_active)

class TestSessionRouter(unittest.IsolatedAsyncioTestCase):
    async def test_sticky_session_reselects_when_previous_entrypoint_is_inactive(self):
        actor_0 = object()
        actor_1 = object()
        rollout_topology = RolloutTopology(
            engines=(
                RolloutEngine(
                    engine_ranks=(0,),
                    dist_init_addr="addr0",
                    server_processes=(
                        RolloutServerProcess(
                            worker_rank=0,
                            placement_group_bundle_idxs=(0,),
                            weight_update_ranks=(0,),
                        ),
                    ),
                ),
                RolloutEngine(
                    engine_ranks=(1,),
                    dist_init_addr="addr1",
                    server_processes=(
                        RolloutServerProcess(
                            worker_rank=1,
                            placement_group_bundle_idxs=(1,),
                            weight_update_ranks=(1,),
                        ),
                    ),
                ),
            ),
        )
        registry = RolloutWorkerRegistry(
            rollout_topology=rollout_topology,
        )
        _register_started_servers(
            registry,
            (
                (0, actor_0, "http://worker-0", "http://session-0"),
                (1, actor_1, "http://worker-1", "http://session-1"),
            ),
        )
        router = SessionRouter(registry, max_idle_seconds=None)

        self.assertIs(await router.get_worker(7), actor_0)
        self.assertIs(await router.get_worker(7), actor_0)

        registry.mark_unhealthy_ranks({0})

        self.assertIs(await router.get_worker(7), actor_1)


class TestSGLangWorker(unittest.TestCase):
    def test_pause_generation_sets_abort_flag_before_server_pause(self):
        # pause_generation 要先设置本地 abort flag，再通知服务端 pause，避免继续接收新生成结果。
        worker = SGLangWorker.__new__(SGLangWorker)
        worker.receive_abort_request = threading.Event()
        worker._send_abort_request = AsyncMock(return_value=True)
        worker._make_request = MagicMock(return_value="ok")

        result = asyncio.run(worker.pause_generation())

        self.assertEqual(result, "ok")
        self.assertTrue(worker.receive_abort_request.is_set())
        worker._send_abort_request.assert_awaited_once_with()
        worker._make_request.assert_called_once_with("pause_generation", {"mode": "abort"})

    def test_continue_generation_clears_abort_flag(self):
        # continue_generation 恢复 rollout 前要清掉 abort flag。
        worker = SGLangWorker.__new__(SGLangWorker)
        worker.receive_abort_request = threading.Event()
        worker.receive_abort_request.set()
        worker._make_request = MagicMock(return_value="ok")

        result = worker.continue_generation()

        self.assertEqual(result, "ok")
        self.assertFalse(worker.receive_abort_request.is_set())
        worker._make_request.assert_called_once_with("continue_generation")


class TestRolloutWorker(unittest.IsolatedAsyncioTestCase):
    def _build_partial_rollout_worker(self, *, eos_token: list[int] | None = None):
        worker = RolloutWorker.__new__(RolloutWorker)
        worker.receive_abort_request = threading.Event()
        worker.enable_partial_rollout = True
        worker.partial_rollout_handler = PartialRolloutHandler()
        worker.server_url = "http://test"
        worker.endpoints = {"generate": "generate", "v1/chat/completions": "v1/chat/completions"}
        worker.config = SimpleNamespace(api_key="test", max_retry_per_sample=0)
        worker.eos_token = eos_token or [999]
        worker.logger = MagicMock()
        worker._get_request_payload = MagicMock(
            side_effect=lambda rollout_state: {
                "input_ids": rollout_state.tokens,
                "max_tokens": rollout_state.sample_params.max_tokens,
            }
        )
        worker._safe_post_request = AsyncMock()
        worker._safe_handle_response = AsyncMock()
        return worker

    def _build_mock_error_rollout_worker(
        self,
        *,
        safe_post_result: HttpRequestResult,
        safe_handle_response=None,
    ):
        worker = RolloutWorker.__new__(RolloutWorker)
        worker.receive_abort_request = threading.Event()
        worker.enable_partial_rollout = False
        worker.server_url = "http://test"
        worker.endpoints = {"generate": "generate", "v1/chat/completions": "v1/chat/completions"}
        worker.config = SimpleNamespace(api_key="test", max_retry_per_sample=3)
        worker.eos_token = [999]
        worker.logger = MagicMock()
        worker._get_request_payload = MagicMock(return_value={"input_ids": [1], "max_tokens": 128})
        worker._safe_post_request = AsyncMock(return_value=safe_post_result)
        worker._safe_handle_response = AsyncMock(side_effect=safe_handle_response)
        return worker

    async def test_generate_returns_aborted_when_abort_flag_is_set(self):
        # worker 已经收到 abort 时，generate 应直接返回 ABORTED 状态，不再请求后端。
        worker = RolloutWorker.__new__(RolloutWorker)
        worker.receive_abort_request = threading.Event()
        worker.receive_abort_request.set()
        rollout_state = MagicMock()

        result = await worker.generate(rollout_state)

        self.assertIs(result, rollout_state)
        self.assertEqual(rollout_state.finish_reason, "abort")
        self.assertEqual(rollout_state.status, Status.ABORTED)

    async def test_pause_generation_sets_abort_flag(self):
        # pause_generation 设置 abort flag，并向后端发送 abort request。
        worker = RolloutWorker.__new__(RolloutWorker)
        worker.receive_abort_request = threading.Event()
        worker._send_abort_request = AsyncMock(return_value=True)

        result = await worker.pause_generation()

        self.assertTrue(result)
        self.assertTrue(worker.receive_abort_request.is_set())
        worker._send_abort_request.assert_awaited_once_with()

    def test_init_binds_launch_spec_and_skips_session_server_for_non_entrypoint(self):
        topology = RolloutTopology(
            engines=(
                RolloutEngine(
                    engine_ranks=(0, 1),
                    dist_init_addr="host0:25000",
                    server_processes=(
                        RolloutServerProcess(
                            worker_rank=0,
                            placement_group_bundle_idxs=(0,),
                            accepts_rollout_requests=True,
                            weight_update_ranks=(0, 1),
                        ),
                        RolloutServerProcess(
                            worker_rank=1,
                            placement_group_bundle_idxs=(1,),
                            accepts_rollout_requests=False,
                            weight_update_ranks=(),
                        ),
                    ),
                ),
            ),
        )
        launch_spec_by_rank = {spec.worker_rank: spec for spec in topology.server_launch_specs()}
        worker = RolloutWorker.__new__(RolloutWorker)
        worker.rank = 1
        worker.server_launch_spec = None
        worker.receive_abort_request = threading.Event()
        worker.receive_abort_request.set()
        worker.server_url = "http://worker-1"
        worker.session_server_url = None
        worker._launch_server = MagicMock()
        worker.session_server_actor = None

        result = worker.init(launch_spec_by_rank[1])

        worker._launch_server.assert_called_once_with()
        self.assertIsNone(worker.session_server_actor)
        self.assertFalse(worker.receive_abort_request.is_set())
        self.assertIs(worker.server_launch_spec, launch_spec_by_rank[1])
        self.assertEqual(result.rank, 1)
        self.assertEqual(result.server_url, "http://worker-1")
        self.assertIsNone(result.session_url)

    def test_reinit_reuses_bound_launch_spec(self):
        topology = RolloutTopology(
            engines=(
                RolloutEngine(
                    engine_ranks=(0,),
                    dist_init_addr="host0:25000",
                    server_processes=(
                        RolloutServerProcess(
                            worker_rank=0,
                            placement_group_bundle_idxs=(0,),
                            accepts_rollout_requests=True,
                            weight_update_ranks=(0,),
                        ),
                    ),
                ),
            ),
        )
        launch_spec = topology.server_launch_specs()[0]
        worker = RolloutWorker.__new__(RolloutWorker)
        worker.rank = 0
        worker.server_launch_spec = launch_spec
        worker.receive_abort_request = threading.Event()
        worker.server_url = "http://worker-0"
        worker.session_server_url = None
        worker._launch_server = MagicMock()

        def start_session_server():
            worker.session_server_url = "http://session-0"

        worker._start_session_server = MagicMock(side_effect=start_session_server)

        result = worker.reinit()

        worker._launch_server.assert_called_once_with()
        worker._start_session_server.assert_called_once_with()
        self.assertIs(worker.server_launch_spec, launch_spec)
        self.assertEqual(result.rank, 0)
        self.assertEqual(result.server_url, "http://worker-0")
        self.assertEqual(result.session_url, "http://session-0")

    async def test_safe_post_request_returns_aborted_without_sending_when_abort_flag_is_set(self):
        # safe post 在发送前发现 abort flag 时，应直接返回 REQUEST_ABORTED，不再发 HTTP 请求。
        worker = RolloutWorker.__new__(RolloutWorker)
        worker.receive_abort_request = threading.Event()
        worker.receive_abort_request.set()
        worker.logger = MagicMock()
        worker.client = MagicMock()

        result = await worker._safe_post_request("http://test", headers={}, payload={"input_ids": [1]})

        self.assertEqual(result.error_type, HttpRequestErrorType.REQUEST_ABORTED)
        worker.client.build_request.assert_not_called()
        worker.client.send.assert_not_called()

    async def test_partial_rollout_eos_response_completes_without_backend_request(self):
        # partial rollout 已以 EOS 结束时，应直接完成，不再请求推理后端。
        worker = self._build_partial_rollout_worker(eos_token=[999])
        rollout_state = RolloutState(
            rollout_id=1,
            message=[],
            prompt_ids=[10, 11],
            response_ids=[101, 999],
            sample_params=SampleParams(max_tokens=8, return_token_ids=True),
            status=Status.ABORTED,
        )

        result = await worker.generate(rollout_state)

        self.assertIs(result, rollout_state)
        self.assertEqual(result.tokens, [10, 11, 101, 999])
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.status, Status.COMPLETED)
        self.assertEqual(result.response_ids, [101, 999])
        worker._safe_post_request.assert_not_awaited()

    async def test_partial_rollout_max_tokens_exhausted_completes_without_backend_request(self):
        # partial rollout 已用完 max_tokens 时，应直接 length 完成，不再继续生成。
        worker = self._build_partial_rollout_worker()
        rollout_state = RolloutState(
            rollout_id=2,
            message=[],
            prompt_ids=[10, 11],
            response_ids=[201, 202, 203],
            sample_params=SampleParams(max_tokens=3, return_token_ids=True),
            status=Status.ABORTED,
        )

        result = await worker.generate(rollout_state)

        self.assertIs(result, rollout_state)
        self.assertEqual(result.tokens, [10, 11, 201, 202, 203])
        self.assertEqual(result.sample_params.max_tokens, 0)
        self.assertEqual(result.finish_reason, "length")
        self.assertEqual(result.status, Status.COMPLETED)
        self.assertEqual(result.response_ids, [201, 202, 203])
        worker._safe_post_request.assert_not_awaited()

    async def test_parallel_mock_rollout_errors_return_failed_status_and_messages(self):
        # 保留旧 test_mock_rollout.py 的 5 类错误语义，但去掉 Ray actor / placement group / tokenizer 依赖。
        def request_error_result():
            req = httpx.Request("POST", "http://test/generate")
            error = httpx.RequestError("Mocked httpx request error", request=req)
            return HttpRequestResult(
                error_type=HttpRequestErrorType.from_exception(error),
                exception=error,
                url="http://test/generate",
                payload={"input_ids": [1]},
            )

        def timeout_result():
            error = httpx.TimeoutException("Mocked timeout error")
            return HttpRequestResult(
                error_type=HttpRequestErrorType.from_exception(error),
                exception=error,
                url="http://test/generate",
                payload={"input_ids": [1]},
            )

        def client_error_result():
            req = httpx.Request("POST", "http://test/generate")
            response = httpx.Response(400, request=req)
            error = httpx.HTTPStatusError("Mocked client error", request=req, response=response)
            return HttpRequestResult(
                error_type=HttpRequestErrorType.from_exception(error),
                exception=error,
                url="http://test/generate",
                payload={"input_ids": [1]},
            )

        def server_error_result():
            req = httpx.Request("POST", "http://test/generate")
            response = httpx.Response(500, request=req)
            error = httpx.HTTPStatusError("Mocked server error", request=req, response=response)
            return HttpRequestResult(
                error_type=HttpRequestErrorType.from_exception(error),
                exception=error,
                url="http://test/generate",
                payload={"input_ids": [1]},
            )

        async def invalid_response(rollout_state, http_response):
            rollout_state.status = Status.FAILED
            return rollout_state

        cases = [
            ("timeout", timeout_result(), None, ("Request failed", "3")),
            ("request_error", request_error_result(), None, ("Request failed", "3")),
            ("client_error", client_error_result(), None, ("Client error",)),
            ("server_error", server_error_result(), None, ("Server error",)),
            (
                "invalid_response",
                HttpRequestResult(response=object()),
                invalid_response,
                ("Invalid rollout response", "3"),
            ),
        ]

        async def run_case(case_name, safe_post_result, safe_handle_response, expected_messages):
            worker = self._build_mock_error_rollout_worker(
                safe_post_result=safe_post_result,
                safe_handle_response=safe_handle_response,
            )
            result_state = await worker.generate(RolloutState(message=[{"role": "user", "content": "Hello!"}]))
            self.assertEqual(
                result_state.status,
                Status.FAILED,
                f"Expected rollout to fail due to {case_name}, but it succeeded.",
            )
            self.assertIsNotNone(
                result_state.error_msg,
                f"Expected an error message for {case_name} case, but got None.",
            )
            for expected in expected_messages:
                self.assertIn(
                    expected,
                    result_state.error_msg,
                    f"Expected error message to include {expected!r} for {case_name}, got: {result_state.error_msg}",
                )

        with patch("xtuner.v1.rl.rollout.worker.asyncio.sleep", new=AsyncMock()):
            await asyncio.gather(*(run_case(*case) for case in cases))


class TestRolloutHealthManager(unittest.TestCase):
    def _worker_by_rank(self, registry, rank):
        return next(worker for worker in registry.all_workers() if worker.rank == rank)

    def _build_registry(self, workers_info):
        engines = []
        for rank in sorted(workers_info):
            engines.append(
                RolloutEngine(
                    engine_ranks=(rank,),
                    dist_init_addr=f"addr{rank}",
                    server_processes=(
                        RolloutServerProcess(
                            worker_rank=rank,
                            placement_group_bundle_idxs=(rank,),
                            accepts_rollout_requests=True,
                            weight_update_ranks=(rank,),
                        ),
                    ),
                )
            )
        rollout_topology = RolloutTopology(
            engines=tuple(engines),
        )
        registry = RolloutWorkerRegistry(
            rollout_topology=rollout_topology,
        )
        _register_started_servers(
            registry,
            (
                (
                    rank,
                    worker_info.actor,
                    worker_info.url,
                    worker_info.session_url or f"http://session-{rank}",
                )
                for rank, worker_info in workers_info.items()
            ),
        )
        for rank, worker_info in workers_info.items():
            if worker_info.lifecycle_state is WorkerLifecycleState.INACTIVE:
                registry.mark_unhealthy_ranks({rank})
        return registry

    def _build_manager(
        self,
        workers_info,
        *,
        failure_threshold=1,
        check_timeout=7.0,
        worker_lifecycle_listeners=None,
    ):
        config = SimpleNamespace(
            health_check_interval_seconds=10,
            health_check_timeout_seconds=check_timeout,
            health_check_failure_threshold=failure_threshold,
        )
        registry = self._build_registry(workers_info)
        return (
            RolloutHealthManager(
                config,
                registry,
                worker_lifecycle_listeners=worker_lifecycle_listeners,
            ),
            registry,
        )

    def test_marks_worker_inactive_after_consecutive_health_failures(self):
        actor = SimpleNamespace(check_health=_FakeAsyncRemoteMethod(False))
        worker_info = WorkerSnapshot(rank=0, actor=actor, url="http://worker-0")
        workers_info = {0: worker_info}
        inactive_groups = []
        listener = SimpleNamespace(
            on_worker_group_inactive=inactive_groups.append,
            on_worker_group_recovered=MagicMock(),
        )
        manager, registry = self._build_manager(
            workers_info,
            failure_threshold=2,
            worker_lifecycle_listeners=[listener],
        )

        manager.run_once()

        self.assertTrue(self._worker_by_rank(registry, 0).is_active())
        self.assertEqual(actor.check_health.calls, [()])
        self.assertEqual(inactive_groups, [])

        manager.run_once()

        self.assertFalse(self._worker_by_rank(registry, 0).is_active())
        self.assertEqual(self._worker_by_rank(registry, 0).lifecycle_state, WorkerLifecycleState.INACTIVE)
        self.assertEqual(actor.check_health.calls, [(), ()])
        self.assertEqual([group.ranks for group in inactive_groups], [(0,)])

    def test_inactive_listener_runs_outside_lifecycle_operation_lock(self):
        actor = SimpleNamespace(check_health=_FakeAsyncRemoteMethod(False))
        worker_info = WorkerSnapshot(rank=0, actor=actor, url="http://worker-0")
        lock_acquired_by_listener = []
        manager, _ = self._build_manager({0: worker_info}, failure_threshold=1)

        def on_worker_group_inactive(group):
            acquired = manager._lifecycle_operation_lock.acquire(blocking=False)
            lock_acquired_by_listener.append(acquired)
            if acquired:
                manager._lifecycle_operation_lock.release()

        manager._worker_lifecycle_listeners = (
            SimpleNamespace(
                on_worker_group_inactive=on_worker_group_inactive,
                on_worker_group_recovered=MagicMock(),
            ),
        )

        manager.run_once()

        self.assertEqual(lock_acquired_by_listener, [True])

    def test_inactive_worker_is_not_cleaned_up_again(self):
        # 已 inactive 的 worker 不再重复健康检查，也不再重复触发 inactive 通知。
        actor = SimpleNamespace(check_health=_FakeAsyncRemoteMethod(True))
        workers_info = {
            0: WorkerSnapshot(
                rank=0,
                actor=actor,
                url="http://worker-0",
                lifecycle_state=WorkerLifecycleState.INACTIVE,
            )
        }
        inactive_groups = []
        listener = SimpleNamespace(
            on_worker_group_inactive=inactive_groups.append,
            on_worker_group_recovered=MagicMock(),
        )
        manager, _ = self._build_manager(workers_info, worker_lifecycle_listeners=[listener])

        manager.run_once()

        self.assertEqual(actor.check_health.calls, [])
        self.assertEqual(inactive_groups, [])

    def test_health_check_threshold_zero_disables_periodic_health_check(self):
        # threshold <= 0 表示关闭周期健康监测，不应把 active worker 直接判 inactive。
        actor = SimpleNamespace(check_health=_FakeAsyncRemoteMethod(False))
        worker_info = WorkerSnapshot(rank=0, actor=actor, url="http://worker-0")
        manager, registry = self._build_manager({0: worker_info}, failure_threshold=0)

        with patch("xtuner.v1.rl.rollout.health_manager.threading.Thread") as thread_cls:
            manager.start()

        thread_cls.assert_not_called()
        self.assertIsNone(manager._thread)
        self.assertTrue(self._worker_by_rank(registry, 0).is_active())
        self.assertEqual(actor.check_health.calls, [])

        manager.run_once()

        self.assertTrue(self._worker_by_rank(registry, 0).is_active())
        self.assertEqual(actor.check_health.calls, [])

    def test_run_once_does_not_log_error_when_no_active_workers(self):
        actor = SimpleNamespace(check_health=_FakeAsyncRemoteMethod(True))
        workers_info = {
            0: WorkerSnapshot(
                rank=0,
                actor=actor,
                url="http://worker-0",
                lifecycle_state=WorkerLifecycleState.INACTIVE,
            )
        }
        manager, _ = self._build_manager(workers_info)

        with patch("xtuner.v1.rl.rollout.health_manager.logger.error") as log_error:
            manager.run_once()

        log_error.assert_not_called()
        self.assertEqual(actor.check_health.calls, [])

    def test_run_once_does_not_log_error_when_last_active_worker_becomes_inactive(self):
        actor = SimpleNamespace(check_health=_FakeAsyncRemoteMethod(False))
        worker_info = WorkerSnapshot(rank=0, actor=actor, url="http://worker-0")
        manager, registry = self._build_manager({0: worker_info}, failure_threshold=1)

        with patch("xtuner.v1.rl.rollout.health_manager.logger.error") as log_error:
            manager.run_once()

        log_error.assert_not_called()
        self.assertFalse(self._worker_by_rank(registry, 0).is_active())
        self.assertEqual(actor.check_health.calls, [()])

    def test_fail_fast_health_check_still_runs_when_periodic_health_check_is_disabled(self):
        actor = SimpleNamespace(check_health=_FakeAsyncRemoteMethod(False))
        worker_info = WorkerSnapshot(rank=0, actor=actor, url="http://worker-0")
        manager, registry = self._build_manager({0: worker_info}, failure_threshold=0)

        with patch.object(manager, "_shutdown_worker_group", return_value=True):
            manager.check_and_shutdown_inactive_workers()

        self.assertFalse(self._worker_by_rank(registry, 0).is_active())
        self.assertEqual(actor.check_health.calls, [()])

    def test_health_check_uses_configured_timeout(self):
        actor = SimpleNamespace(check_health=_FakeAsyncRemoteMethod(True))
        worker_info = WorkerSnapshot(rank=0, actor=actor, url="http://worker-0")
        manager, _ = self._build_manager({0: worker_info}, check_timeout=2.5)
        observed_timeouts = []

        async def fake_wait_for(awaitable, timeout):
            observed_timeouts.append(timeout)
            return await awaitable

        with patch("xtuner.v1.rl.rollout.health_manager.asyncio.wait_for", side_effect=fake_wait_for):
            manager.run_once()

        self.assertEqual(observed_timeouts, [2.5])

    def test_wait_until_next_check_waits_for_resume_when_paused_during_interval(self):
        actor = SimpleNamespace(check_health=_FakeAsyncRemoteMethod(True))
        worker_info = WorkerSnapshot(rank=0, actor=actor, url="http://worker-0")
        manager, _ = self._build_manager({0: worker_info})
        manager._check_interval = 0.01
        manager._pause_event = threading.Event()

        class _FakeStopEvent:
            def __init__(self):
                self.wait_calls = []
                self._paused_once = False

            def is_set(self):
                return False

            def wait(self, timeout=None):
                self.wait_calls.append(timeout)
                if timeout == manager._check_interval and not self._paused_once:
                    self._paused_once = True
                    manager._pause_event.set()
                elif timeout == 0.5:
                    manager._pause_event.clear()
                return False

        stop_event = _FakeStopEvent()
        manager._stop_event = stop_event

        self.assertTrue(manager._wait_until_next_check())
        self.assertEqual(stop_event.wait_calls, [manager._check_interval, 0.5, manager._check_interval])

    def test_shutdown_barrier_keeps_failed_shutdown_group_inactive(self):
        actor = SimpleNamespace(check_health=_FakeAsyncRemoteMethod(True))
        worker_info = WorkerSnapshot(
            rank=0,
            actor=actor,
            url="http://worker-0",
            lifecycle_state=WorkerLifecycleState.INACTIVE,
        )
        manager, registry = self._build_manager({0: worker_info})

        with patch.object(manager, "_shutdown_worker_group", return_value=False):
            manager.check_and_shutdown_inactive_workers()

        self.assertEqual(self._worker_by_rank(registry, 0).lifecycle_state, WorkerLifecycleState.INACTIVE)

    def test_restart_barrier_keeps_failed_recovery_group_inactive(self):
        actor = SimpleNamespace(check_health=_FakeAsyncRemoteMethod(True))
        worker_info = WorkerSnapshot(
            rank=0,
            actor=actor,
            url="http://worker-0",
            lifecycle_state=WorkerLifecycleState.INACTIVE,
        )
        manager, registry = self._build_manager({0: worker_info})

        with (
            patch.object(manager, "_restart_worker_group", return_value=False),
            patch("xtuner.v1.rl.rollout.health_manager.logger.error") as log_error,
        ):
            manager.restart_inactive_workers()

        self.assertEqual(self._worker_by_rank(registry, 0).lifecycle_state, WorkerLifecycleState.INACTIVE)
        self.assertTrue(
            any("training can continue" in call.args[0] for call in log_error.call_args_list),
            f"Expected restart failure log to explain why it is non-fatal, got: {log_error.call_args_list}",
        )

    def test_restart_barrier_notifies_recovered_group_after_success(self):
        actor = SimpleNamespace(check_health=_FakeAsyncRemoteMethod(True))
        worker_info = WorkerSnapshot(
            rank=0,
            actor=actor,
            url="http://worker-0",
            session_url="http://session-0",
            lifecycle_state=WorkerLifecycleState.INACTIVE,
        )
        recovered_groups = []
        listener = SimpleNamespace(
            on_worker_group_inactive=MagicMock(),
            on_worker_group_recovered=recovered_groups.append,
        )
        manager, registry = self._build_manager(
            {0: worker_info},
            worker_lifecycle_listeners=[listener],
        )

        with patch.object(manager, "_restart_worker_group", return_value=True):
            manager.restart_inactive_workers()

        self.assertTrue(self._worker_by_rank(registry, 0).is_active())
        self.assertEqual([group.ranks for group in recovered_groups], [(0,)])
        self.assertTrue(all(worker.is_active() for worker in recovered_groups[0].workers))

    def test_restart_barrier_cleans_claimed_groups_when_stopping(self):
        actor = SimpleNamespace(check_health=_FakeAsyncRemoteMethod(True))
        worker_info = WorkerSnapshot(
            rank=0,
            actor=actor,
            url="http://worker-0",
            lifecycle_state=WorkerLifecycleState.INACTIVE,
        )
        manager, registry = self._build_manager({0: worker_info})

        def stop_after_restart(groups):
            manager._stop_event.set()
            return {group.ranks: False for group in groups}

        with (
            patch.object(manager, "_restart_worker_groups", side_effect=stop_after_restart),
            patch.object(manager, "_shutdown_worker_group", return_value=True) as shutdown_group,
        ):
            manager.restart_inactive_workers()

        shutdown_group.assert_called_once()
        self.assertEqual(shutdown_group.call_args.args[0].ranks, (0,))
        self.assertEqual(shutdown_group.call_args.kwargs, {"wait_server_down": False})
        self.assertEqual(self._worker_by_rank(registry, 0).lifecycle_state, WorkerLifecycleState.INACTIVE)

    def test_shutdown_without_waiting_server_down_does_not_probe_worker_server(self):
        actor = SimpleNamespace(
            shutdown=_FakeAsyncRemoteMethod(None),
            check_health=_FakeAsyncRemoteMethod(True),
        )
        worker_info = WorkerSnapshot(
            rank=0,
            actor=actor,
            url="http://worker-0",
            lifecycle_state=WorkerLifecycleState.INACTIVE,
        )
        manager, registry = self._build_manager({0: worker_info})
        group = registry.claim_inactive_groups_for_recovery()[0]

        def fake_ray_get(ref, timeout=None):
            del timeout
            return asyncio.run(ref)

        with patch("xtuner.v1.rl.rollout.health_manager.ray.get", side_effect=fake_ray_get):
            self.assertTrue(manager._shutdown_worker_group(group, wait_server_down=False))

        self.assertEqual(actor.shutdown.calls, [()])
        self.assertEqual(actor.check_health.calls, [])

    def test_restart_worker_group_uses_reinit(self):
        init_result = RolloutWorkerInitResult(
            rank=0,
            server_url="http://worker-0",
            session_url="http://session-0",
        )
        actor = SimpleNamespace(
            set_skip_load_weights=_FakeAsyncRemoteMethod(None),
            init=_FakeAsyncRemoteMethod(init_result),
            reinit=_FakeAsyncRemoteMethod(init_result),
            check_health=_FakeAsyncRemoteMethod(True),
            offload=_FakeAsyncRemoteMethod(None),
            restore_skip_load_weights=_FakeAsyncRemoteMethod(None),
        )
        worker_info = WorkerSnapshot(
            rank=0,
            actor=actor,
            url="http://worker-0",
            session_url="http://session-0",
            lifecycle_state=WorkerLifecycleState.INACTIVE,
        )
        manager, registry = self._build_manager({0: worker_info})
        group = registry.claim_inactive_groups_for_recovery()[0]

        def fake_ray_get(refs, timeout=None):
            del timeout
            return [asyncio.run(ref) for ref in refs]

        with (
            patch.object(manager, "_shutdown_worker_group", return_value=True),
            patch("xtuner.v1.rl.rollout.health_manager.ray.get", side_effect=fake_ray_get),
        ):
            result = manager._restart_worker_group(group)

        self.assertTrue(result)
        self.assertEqual(actor.set_skip_load_weights.calls, [(True,)])
        self.assertEqual(actor.reinit.calls, [()])
        self.assertEqual(actor.init.calls, [])
        self.assertEqual(actor.check_health.calls, [()])
        self.assertEqual(actor.offload.calls, [()])
        self.assertEqual(actor.restore_skip_load_weights.calls, [()])

    def test_recovered_listener_runs_outside_lifecycle_operation_lock(self):
        actor = SimpleNamespace(check_health=_FakeAsyncRemoteMethod(True))
        worker_info = WorkerSnapshot(
            rank=0,
            actor=actor,
            url="http://worker-0",
            lifecycle_state=WorkerLifecycleState.INACTIVE,
        )
        lock_acquired_by_listener = []
        manager, _ = self._build_manager({0: worker_info})

        def on_worker_group_recovered(group):
            acquired = manager._lifecycle_operation_lock.acquire(blocking=False)
            lock_acquired_by_listener.append(acquired)
            if acquired:
                manager._lifecycle_operation_lock.release()

        manager._worker_lifecycle_listeners = (
            SimpleNamespace(
                on_worker_group_inactive=MagicMock(),
                on_worker_group_recovered=on_worker_group_recovered,
            ),
        )

        with patch.object(manager, "_restart_worker_group", return_value=True):
            manager.restart_inactive_workers()

        self.assertEqual(lock_acquired_by_listener, [True])


class TestPartialRolloutHandler(unittest.IsolatedAsyncioTestCase):
    async def test_preprocess_and_postprocess_preserve_response_prefix(self):
        # partial rollout 续写时应复用 prompt+历史 response，并把新 response token 追加到历史后面。
        rollout_state = RolloutState(
            rollout_id=1,
            message=[],
            prompt_ids=[10, 11],
            response="old",
            response_ids=[101, 102],
            logprobs=[0.1, 0.2],
            sample_params=SampleParams(max_tokens=5, return_token_ids=True),
            status=Status.ABORTED,
        )
        handler = PartialRolloutHandler()

        out = handler.preprocess(rollout_state, max_tokens=5)
        self.assertEqual(out.tokens, [10, 11, 101, 102])
        self.assertEqual(out.sample_params.max_tokens, 3)

        out = await handler.postprocess(
            out,
            response="new",
            response_ids=[201, 202],
            logprobs=[0.3, 0.4],
            routed_experts=None,
            finish_reason="stop",
            status=Status.COMPLETED,
            prompt_tokens=4,
            completion_tokens=2,
        )

        self.assertEqual(out.response, "oldnew")
        self.assertEqual(out.response_ids, [101, 102, 201, 202])
        self.assertEqual(out.logprobs, [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(out.finish_reason, "stop")
        self.assertEqual(out.status, Status.COMPLETED)

    async def test_multi_round_partial_rollout_never_exceeds_max_tokens(self):
        # 多轮 abort + continue 后，累计 response_ids 不应超过原始 max_tokens 预算。
        max_tokens = 5
        handler = PartialRolloutHandler()
        rollout_state = RolloutState(
            rollout_id=2,
            message=[],
            prompt_ids=[10],
            response="a",
            response_ids=[101, 102],
            logprobs=[0.1, 0.2],
            sample_params=SampleParams(max_tokens=max_tokens, return_token_ids=True),
            status=Status.ABORTED,
        )

        rollout_state = handler.preprocess(rollout_state, max_tokens=max_tokens)
        self.assertEqual(rollout_state.sample_params.max_tokens, 3)
        rollout_state = await handler.postprocess(
            rollout_state,
            response="b",
            response_ids=[201, 202],
            logprobs=[0.3, 0.4],
            routed_experts=None,
            finish_reason="abort",
            status=Status.ABORTED,
            prompt_tokens=3,
            completion_tokens=2,
        )
        self.assertLessEqual(len(rollout_state.response_ids), max_tokens)

        rollout_state = handler.preprocess(rollout_state, max_tokens=max_tokens)
        self.assertEqual(rollout_state.sample_params.max_tokens, 1)
        rollout_state = await handler.postprocess(
            rollout_state,
            response="c",
            response_ids=[301],
            logprobs=[0.5],
            routed_experts=None,
            finish_reason="stop",
            status=Status.COMPLETED,
            prompt_tokens=5,
            completion_tokens=1,
        )

        self.assertEqual(rollout_state.response_ids, [101, 102, 201, 202, 301])
        self.assertLessEqual(len(rollout_state.response_ids), max_tokens)

    async def test_postprocess_frees_old_routed_expert_refs_after_concat(self):
        # partial rollout 拼接 routed_experts 后，应释放历史和当前 ObjectRef，避免长期占用对象存储。
        class FakeObjectRef:
            def __init__(self, value):
                self.value = value

            def __await__(self):
                async def _resolve():
                    return self.value

                return _resolve().__await__()

        history_ref = FakeObjectRef([[1], [2]])
        cur_ref = FakeObjectRef([[1], [2], [3]])
        concat_ref = FakeObjectRef(None)
        rollout_state = RolloutState(
            message=[],
            response="old",
            response_ids=[1, 2],
            logprobs=[0.1, 0.2],
            routed_experts=history_ref,
            status=Status.ABORTED,
        )

        with (
            patch("xtuner.v1.rl.rollout.utils.RayObjectRef", FakeObjectRef),
            patch("xtuner.v1.rl.rollout.utils.ray.put", return_value=concat_ref) as ray_put,
            patch("xtuner.v1.rl.rollout.utils.free_object_refs") as free_object_refs,
        ):
            out = await PartialRolloutHandler().postprocess(
                rollout_state,
                response="new",
                response_ids=[3],
                logprobs=[0.3],
                routed_experts=cur_ref,
                finish_reason="abort",
                status=Status.ABORTED,
                prompt_tokens=3,
                completion_tokens=1,
            )

        self.assertIs(out.routed_experts, concat_ref)
        self.assertEqual(ray_put.call_args.args[0].tolist(), [[1], [2], [3]])
        free_object_refs.assert_any_call([history_ref])
        free_object_refs.assert_any_call([cur_ref])
        self.assertEqual(free_object_refs.call_count, 2)
