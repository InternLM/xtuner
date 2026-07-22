"""RL immediate recovery 的故障恢复与慢导出回退测试。"""

import asyncio
import inspect
import os
import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from xtuner.v1.rl.rollout.controller import RolloutController
from xtuner.v1.rl.rollout.health_manager import RolloutHealthManager
from xtuner.v1.rl.rollout.rollout_topology import RolloutEngine, RolloutServerProcess, RolloutTopology
from xtuner.v1.rl.rollout.worker import RolloutWorker, RolloutWorkerInitResult
from xtuner.v1.rl.rollout.worker_registry import RolloutWorkerRegistry
from xtuner.v1.train.rl_trainer import RLColocateTrainer, RLDisaggregatedTrainer


def _identity_ray_get(value, *, timeout=None):
    del timeout
    return value


class _LocalRemoteMethod:
    """Expose a local callable through the small Ray method surface used here."""

    def __init__(self, func):
        self._func = func
        self.calls = []

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self._func(*args, **kwargs)


class _LocalAsyncRemoteMethod:
    """Expose a local callable through the async ``.remote()`` surface used by Ray actors."""

    def __init__(self, func):
        self._func = func
        self.calls = []

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))

        async def invoke():
            return self._func(*args, **kwargs)

        return invoke()


class _RecoverableRolloutActor:
    """A fake actor whose backend health changes across crash and reinit."""

    def __init__(self, rank: int, *, healthy: bool = True):
        self.rank = rank
        self.server_url = f"http://worker-{rank}"
        self.session_url = f"http://session-{rank}"
        self.healthy = healthy
        self.loaded_model_path = None
        self.loaded_tokenizer_path = None
        self.loaded_skip_load_weights = None

        worker = RolloutWorker.__new__(RolloutWorker)
        worker.rank = rank
        worker.server_url = self.server_url
        worker.server_task = object()
        worker.server_process = None
        worker.logger = MagicMock()
        self.worker = worker

        self.inject_backend_crash_for_test = _LocalAsyncRemoteMethod(worker.inject_backend_crash_for_test)
        self.check_health = _LocalAsyncRemoteMethod(lambda: self.healthy)
        self.shutdown = _LocalAsyncRemoteMethod(self._shutdown)
        self.reinit = _LocalAsyncRemoteMethod(self._reinit)
        self.offload = _LocalAsyncRemoteMethod(lambda: None)

    def _shutdown(self) -> None:
        self.healthy = False

    def _reinit(
        self,
        *,
        model_path: str | None = None,
        tokenizer_path: str | None = None,
        skip_load_weights: bool | None = None,
    ) -> RolloutWorkerInitResult:
        self.loaded_model_path = model_path
        self.loaded_tokenizer_path = tokenizer_path
        self.loaded_skip_load_weights = skip_load_weights
        self.healthy = True
        return RolloutWorkerInitResult(
            rank=self.rank,
            server_url=self.server_url,
            session_url=self.session_url,
        )


def _local_ray_get(value, *, timeout=None):
    del timeout

    def resolve(item):
        if inspect.isawaitable(item):
            return asyncio.run(item)
        return item

    if isinstance(value, list):
        return [resolve(item) for item in value]
    return resolve(value)


class TestRLImmediateRecovery(unittest.TestCase):
    _MODEL_PATH = "/tmp/ready-recovery-hf"
    _TOKENIZER_PATH = "/tmp/ready-recovery-tokenizer"

    def _build_registry(self, actors: list[_RecoverableRolloutActor]) -> RolloutWorkerRegistry:
        topology = RolloutTopology(
            engines=tuple(
                RolloutEngine(
                    engine_ranks=(actor.rank,),
                    dist_init_addr=f"addr-{actor.rank}",
                    server_processes=(
                        RolloutServerProcess(
                            worker_rank=actor.rank,
                            placement_group_bundle_idxs=(actor.rank,),
                            accepts_rollout_requests=True,
                            weight_update_ranks=(actor.rank,),
                        ),
                    ),
                )
                for actor in actors
            )
        )
        registry = RolloutWorkerRegistry(rollout_topology=topology)
        registry.register_started_servers(
            init_results=tuple(
                RolloutWorkerInitResult(
                    rank=actor.rank,
                    server_url=actor.server_url,
                    session_url=actor.session_url,
                )
                for actor in actors
            ),
            workers_by_rank=tuple(actors),
        )
        return registry

    def _build_health_manager(
        self,
        registry: RolloutWorkerRegistry,
        *,
        listeners=(),
    ) -> RolloutHealthManager:
        config = SimpleNamespace(
            health_check_interval_seconds=10,
            health_check_timeout_seconds=1.0,
            health_check_failure_threshold=1,
        )
        return RolloutHealthManager(
            config=config,
            registry=registry,
            worker_lifecycle_listeners=listeners,
        )

    def _build_controller(
        self,
        registry: RolloutWorkerRegistry,
        health_manager: RolloutHealthManager,
    ) -> RolloutController:
        controller = RolloutController.__new__(RolloutController)
        controller.logger = MagicMock()
        controller.registry = registry
        controller.health_manager = health_manager
        return controller

    def test_injected_single_server_failure_recovers_immediately_from_ready_hf(self):
        """测试内容：注入单个 rollout server 故障后，使用 ready HF 立即恢复 active 状态。

        测试流程：
        1. 注册一个健康 worker，并向 HealthManager 发布已经完成的 ready recovery HF。
        2. 通过 Controller 调用 Worker 的测试注入接口强制停止 backend server。
        3. 执行一次健康检查，由 HealthManager 检测并立即恢复该 worker。
        4. 验证 reinit 使用 ready HF，且 worker 重新变为 active。
        """
        actor = _RecoverableRolloutActor(rank=0)
        registry = self._build_registry([actor])
        health_manager = self._build_health_manager(registry)
        health_manager.set_ready_recovery_hf(
            model_path=self._MODEL_PATH,
            tokenizer_path=self._TOKENIZER_PATH,
        )
        controller = self._build_controller(registry, health_manager)
        server_task = actor.worker.server_task

        def cancel_backend(*args, **kwargs):
            del args, kwargs
            actor.healthy = False

        with (
            patch.dict(os.environ, {"XTUNER_TEST_IMMEDIATE_RECOVERY": "1"}),
            patch("xtuner.v1.rl.rollout.controller.ray.get", side_effect=_local_ray_get),
            patch("xtuner.v1.rl.rollout.health_manager.ray.get", side_effect=_local_ray_get),
            patch("xtuner.v1.rl.rollout.worker.ray.cancel", side_effect=cancel_backend) as ray_cancel,
        ):
            controller.inject_backend_crash_for_test(rank=0)
            health_manager.run_once()

        ray_cancel.assert_called_once_with(server_task, force=True, recursive=True)
        self.assertIsNone(actor.worker.server_task)
        self.assertEqual(
            actor.reinit.calls,
            [
                (
                    (),
                    {
                        "model_path": self._MODEL_PATH,
                        "tokenizer_path": self._TOKENIZER_PATH,
                        "skip_load_weights": False,
                    },
                )
            ],
        )
        self.assertTrue(registry.active_entrypoint_by_rank(0).is_active())
        self.assertEqual(actor.offload.calls, [])

    def test_all_server_groups_failure_recovers_every_group_from_same_ready_hf(self):
        """测试内容：所有 rollout server group 同时失效后，全部使用同一 ready HF 立即恢复。

        测试流程：
        1. 注册三个独立 lifecycle group，并预先发布同一个 ready recovery HF。
        2. 将三个 fake backend 同时置为不健康，执行一次健康检查模拟全部 worker 挂掉。
        3. HealthManager 检测故障后自动恢复全部 inactive group。
        4. 验证每组都用同一路径 reinit、恢复通知完整，最终所有 worker 重新 active。
        """
        actors = [_RecoverableRolloutActor(rank=rank) for rank in range(3)]
        registry = self._build_registry(actors)
        inactive_groups = []
        recovered_groups = []
        lifecycle_events = []
        listener = SimpleNamespace(
            on_worker_group_inactive=lambda group: (
                inactive_groups.append(group),
                lifecycle_events.append(("inactive", group.ranks)),
            ),
            on_worker_group_recovered=lambda group: (
                recovered_groups.append(group),
                lifecycle_events.append(("recovered", group.ranks)),
            ),
        )
        health_manager = self._build_health_manager(registry, listeners=[listener])
        health_manager.set_ready_recovery_hf(
            model_path=self._MODEL_PATH,
            tokenizer_path=self._TOKENIZER_PATH,
        )
        for actor in actors:
            actor.healthy = False

        with patch("xtuner.v1.rl.rollout.health_manager.ray.get", side_effect=_local_ray_get):
            health_manager.run_once()

        expected_reinit_call = [
            (
                (),
                {
                    "model_path": self._MODEL_PATH,
                    "tokenizer_path": self._TOKENIZER_PATH,
                    "skip_load_weights": False,
                },
            )
        ]
        self.assertEqual([group.ranks for group in inactive_groups], [(0,), (1,), (2,)])
        self.assertEqual([group.ranks for group in recovered_groups], [(0,), (1,), (2,)])
        self.assertEqual(
            lifecycle_events,
            [
                ("inactive", (0,)),
                ("inactive", (1,)),
                ("inactive", (2,)),
                ("recovered", (0,)),
                ("recovered", (1,)),
                ("recovered", (2,)),
            ],
        )
        self.assertTrue(all(worker.is_active() for worker in registry.all_workers()))
        for actor in actors:
            self.assertEqual(actor.reinit.calls, expected_reinit_call)
            self.assertEqual(actor.loaded_model_path, self._MODEL_PATH)
            self.assertEqual(actor.loaded_tokenizer_path, self._TOKENIZER_PATH)
            self.assertFalse(actor.loaded_skip_load_weights)
            self.assertEqual(actor.offload.calls, [])

    def test_inactive_group_recovers_on_health_run_after_ready_hf_is_published(self):
        """测试内容：ready HF 只唤醒 HealthManager，由下一次 health run 恢复 inactive group。"""
        actor = _RecoverableRolloutActor(rank=0, healthy=False)
        registry = self._build_registry([actor])
        health_manager = self._build_health_manager(registry)

        with patch("xtuner.v1.rl.rollout.health_manager.ray.get", side_effect=_local_ray_get):
            health_manager.run_once()
            self.assertIsNone(registry.active_entrypoint_by_rank(0))
            self.assertEqual(actor.reinit.calls, [])

            health_manager.set_ready_recovery_hf(
                model_path=self._MODEL_PATH,
                tokenizer_path=self._TOKENIZER_PATH,
            )
            self.assertIsNone(registry.active_entrypoint_by_rank(0))
            self.assertEqual(actor.reinit.calls, [])
            self.assertTrue(health_manager._health_loop_wakeup_event.is_set())

            # Simulate the background loop consuming the wakeup and owning the
            # next health/recovery workflow.
            health_manager._health_loop_wakeup_event.clear()
            health_manager.run_once()

        self.assertTrue(registry.active_entrypoint_by_rank(0).is_active())
        self.assertEqual(
            actor.reinit.calls,
            [
                (
                    (),
                    {
                        "model_path": self._MODEL_PATH,
                        "tokenizer_path": self._TOKENIZER_PATH,
                        "skip_load_weights": False,
                    },
                )
            ],
        )

class TestImmediateRecoveryHFInterval(unittest.TestCase):
    def test_hf_interval_reuses_regular_hf_without_async_export(self):
        """测试内容：命中 hf_interval 时复用常规 HF，并跳过 recovery HF 的异步导出。

        测试流程：
        1. 构造 step 2 命中 hf_interval=2 的 trainer，并准备常规保存产生的 hf-step-2。
        2. 调用 recovery HF 调度入口，模拟常规 HF 保存完成后的 immediate-recovery 发布阶段。
        3. 验证该路径直接发布 hf-step-2，并将其记录为当前 ready recovery HF。
        4. 验证 start_hf_export 从未调用，且没有创建 pending 异步导出 Future。
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = RLColocateTrainer.__new__(RLColocateTrainer)
            trainer._enable_immediate_recovery = True
            trainer._hf_interval = 2
            trainer._total_train_steps = 4
            trainer._rollout_config = SimpleNamespace(
                tokenizer_path="/tmp/tokenizer",
                model_path="/tmp/model",
            )
            trainer._ready_recovery_hf_path = None
            trainer._pending_hf_export = None
            trainer._meta = SimpleNamespace(
                latest_exp=SimpleNamespace(
                    exp_dir=temp_dir,
                    hf_checkpoint_list=[],
                ),
            )
            regular_hf_path = trainer.exp_dir / trainer._HF_DIR / "hf-step-2"
            regular_hf_path.mkdir(parents=True)
            trainer._meta.latest_exp.hf_checkpoint_list.append(str(regular_hf_path))

            clear_ready_hf = _LocalRemoteMethod(lambda: None)
            set_ready_hf = _LocalRemoteMethod(lambda **_kwargs: None)
            trainer.rollout_controller = SimpleNamespace(
                clear_ready_recovery_hf=clear_ready_hf,
                set_ready_recovery_hf=set_ready_hf,
            )
            start_hf_export = MagicMock()
            trainer.train_controller = SimpleNamespace(start_hf_export=start_hf_export)

            with patch("xtuner.v1.train.rl_trainer.ray.get", side_effect=_identity_ray_get):
                trainer._maybe_save_recovery_hf(cur_step=2)

        self.assertEqual(clear_ready_hf.calls, [((), {})])
        self.assertEqual(
            set_ready_hf.calls,
            [
                (
                    (),
                    {
                        "model_path": str(regular_hf_path),
                        "tokenizer_path": "/tmp/tokenizer",
                    },
                )
            ],
        )
        start_hf_export.assert_not_called()
        self.assertIsNone(trainer._pending_hf_export)
        self.assertEqual(trainer._ready_recovery_hf_path, regular_hf_path)


class TestImmediateRecoverySlowExportFallback(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _build_pending_export(self, events: list[str], name: str) -> tuple[MagicMock, Path]:
        recovery_hf_path = Path(self.temp_dir.name) / name
        recovery_hf_path.mkdir(parents=True)
        pending = MagicMock(spec=Future)
        pending.done.return_value = False

        def finish_export():
            events.append("wait_for_pending_export")
            return recovery_hf_path

        pending.result.side_effect = finish_export
        return pending, recovery_hf_path

    def test_colocate_slow_export_falls_back_before_weight_update(self):
        """测试内容：colocate 下慢于下一次同步点的异步 HF 必须回退到权重更新恢复。

        测试流程：
        1. 构造尚未完成的 recovery HF Future，模拟异步保存跨过下一次同步点。
        2. 执行完整的 colocate save/sync 编排，等待旧 Future 收尾并清空 ready HF。
        3. 确认 immediate recovery 被关闭、未完成快照被删除且没有启动下一次异步保存。
        4. 确认 inactive group 的恢复发生在 bind/update_weights 之前，走权重更新基线。
        """
        events: list[str] = []
        pending, recovery_hf_path = self._build_pending_export(events, "colocate-hf-step-1")
        trainer = RLColocateTrainer.__new__(RLColocateTrainer)
        trainer.logger = MagicMock()
        trainer._enable_immediate_recovery = True
        trainer._pending_hf_export = pending
        trainer._ready_recovery_hf_path = None
        trainer._sync_weights_interval = 1
        trainer._enable_evaluate = False
        trainer._evaluate_step = 1
        trainer._total_train_steps = 2
        trainer._rollout_config = SimpleNamespace()
        trainer._maybe_save_checkpoint = AsyncMock(side_effect=lambda _step: events.append("save_checkpoint"))
        trainer._maybe_save_hf = MagicMock(side_effect=lambda _step: events.append("save_regular_hf"))
        trainer.train_controller = SimpleNamespace(
            offload=MagicMock(side_effect=lambda *, target: events.append(f"offload_{target}")),
            update_weights=MagicMock(side_effect=lambda: events.append("update_weights")),
        )
        trainer.rollout_controller = SimpleNamespace(
            clear_ready_recovery_hf=_LocalRemoteMethod(lambda: events.append("clear_ready_hf")),
            restart_inactive_workers=_LocalRemoteMethod(lambda: events.append("restart_inactive")),
            onload_weights=_LocalRemoteMethod(lambda: events.append("onload_weights")),
            onload_kvcache=_LocalRemoteMethod(lambda: events.append("onload_kvcache")),
        )

        with (
            patch("xtuner.v1.train.rl_trainer.asyncio_run", side_effect=asyncio.run),
            patch("xtuner.v1.train.rl_trainer.ray.get", side_effect=_identity_ray_get),
            patch(
                "xtuner.v1.train.rl_trainer.bind_train_rollout",
                side_effect=lambda **_kwargs: events.append("bind"),
            ),
        ):
            synced = trainer._sync_weights_and_save(train_step=1, step_timer_dict={})

        self.assertTrue(synced)
        self.assertFalse(trainer._enable_immediate_recovery)
        self.assertIsNone(trainer._pending_hf_export)
        self.assertIsNone(trainer._ready_recovery_hf_path)
        self.assertFalse(recovery_hf_path.exists())
        self.assertEqual(
            events,
            [
                "wait_for_pending_export",
                "clear_ready_hf",
                "offload_optimizer",
                "save_checkpoint",
                "save_regular_hf",
                "restart_inactive",
                "bind",
                "onload_weights",
                "update_weights",
                "offload_model",
                "onload_kvcache",
            ],
        )

    def test_disaggregated_slow_export_falls_back_before_weight_update(self):
        """测试内容：disaggregated 下慢异步 HF 也必须在权重同步前切回恢复基线。

        测试流程：
        1. 构造跨过下一次同步点的 pending recovery HF Future。
        2. 执行 disaggregated 的 save -> restart -> bind -> update_weights 编排。
        3. 确认同步点先等待并撤销 ready HF，然后关闭 immediate recovery、删除旧快照。
        4. 确认只恢复 inactive group 并直接进入本轮权重更新，不再启动 recovery HF。
        """
        events: list[str] = []
        pending, recovery_hf_path = self._build_pending_export(events, "disaggregated-hf-step-1")
        trainer = RLDisaggregatedTrainer.__new__(RLDisaggregatedTrainer)
        trainer.logger = MagicMock()
        trainer._enable_immediate_recovery = True
        trainer._pending_hf_export = pending
        trainer._ready_recovery_hf_path = None
        trainer._rollout_config = SimpleNamespace(weight_update_host="host", weight_update_port=1234)
        trainer._maybe_save_checkpoint = AsyncMock(side_effect=lambda _step: events.append("save_checkpoint"))
        trainer._maybe_save_hf = MagicMock(side_effect=lambda _step: events.append("save_regular_hf"))
        trainer.update_weights = MagicMock(side_effect=lambda: events.append("update_weights"))
        trainer.train_controller = MagicMock()
        trainer.rollout_controller = SimpleNamespace(
            clear_ready_recovery_hf=_LocalRemoteMethod(lambda: events.append("clear_ready_hf")),
            restart_inactive_workers=SimpleNamespace(
                remote=AsyncMock(side_effect=lambda: events.append("restart_inactive"))
            ),
        )

        with (
            patch("xtuner.v1.train.rl_trainer.ray.get", side_effect=_identity_ray_get),
            patch(
                "xtuner.v1.train.rl_trainer.bind_train_rollout",
                side_effect=lambda **_kwargs: events.append("bind"),
            ),
        ):
            asyncio.run(trainer._sync_weights_and_save(model_step=1, step_timer_dict={}))

        self.assertFalse(trainer._enable_immediate_recovery)
        self.assertIsNone(trainer._pending_hf_export)
        self.assertIsNone(trainer._ready_recovery_hf_path)
        self.assertFalse(recovery_hf_path.exists())
        self.assertEqual(
            events,
            [
                "wait_for_pending_export",
                "clear_ready_hf",
                "save_checkpoint",
                "save_regular_hf",
                "restart_inactive",
                "bind",
                "update_weights",
            ],
        )


if __name__ == "__main__":
    unittest.main()
