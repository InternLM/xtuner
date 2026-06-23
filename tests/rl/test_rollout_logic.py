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
from xtuner.v1.rl.rollout.proxy_manager import RolloutProxyManager
from xtuner.v1.rl.rollout.worker_registry import RolloutWorkerRegistry, WorkerLifecycleState, WorkerSnapshot
from xtuner.v1.rl.rollout.sglang import SGLangWorker
from xtuner.v1.rl.rollout.utils import PartialRolloutHandler, SessionRouter
from xtuner.v1.rl.rollout.worker import RolloutWorker
from xtuner.v1.rl.utils.misc import delete_from_routedapiproxy
from xtuner.v1.train.rl_trainer import BaseRLTrainer, _agent_loop_manager_requires_rollout_proxy
from xtuner.v1.utils.httpx_utils import HttpRequestErrorType, HttpRequestResult


class _FakeAsyncRemoteMethod:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def remote(self):
        self.calls.append(())

        async def _result():
            if isinstance(self.result, Exception):
                raise self.result
            return self.result

        return _result()


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
        trainer.logger.info.assert_called_once()


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
        controller._tool_call_parser = None
        controller._reasoning_parser = None
        controller.logger = MagicMock()
        return controller

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
        controller.registry = RolloutWorkerRegistry(engine_rank_mesh_array=[[0], [1]], rollout_config=SimpleNamespace())
        controller.registry.register_started_server(
            rank=0,
            actor=object(),
            server_url="http://worker-0",
            session_url="http://session-0",
            is_request_entrypoint=True,
        )
        controller.registry.register_started_server(
            rank=1,
            actor=object(),
            server_url="http://worker-1",
            session_url="http://session-1",
            is_request_entrypoint=True,
        )
        controller.registry.mark_unhealthy_ranks({1})
        controller.proxy_manager = MagicMock()

        controller.register_active_workers_to_proxy()

        controller.proxy_manager.replace_registered_session_urls.assert_called_once_with(["http://session-0"])

    def test_register_active_workers_to_proxy_requires_proxy_manager(self):
        controller = RolloutController.__new__(RolloutController)
        controller.registry = RolloutWorkerRegistry(engine_rank_mesh_array=[], rollout_config=SimpleNamespace())
        controller.proxy_manager = None

        with self.assertRaisesRegex(AssertionError, "Proxy manager must be initialized"):
            controller.register_active_workers_to_proxy()


class TestRolloutProxyManager(unittest.TestCase):
    def _build_manager(self):
        config = SimpleNamespace(model_name="test-model", worker_log_dir=None)
        return RolloutProxyManager(config)

    def test_replace_registered_session_urls_replaces_proxy_registrations(self):
        manager = self._build_manager()

        with (
            patch("xtuner.v1.rl.rollout.proxy_manager.delete_from_routedapiproxy") as delete_proxy,
            patch("xtuner.v1.rl.rollout.proxy_manager.register_to_routedapiproxy") as register_proxy,
            patch("xtuner.v1.rl.rollout.proxy_manager.check_chat_completions", return_value=True),
        ):
            manager.replace_registered_session_urls(["http://session-1", "http://session-0", "http://session-1"])

        delete_proxy.assert_called_once_with("test-model")
        self.assertEqual(
            register_proxy.call_args_list,
            [
                call("test-model", "http://session-0"),
                call("test-model", "http://session-1"),
            ],
        )

    def test_replace_registered_session_urls_rolls_back_when_validation_fails(self):
        manager = self._build_manager()

        with (
            patch("xtuner.v1.rl.rollout.proxy_manager.delete_from_routedapiproxy") as delete_proxy,
            patch("xtuner.v1.rl.rollout.proxy_manager.register_to_routedapiproxy"),
            patch("xtuner.v1.rl.rollout.proxy_manager.check_chat_completions", return_value=False),
            patch("xtuner.v1.rl.rollout.proxy_manager.time.sleep"),
            self.assertRaisesRegex(RuntimeError, "check chat completions failed"),
        ):
            manager.replace_registered_session_urls(["http://session-0"])

        self.assertEqual(
            delete_proxy.call_args_list,
            [
                call("test-model"),
                call("test-model", "http://session-0"),
            ],
        )

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

        delete_proxy.assert_called_once_with("test-model", "http://session-0")
        register_proxy.assert_called_once_with("test-model", "http://session-0")

    def test_lifecycle_listener_methods_delegate_entrypoint_session_urls(self):
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
        manager.on_worker_group_recovered(worker_group)

        manager._delete_session_url.assert_called_once_with("http://session-0")
        manager._register_session_url.assert_called_once_with("http://session-0")


class TestRoutedApiProxyUtils(unittest.TestCase):
    def test_delete_from_routedapiproxy_includes_api_base_when_provided(self):
        response = MagicMock()
        response.json.return_value = {"ok": True}

        with patch("xtuner.v1.rl.utils.misc.requests.post", return_value=response) as post:
            delete_from_routedapiproxy("test-model", "http://session-0")

        post.assert_called_once()
        self.assertEqual(
            post.call_args.kwargs["json"],
            {"model_name": "test-model", "api_base": "http://session-0"},
        )
        response.raise_for_status.assert_called_once_with()


class TestRolloutWorkerRegistry(unittest.TestCase):
    def _worker_by_rank(self, registry, rank):
        return next(worker for worker in registry.all_workers() if worker.rank == rank)

    def test_registry_filters_entrypoints_and_builds_metadata_snapshot(self):
        config = SimpleNamespace()
        registry = RolloutWorkerRegistry(engine_rank_mesh_array=[[0, 1]], rollout_config=config)
        registry.register_started_server(
            rank=0,
            actor=object(),
            server_url="http://worker-0",
            session_url="http://session-0",
            lifecycle_group_ranks=(0, 1),
            is_request_entrypoint=True,
        )
        registry.register_started_server(
            rank=1,
            actor=object(),
            server_url="http://worker-1",
            session_url=None,
            lifecycle_group_ranks=(0, 1),
            is_request_entrypoint=False,
        )

        metadata = registry.training_metadata_snapshot()

        self.assertEqual(metadata["engine_rank_mesh_array"], [[0, 1]])
        self.assertIs(metadata["rollout_config"], config)
        self.assertEqual(metadata["server_url_dict"], {0: "http://worker-0"})
        self.assertEqual(metadata["worker_server_urls_status"], {"http://worker-0": True})
        self.assertEqual(metadata["worker_session_url_dict"], {0: "http://session-0"})
        self.assertEqual(metadata["worker_session_urls_status"], {"http://session-0": True})
        active_entrypoint = registry.active_entrypoints()[0]
        self.assertIsInstance(active_entrypoint, WorkerSnapshot)
        self.assertEqual(active_entrypoint.rank, 0)
        with self.assertRaises(FrozenInstanceError):
            active_entrypoint.lifecycle_state = WorkerLifecycleState.INACTIVE

        unhealthy_groups = registry.mark_unhealthy_ranks({0})
        metadata = registry.training_metadata_snapshot()

        self.assertEqual(unhealthy_groups[0].ranks, (0, 1))
        self.assertEqual(metadata["worker_server_urls_status"], {"http://worker-0": False})
        self.assertEqual(metadata["worker_session_urls_status"], {"http://session-0": False})
        self.assertEqual(tuple(worker.rank for worker in registry.inactive_workers()), (0, 1))
        self.assertEqual(registry.active_entrypoints(), ())
        claimed_groups = registry.claim_inactive_groups_for_recovery()
        self.assertEqual(claimed_groups[0].ranks, (0, 1))
        self.assertEqual(self._worker_by_rank(registry, 0).lifecycle_state, WorkerLifecycleState.RECOVERING)
        registry.set_group_recovery_result(claimed_groups[0], recovered=False)
        self.assertEqual(self._worker_by_rank(registry, 0).lifecycle_state, WorkerLifecycleState.INACTIVE)

class TestSessionRouter(unittest.IsolatedAsyncioTestCase):
    async def test_sticky_session_reselects_when_previous_entrypoint_is_inactive(self):
        actor_0 = object()
        actor_1 = object()
        registry = RolloutWorkerRegistry(engine_rank_mesh_array=[[0], [1]], rollout_config=SimpleNamespace())
        registry.register_started_server(rank=0, actor=actor_0, server_url="http://worker-0")
        registry.register_started_server(rank=1, actor=actor_1, server_url="http://worker-1")
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
        registry = RolloutWorkerRegistry(
            engine_rank_mesh_array=[sorted(workers_info)],
            rollout_config=SimpleNamespace(),
        )
        for rank, worker_info in workers_info.items():
            lifecycle_group_ranks = worker_info.lifecycle_group_ranks or (rank,)
            registry.register_started_server(
                rank=rank,
                actor=worker_info.actor,
                server_url=worker_info.url,
                session_url=worker_info.session_url,
                lifecycle_group_ranks=lifecycle_group_ranks,
                is_request_entrypoint=worker_info.is_request_entrypoint,
            )
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
        worker_info = WorkerSnapshot(actor=actor, url="http://worker-0")
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

        manager._check_and_deactivate_failed_worker_groups()

        self.assertTrue(self._worker_by_rank(registry, 0).is_active())
        self.assertEqual(actor.check_health.calls, [()])
        self.assertEqual(inactive_groups, [])

        manager._check_and_deactivate_failed_worker_groups()

        self.assertFalse(self._worker_by_rank(registry, 0).is_active())
        self.assertEqual(self._worker_by_rank(registry, 0).lifecycle_state, WorkerLifecycleState.INACTIVE)
        self.assertEqual(actor.check_health.calls, [(), ()])
        self.assertEqual([group.ranks for group in inactive_groups], [(0,)])

    def test_inactive_worker_is_not_cleaned_up_again(self):
        # 已 inactive 的 worker 不再重复健康检查。
        actor = SimpleNamespace(check_health=_FakeAsyncRemoteMethod(True))
        workers_info = {
            0: WorkerSnapshot(
                actor=actor,
                url="http://worker-0",
                lifecycle_state=WorkerLifecycleState.INACTIVE,
            )
        }
        manager, _ = self._build_manager(workers_info)

        checked_count = manager._check_and_deactivate_failed_worker_groups()

        self.assertEqual(checked_count, 0)
        self.assertEqual(actor.check_health.calls, [])

    def test_health_check_threshold_zero_disables_periodic_health_check(self):
        # threshold <= 0 表示关闭周期健康监测，不应把 active worker 直接判 inactive。
        actor = SimpleNamespace(check_health=_FakeAsyncRemoteMethod(False))
        worker_info = WorkerSnapshot(actor=actor, url="http://worker-0")
        manager, registry = self._build_manager({0: worker_info}, failure_threshold=0)

        checked_count = manager._check_and_deactivate_failed_worker_groups()

        self.assertEqual(checked_count, 0)
        self.assertTrue(self._worker_by_rank(registry, 0).is_active())
        self.assertEqual(actor.check_health.calls, [])

    def test_fail_fast_health_check_still_runs_when_periodic_health_check_is_disabled(self):
        actor = SimpleNamespace(check_health=_FakeAsyncRemoteMethod(False))
        worker_info = WorkerSnapshot(actor=actor, url="http://worker-0")
        manager, registry = self._build_manager({0: worker_info}, failure_threshold=0)

        checked_count = manager._check_and_deactivate_failed_worker_groups(fail_fast=True)

        self.assertEqual(checked_count, 1)
        self.assertFalse(self._worker_by_rank(registry, 0).is_active())
        self.assertEqual(actor.check_health.calls, [()])

    def test_health_check_uses_configured_timeout(self):
        actor = SimpleNamespace(check_health=_FakeAsyncRemoteMethod(True))
        worker_info = WorkerSnapshot(actor=actor, url="http://worker-0")
        manager, _ = self._build_manager({0: worker_info}, check_timeout=2.5)
        observed_timeouts = []

        async def fake_wait_for(awaitable, timeout):
            observed_timeouts.append(timeout)
            return await awaitable

        with patch("xtuner.v1.rl.rollout.health_manager.asyncio.wait_for", side_effect=fake_wait_for):
            manager._check_and_deactivate_failed_worker_groups()

        self.assertEqual(observed_timeouts, [2.5])

    def test_shutdown_barrier_keeps_failed_shutdown_group_inactive(self):
        actor = SimpleNamespace(check_health=_FakeAsyncRemoteMethod(True))
        worker_info = WorkerSnapshot(
            actor=actor,
            url="http://worker-0",
            lifecycle_state=WorkerLifecycleState.INACTIVE,
        )
        manager, registry = self._build_manager({0: worker_info})

        with (
            patch.object(manager, "_shutdown_worker_group", return_value=False),
            patch("xtuner.v1.rl.rollout.health_manager.logger.error") as log_error,
        ):
            manager.check_and_shutdown_inactive_workers()

        self.assertEqual(self._worker_by_rank(registry, 0).lifecycle_state, WorkerLifecycleState.INACTIVE)
        self.assertTrue(
            any("training can continue" in call.args[0] for call in log_error.call_args_list),
            f"Expected shutdown failure log to explain why it is non-fatal, got: {log_error.call_args_list}",
        )

    def test_restart_barrier_keeps_failed_recovery_group_inactive(self):
        actor = SimpleNamespace(check_health=_FakeAsyncRemoteMethod(True))
        worker_info = WorkerSnapshot(
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
