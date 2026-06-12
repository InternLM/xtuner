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
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.rl.rollout.controller import RolloutController, WorkerInfo
from xtuner.v1.rl.rollout.health_manager import RolloutHealthManager
from xtuner.v1.rl.rollout.sglang import SGLangWorker
from xtuner.v1.rl.rollout.utils import PartialRolloutHandler, WorkerLifecycleState
from xtuner.v1.rl.rollout.worker import RolloutWorker
from xtuner.v1.utils.httpx_utils import HttpRequestErrorType


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


class TestRolloutController(unittest.IsolatedAsyncioTestCase):
    def _state(self, uid: int, session_uid: int) -> RolloutState:
        return RolloutState(
            uid=uid,
            message_uid=uid,
            session_uid=session_uid,
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
        state = self._state(uid=1, session_uid=123)
        router = _FakeRolloutRouter(worker=None)
        controller = self._build_controller(router)

        with patch("xtuner.v1.rl.rollout.controller.XTUNER_DETERMINISTIC", False):
            result = await controller.generate(state)

        self.assertIs(result, state)
        self.assertEqual(router.session_ids, [123])
        self.assertEqual(result.status, Status.FAILED)
        self.assertEqual(result.error_msg, "No active rollout worker available.")

    async def test_generate_routes_to_active_worker(self):
        # 有 active worker 时，controller 要按 session_uid 路由，并返回 worker 的 rollout 结果。
        request_state = self._state(uid=1, session_uid=456)
        returned_state = self._state(uid=1, session_uid=456)
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

    async def test_safe_post_request_returns_response_when_send_succeeds(self):
        # safe post 的正常路径只负责发请求、raise_for_status，并把 response 放进 HttpRequestResult。
        worker = RolloutWorker.__new__(RolloutWorker)
        worker.receive_abort_request = threading.Event()
        worker.logger = MagicMock()

        class _Response:
            def __init__(self):
                self.raise_for_status = MagicMock()

        response = _Response()
        request = object()

        class _Client:
            def build_request(self, *args, **kwargs):
                return request

            async def send(self, req):
                self.sent_request = req
                return response

        client = _Client()
        worker.client = client

        result = await worker._safe_post_request("http://test", headers={"h": "v"}, payload={"input_ids": [1]})

        self.assertIs(client.sent_request, request)
        self.assertIs(result.response, response)
        response.raise_for_status.assert_called_once_with()


class TestRolloutHealthManager(unittest.TestCase):
    def _build_manager(self, workers_info, *, failure_threshold=1):
        config = SimpleNamespace(health_check_interval_seconds=10, health_check_failure_threshold=failure_threshold)
        return RolloutHealthManager(config, workers_info, worker_infos_lock=threading.RLock())

    def test_marks_worker_inactive_after_consecutive_health_failures(self):
        actor = SimpleNamespace(check_health=_FakeAsyncRemoteMethod(False))
        worker_info = WorkerInfo(actor=actor, url="http://worker-0")
        workers_info = {0: worker_info}
        manager = self._build_manager(workers_info, failure_threshold=2)

        manager._check_active_workers_and_mark_failed_groups()

        self.assertTrue(worker_info.is_active())
        self.assertEqual(actor.check_health.calls, [()])

        manager._check_active_workers_and_mark_failed_groups()

        self.assertFalse(worker_info.is_active())
        self.assertEqual(worker_info.lifecycle_state, WorkerLifecycleState.INACTIVE)
        self.assertEqual(actor.check_health.calls, [(), ()])

    def test_inactive_worker_is_not_cleaned_up_again(self):
        # 已 inactive 的 worker 不再重复健康检查。
        actor = SimpleNamespace(check_health=_FakeAsyncRemoteMethod(True))
        workers_info = {
            0: WorkerInfo(
                actor=actor,
                url="http://worker-0",
                lifecycle_state=WorkerLifecycleState.INACTIVE,
            )
        }
        manager = self._build_manager(workers_info)

        checked_count = manager._check_active_workers_and_mark_failed_groups()

        self.assertEqual(checked_count, 0)
        self.assertEqual(actor.check_health.calls, [])


class TestPartialRolloutHandler(unittest.IsolatedAsyncioTestCase):
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
