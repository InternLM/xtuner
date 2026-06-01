"""Rollout worker / utils 的 PR-fast 逻辑测试。

本文件合并旧的 test_rollout_worker.py 和 test_rollout_utils.py 中不依赖真实模型/后端的测试：
- SGLangWorker pause/continue 对 abort flag 和 server request 的控制。
- RolloutWorker abort、abort request timeout 和 in-flight request 取消语义。
- RolloutHealthChecker 对 inactive/unhealthy worker 的清理逻辑。
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
from xtuner.v1.rl.rollout.sglang import SGLangWorker
from xtuner.v1.rl.rollout.utils import PartialRolloutHandler, RolloutHealthChecker
from xtuner.v1.rl.rollout.worker import RolloutWorker
from xtuner.v1.utils.httpx_utils import HttpRequestErrorType


class _FakeRemoteMethod:
    def __init__(self, name, call_log):
        self.name = name
        self.call_log = call_log

    def remote(self):
        self.call_log.append((self.name, "remote"))
        return self.name


class _FakeWorker:
    def __init__(self):
        self.call_log = []
        self.offload = _FakeRemoteMethod("offload", self.call_log)
        self.shutdown = _FakeRemoteMethod("shutdown", self.call_log)


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

    async def test_send_abort_request_uses_abort_timeout(self):
        # abort request 使用独立 timeout，避免 pause 时长被普通请求超时配置拖住。
        worker = RolloutWorker.__new__(RolloutWorker)
        worker.server_url = "http://test"
        worker.abort_timeout = 0.25
        worker.logger = MagicMock()
        response = MagicMock()
        response.raise_for_status = MagicMock()

        client = MagicMock()
        client.post = AsyncMock(return_value=response)
        client_context = MagicMock()
        client_context.__aenter__ = AsyncMock(return_value=client)
        client_context.__aexit__ = AsyncMock(return_value=None)

        with patch("xtuner.v1.rl.rollout.worker.httpx.AsyncClient", return_value=client_context) as client_cls:
            result = await worker._send_abort_request()

        self.assertTrue(result)
        client_cls.assert_called_once_with(timeout=0.25)
        client.post.assert_awaited_once_with(
            "http://test/abort_request",
            json={"abort_all": True},
        )

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


class TestRolloutHealthChecker(unittest.TestCase):
    def _build_checker(self, workers_info):
        config = SimpleNamespace(health_check_interval_seconds=10, health_check_failure_threshold=1)
        return RolloutHealthChecker(config, workers_info)

    def test_shutdown_runs_when_offload_fails(self):
        # worker 健康检查失败且 offload 也失败时，health checker 应 shutdown 并标记 inactive。
        worker = _FakeWorker()
        workers_info = {0: SimpleNamespace(actor=worker, url="http://worker-0", is_active=True)}
        checker = self._build_checker(workers_info)

        async def unhealthy_worker(*args, **kwargs):
            return False

        def ray_get(ref, timeout=None):
            worker.call_log.append((ref, "get"))
            if ref == "offload":
                raise RuntimeError("offload failed")
            return None

        with (
            patch("xtuner.v1.rl.rollout.utils.check_worker_health", side_effect=unhealthy_worker),
            patch("xtuner.v1.rl.rollout.utils.ray.get", side_effect=ray_get),
        ):
            checker.run_once()

        self.assertFalse(workers_info[0].is_active)
        self.assertEqual(
            worker.call_log,
            [
                ("offload", "remote"),
                ("offload", "get"),
                ("shutdown", "remote"),
                ("shutdown", "get"),
            ],
        )

    def test_inactive_worker_is_not_cleaned_up_again(self):
        # 已 inactive 的 worker 不再重复健康检查、offload 或 shutdown。
        worker = _FakeWorker()
        workers_info = {0: SimpleNamespace(actor=worker, url="http://worker-0", is_active=False)}
        checker = self._build_checker(workers_info)

        with (
            patch("xtuner.v1.rl.rollout.utils.check_worker_health") as check_worker_health_mock,
            patch("xtuner.v1.rl.rollout.utils.ray.get") as ray_get_mock,
        ):
            checker.run_once()

        check_worker_health_mock.assert_not_called()
        ray_get_mock.assert_not_called()
        self.assertEqual(worker.call_log, [])


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
