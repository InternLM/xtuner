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

import httpx
from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams, Status
from xtuner.v1.rl.rollout.controller import RolloutController
from xtuner.v1.rl.rollout.sglang import SGLangWorker
from xtuner.v1.rl.rollout.utils import PartialRolloutHandler, RolloutHealthChecker
from xtuner.v1.rl.rollout.worker import RolloutWorker
from xtuner.v1.utils.httpx_utils import HttpRequestErrorType, HttpRequestResult


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
        async def passthrough_response(rollout_state, http_response):
            return rollout_state

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
        worker._safe_handle_response = AsyncMock(side_effect=safe_handle_response or passthrough_response)
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

    async def test_partial_rollout_eos_response_completes_without_backend_request(self):
        # partial rollout 已以 EOS 结束时，应直接完成，不再请求推理后端。
        worker = self._build_partial_rollout_worker(eos_token=[999])
        rollout_state = RolloutState(
            uid=1,
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
            uid=2,
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

        cases = [
            ("timeout", timeout_result(), None, ("Request failed", "3")),
            ("request_error", request_error_result(), None, ("Request failed", "3")),
            ("client_error", client_error_result(), None, ("Client error",)),
            ("server_error", server_error_result(), None, ("Server error",)),
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

    async def test_generate_attempt_counts_for_retryable_and_non_retryable_errors(self):
        # max_retry_per_sample=3 表示首轮请求 + 3 次 retry；非 retryable HTTP 错误不应重试。
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

        for safe_post_result, expected_calls in (
            (timeout_result(), 4),
            (client_error_result(), 1),
            (server_error_result(), 1),
        ):
            worker = self._build_mock_error_rollout_worker(safe_post_result=safe_post_result)
            await worker.generate(RolloutState(message=[{"role": "user", "content": "Hello!"}]))
            self.assertEqual(worker._safe_post_request.await_count, expected_calls)

    async def test_generate_retries_invalid_response_for_configured_attempts(self):
        # invalid response 回到 response handler 内判定；handler 抛 retryable 异常后由 Tenacity 重试。
        from xtuner.v1.rl.rollout import worker as rollout_worker

        async def invalid_response(rollout_state, http_response):
            raise rollout_worker._RetryableInvalidRolloutResponseError("missing finish_reason")

        worker = self._build_mock_error_rollout_worker(
            safe_post_result=HttpRequestResult(response=object()),
            safe_handle_response=invalid_response,
        )

        result_state = await worker.generate(RolloutState(message=[{"role": "user", "content": "Hello!"}]))

        self.assertEqual(worker._safe_post_request.await_count, 4)
        self.assertEqual(worker._safe_handle_response.await_count, 4)
        self.assertEqual(result_state.status, Status.FAILED)
        self.assertIn("missing finish_reason", result_state.error_msg)

    async def test_generate_retries_response_handler_failed_status_for_backwards_compatibility(self):
        # Some rollout backends still signal invalid responses by returning
        # Status.FAILED instead of raising the retryable invalid-response error.
        async def failed_response(rollout_state, http_response):
            rollout_state.status = Status.FAILED
            rollout_state.error_msg = "handler returned incomplete token data"
            return rollout_state

        worker = self._build_mock_error_rollout_worker(
            safe_post_result=HttpRequestResult(response=object()),
            safe_handle_response=failed_response,
        )

        result_state = await worker.generate(RolloutState(message=[{"role": "user", "content": "Hello!"}]))

        self.assertEqual(worker._safe_post_request.await_count, 4)
        self.assertEqual(worker._safe_handle_response.await_count, 4)
        self.assertEqual(result_state.status, Status.FAILED)
        self.assertIn("handler returned incomplete token data", result_state.error_msg)


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
    async def test_preprocess_and_postprocess_preserve_response_prefix(self):
        # partial rollout 续写时应复用 prompt+历史 response，并把新 response token 追加到历史后面。
        rollout_state = RolloutState(
            uid=1,
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
            uid=2,
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
