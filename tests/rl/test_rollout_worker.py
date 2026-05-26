import asyncio
import threading
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from xtuner.v1.data_proto.rl_data import Status
from xtuner.v1.rl.rollout.sglang import SGLangWorker
from xtuner.v1.rl.rollout.worker import RolloutWorker
from xtuner.v1.utils.httpx_utils import HttpRequestErrorType


class TestSGLangWorker(unittest.TestCase):
    def test_pause_generation_sets_abort_flag_before_server_pause(self):
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
        worker = RolloutWorker.__new__(RolloutWorker)
        worker.receive_abort_request = threading.Event()
        worker.receive_abort_request.set()
        rollout_state = MagicMock()

        result = await worker.generate(rollout_state)

        self.assertIs(result, rollout_state)
        self.assertEqual(rollout_state.finish_reason, "abort")
        self.assertEqual(rollout_state.status, Status.ABORTED)

    async def test_pause_generation_sets_abort_flag(self):
        worker = RolloutWorker.__new__(RolloutWorker)
        worker.receive_abort_request = threading.Event()
        worker._send_abort_request = AsyncMock(return_value=True)

        result = await worker.pause_generation()

        self.assertTrue(result)
        self.assertTrue(worker.receive_abort_request.is_set())
        worker._send_abort_request.assert_awaited_once_with()

    async def test_send_abort_request_uses_abort_timeout(self):
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

    async def test_safe_post_request_returns_aborted_on_cancellation(self):
        worker = RolloutWorker.__new__(RolloutWorker)
        worker.receive_abort_request = threading.Event()
        worker.logger = MagicMock()
        send_started = asyncio.Event()
        send_cancelled = asyncio.Event()

        class _Client:
            def build_request(self, *args, **kwargs):
                return object()

            async def send(self, req):
                send_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    send_cancelled.set()
                    raise

        worker.client = _Client()

        task = asyncio.create_task(worker._safe_post_request("http://test", headers={}, payload={"input_ids": [1]}))
        await send_started.wait()
        task.cancel()

        result = await task

        self.assertEqual(result.error_type, HttpRequestErrorType.REQUEST_ABORTED)
        self.assertTrue(worker.receive_abort_request.is_set())
        self.assertTrue(send_cancelled.is_set())

    async def test_safe_post_request_cancels_inflight_request_after_abort_timeout(self):
        worker = RolloutWorker.__new__(RolloutWorker)
        worker.receive_abort_request = threading.Event()
        worker.abort_timeout = 0.01
        worker.logger = MagicMock()
        send_started = asyncio.Event()
        send_cancelled = asyncio.Event()

        class _Client:
            def build_request(self, *args, **kwargs):
                return object()

            async def send(self, req):
                send_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    send_cancelled.set()
                    raise

        worker.client = _Client()

        task = asyncio.create_task(worker._safe_post_request("http://test", headers={}, payload={"input_ids": [1]}))
        await send_started.wait()
        worker.receive_abort_request.set()

        result = await task

        self.assertEqual(result.error_type, HttpRequestErrorType.REQUEST_ABORTED)
        self.assertTrue(send_cancelled.is_set())

    async def test_safe_post_request_keeps_abort_response_within_timeout(self):
        worker = RolloutWorker.__new__(RolloutWorker)
        worker.receive_abort_request = threading.Event()
        worker.abort_timeout = 1.0
        worker.logger = MagicMock()
        send_started = asyncio.Event()
        finish_send = asyncio.Event()

        class _Response:
            def raise_for_status(self):
                return None

        response = _Response()

        class _Client:
            def build_request(self, *args, **kwargs):
                return object()

            async def send(self, req):
                send_started.set()
                await finish_send.wait()
                return response

        worker.client = _Client()

        task = asyncio.create_task(worker._safe_post_request("http://test", headers={}, payload={"input_ids": [1]}))
        await send_started.wait()
        worker.receive_abort_request.set()
        finish_send.set()

        result = await task

        self.assertIs(result.response, response)


if __name__ == "__main__":
    unittest.main()
