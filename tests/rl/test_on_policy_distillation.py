import asyncio
import json
import math
import socket
import threading
import time
import unittest
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from unittest.mock import MagicMock

import torch

from xtuner.v1.data_proto.rl_data import (
    RolloutState,
    SampleParams,
    Status,
    get_group_status,
    reset_rollout_response,
)
from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.rl.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from xtuner.v1.rl.loss import GRPOLossConfig, GRPOLossContext
from xtuner.v1.rl.on_policy_distillation import (
    OPDConfig,
    OPDTeacherConfig,
    TeacherLogprobClient,
    compute_pg_opd_token_advantages,
)
from xtuner.v1.rl.replay_buffer import AsyncReplayBufferConfig
from xtuner.v1.rl.rollout_is import RolloutImportanceSampling
from xtuner.v1.rl.trainer.controller import TrainingController


@dataclass
class _HTTPResponse:
    body: Any
    status: int = 200
    delay_s: float = 0.0


class _EndpointState:
    def __init__(
        self,
        responses: list[_HTTPResponse] | None = None,
        on_request: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.responses = list(responses or [])
        self.on_request = on_request
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        self.request_event = threading.Event()
        self.max_active_requests = 0
        self._active_requests = 0
        self._lock = threading.Lock()

    def handle(self, payload: dict[str, Any], headers: dict[str, str]) -> _HTTPResponse:
        with self._lock:
            self.requests.append(payload)
            self.headers.append(headers)
            self._active_requests += 1
            self.max_active_requests = max(self.max_active_requests, self._active_requests)
            response = self.responses.pop(0) if self.responses else self._success_response(payload)
            self.request_event.set()
        if self.on_request is not None:
            self.on_request(payload)
        try:
            if response.delay_s:
                time.sleep(response.delay_s)
            return response
        finally:
            with self._lock:
                self._active_requests -= 1

    @staticmethod
    def _success_response(payload: dict[str, Any]) -> _HTTPResponse:
        scored_tokens = payload["input_ids"][payload["logprob_start_len"] :]
        return _HTTPResponse(
            body={
                "meta_info": {
                    "input_token_logprobs": [[-0.1 - index / 100, token] for index, token in enumerate(scored_tokens)]
                }
            }
        )


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        content_length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(content_length))
        response = self.server.endpoint_state.handle(payload, dict(self.headers))  # type: ignore[attr-defined]
        body = response.body if isinstance(response.body, bytes) else json.dumps(response.body).encode()
        self.send_response(response.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format: str, *args: Any) -> None:
        return


class _FakeEndpoint:
    def __init__(
        self,
        responses: list[_HTTPResponse] | None = None,
        on_request: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.state = _EndpointState(responses, on_request)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.server.endpoint_state = self.state  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


def _state(
    rollout_id: int,
    *,
    data_source: str = "math",
    prompt_ids: list[int] | None = None,
    response_ids: list[int] | None = None,
) -> RolloutState:
    prompt_ids = prompt_ids or [100 + rollout_id]
    response_ids = response_ids or [200 + rollout_id, 2]
    return RolloutState(
        rollout_id=rollout_id,
        group_id=0,
        message=[{"role": "user", "content": f"prompt {rollout_id}"}],
        prompt_ids=prompt_ids,
        tokens=prompt_ids,
        response="response",
        response_ids=response_ids,
        logprobs=[-0.4] * len(response_ids),
        response_mask=[1] * len(response_ids),
        status=Status.COMPLETED,
        extra_fields={"origin_data_source": data_source},
    )


def _opd_config(endpoint_by_name: dict[str, str], data_source_teacher_map: dict[str, str], **kwargs) -> OPDConfig:
    return OPDConfig(
        teachers=[
            OPDTeacherConfig(name=name, endpoint=endpoint, **kwargs) for name, endpoint in endpoint_by_name.items()
        ],
        data_source_teacher_map=data_source_teacher_map,
    )


def _algorithm_config(*, task_adv_weight: float = 0.0, opd_adv_weight: float = 1.0) -> OPDConfig:
    return OPDConfig(
        task_adv_weight=task_adv_weight,
        opd_adv_weight=opd_adv_weight,
        teachers=[OPDTeacherConfig(name="teacher", endpoint="http://127.0.0.1:1")],
        data_source_teacher_map={"math": "teacher"},
    )


class _FixedAdvantageEstimator:
    def __init__(self, values: list[float]) -> None:
        self.values = values
        self.calls: list[tuple[torch.Tensor, list[RolloutState]]] = []

    def compute(self, rewards: torch.Tensor, group: list[RolloutState]) -> torch.Tensor:
        self.calls.append((rewards.clone(), group))
        return torch.tensor(self.values[: len(group)], dtype=torch.float32)


class TestTeacherLogprobClient(unittest.IsolatedAsyncioTestCase):
    def _endpoint(self, responses: list[_HTTPResponse] | None = None) -> _FakeEndpoint:
        endpoint = _FakeEndpoint(responses)
        self.addCleanup(endpoint.close)
        return endpoint

    async def test_compute_logprobs_uses_sglang_prefill_protocol(self):
        endpoint = self._endpoint()
        client = TeacherLogprobClient(
            OPDTeacherConfig(name="teacher", endpoint=endpoint.url, api_key="secret", max_retry_per_sample=0)
        )
        self.addAsyncCleanup(client.close)
        state = _state(1, prompt_ids=[10], response_ids=[20, 2])

        result = await client.compute_logprobs(state)

        self.assertIs(result, state)
        self.assertEqual(result.teacher_tokens, [20, 2])
        for actual, expected in zip(result.teacher_logprobs or [], [-0.11, -0.12], strict=True):
            self.assertAlmostEqual(actual, expected)
        payload = endpoint.state.requests[0]
        self.assertEqual(payload["input_ids"], [10, 20, 2])
        self.assertEqual(payload["logprob_start_len"], 0)
        self.assertEqual(payload["top_logprobs_num"], 0)
        self.assertEqual(
            payload["sampling_params"],
            {"max_new_tokens": 0, "temperature": 1.0, "skip_special_tokens": False},
        )
        self.assertEqual(endpoint.state.headers[0]["Authorization"], "Bearer secret")

    async def test_invalid_teacher_responses_fail_without_signal(self):
        responses = [
            _HTTPResponse(b"not-json"),
            _HTTPResponse({"missing": "meta_info"}),
            _HTTPResponse({"meta_info": {"input_token_logprobs": [[-0.1, 10], [-0.2, 20]]}}),
            _HTTPResponse({"meta_info": {"input_token_logprobs": [[-0.1, 10], [-0.2, 20], [-0.3, 2], [-0.4, 3]]}}),
            _HTTPResponse({"meta_info": {"input_token_logprobs": [[-0.1, 10], [-0.2, 999], [-0.3, 2]]}}),
            _HTTPResponse({"meta_info": {"input_token_logprobs": [[-0.1, 10], [float("nan"), 20], [-0.3, 2]]}}),
            _HTTPResponse({"meta_info": {"input_token_logprobs": [[-0.1, 10], [float("inf"), 20], [-0.3, 2]]}}),
        ]
        endpoint = self._endpoint(responses)
        client = TeacherLogprobClient(OPDTeacherConfig(name="teacher", endpoint=endpoint.url, max_retry_per_sample=0))
        self.addAsyncCleanup(client.close)

        for rollout_id in range(len(responses)):
            with self.subTest(rollout_id=rollout_id):
                result = await client.compute_logprobs(_state(rollout_id, prompt_ids=[10], response_ids=[20, 2]))
                self.assertEqual(result.status, Status.FAILED)
                self.assertIsNone(result.teacher_tokens)
                self.assertIsNone(result.teacher_logprobs)
                self.assertIn("scoring failed after 1 attempts", result.error_msg or "")

    async def test_http_status_retries_then_succeeds(self):
        endpoint = self._endpoint([_HTTPResponse({"error": "busy"}, status=503)])
        client = TeacherLogprobClient(OPDTeacherConfig(name="teacher", endpoint=endpoint.url, max_retry_per_sample=1))
        self.addAsyncCleanup(client.close)

        result = await client.compute_logprobs(_state(1))

        self.assertEqual(result.status, Status.COMPLETED)
        self.assertEqual(len(endpoint.state.requests), 2)

    async def test_http_status_retry_exhaustion_marks_failed(self):
        for status in (400, 500):
            with self.subTest(status=status):
                endpoint = self._endpoint([_HTTPResponse({"error": "failed"}, status=status) for _ in range(2)])
                client = TeacherLogprobClient(
                    OPDTeacherConfig(name="teacher", endpoint=endpoint.url, max_retry_per_sample=1)
                )
                self.addAsyncCleanup(client.close)

                result = await client.compute_logprobs(_state(status))

                self.assertEqual(result.status, Status.FAILED)
                self.assertEqual(len(endpoint.state.requests), 2)

    async def test_timeout_and_connection_error_use_bounded_retries(self):
        endpoint = self._endpoint([_HTTPResponse({}, delay_s=0.05) for _ in range(2)])
        timeout_client = TeacherLogprobClient(
            OPDTeacherConfig(
                name="timeout",
                endpoint=endpoint.url,
                request_timeout_s=0.01,
                max_retry_per_sample=1,
            )
        )
        self.addAsyncCleanup(timeout_client.close)

        timeout_result = await timeout_client.compute_logprobs(_state(1))

        self.assertEqual(timeout_result.status, Status.FAILED)
        self.assertEqual(len(endpoint.state.requests), 2)

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        closed_port = sock.getsockname()[1]
        sock.close()
        connection_client = TeacherLogprobClient(
            OPDTeacherConfig(
                name="connection",
                endpoint=f"http://127.0.0.1:{closed_port}",
                request_timeout_s=0.1,
                max_retry_per_sample=1,
            )
        )
        self.addAsyncCleanup(connection_client.close)

        connection_result = await connection_client.compute_logprobs(_state(2))

        self.assertEqual(connection_result.status, Status.FAILED)

    async def test_max_concurrency_limits_requests_per_client(self):
        endpoint = self._endpoint([_HTTPResponse({}, delay_s=0.05) for _ in range(5)])
        client = TeacherLogprobClient(
            OPDTeacherConfig(name="teacher", endpoint=endpoint.url, max_retry_per_sample=0, max_concurrency=2)
        )
        self.addAsyncCleanup(client.close)

        results = await asyncio.gather(*(client.compute_logprobs(_state(index)) for index in range(5)))

        self.assertEqual(endpoint.state.max_active_requests, 2)
        self.assertTrue(all(result.status == Status.FAILED for result in results))


class TestAgentLoopTeacherScoring(unittest.IsolatedAsyncioTestCase):
    def _endpoint(
        self,
        responses: list[_HTTPResponse] | None = None,
        on_request: Callable[[dict[str, Any]], None] | None = None,
    ) -> _FakeEndpoint:
        endpoint = _FakeEndpoint(responses, on_request)
        self.addCleanup(endpoint.close)
        return endpoint

    def _loop(self, opd_config: OPDConfig) -> SingleTurnAgentLoop:
        loop = SingleTurnAgentLoop.__new__(SingleTurnAgentLoop)
        loop.rollout_ctl = MagicMock()
        loop.sample_params = SampleParams(max_tokens=8)
        loop.judger = None
        loop.enable_batch_judge = False
        loop._judger_pause_event = asyncio.Event()
        loop.logger = MagicMock()
        loop.configure_opd(opd_config)
        self.addAsyncCleanup(loop.close)
        return loop

    @staticmethod
    def _complete(state: RolloutState) -> RolloutState:
        state.status = Status.COMPLETED
        return state

    async def test_eager_scoring_starts_before_other_samples_finish_generation(self):
        endpoint = self._endpoint()
        loop = self._loop(_opd_config({"teacher": endpoint.url}, {"math": "teacher"}))
        release_second = asyncio.Event()
        second_state = _state(2)
        second_state.status = Status.INIT

        async def generate_sample(state, **kwargs):
            if state is second_state:
                await release_second.wait()
            return self._complete(state)

        loop.generate_sample = generate_sample
        task = asyncio.create_task(loop.collect_rollout_group([_state(1), second_state]))

        request_started = await asyncio.to_thread(endpoint.state.request_event.wait, 1.0)
        self.assertTrue(request_started)
        self.assertEqual(second_state.status, Status.INIT)
        release_second.set()
        result = await task

        self.assertEqual(get_group_status(result), Status.COMPLETED)
        self.assertEqual(len(endpoint.state.requests), 2)
        self.assertTrue(all(state.teacher_logprobs is not None for state in result))
        self.assertTrue(all("teacher_score_time_s" in state.extra_fields for state in result))

    async def test_filter_false_skips_teacher_endpoint(self):
        endpoint = self._endpoint()
        loop = self._loop(_opd_config({"teacher": endpoint.url}, {"math": "teacher"}))

        async def generate_sample(state, **kwargs):
            return self._complete(state)

        loop.generate_sample = generate_sample
        result = await loop.collect_rollout_group(
            [_state(1), _state(2)],
            is_valid_sample_func=lambda group: False,
        )

        self.assertEqual(get_group_status(result), Status.FILTERED)
        self.assertEqual(endpoint.state.requests, [])
        self.assertTrue(all("teacher_score_time_s" not in state.extra_fields for state in result))

    async def test_filter_true_runs_before_lazy_teacher_scoring(self):
        order = []
        endpoint = self._endpoint(on_request=lambda payload: order.append("teacher"))

        def filter_func(group):
            order.append("filter")
            return True

        loop = self._loop(_opd_config({"teacher": endpoint.url}, {"math": "teacher"}))

        async def generate_sample(state, **kwargs):
            return self._complete(state)

        loop.generate_sample = generate_sample
        result = await loop.collect_rollout_group([_state(1)], is_valid_sample_func=filter_func)

        self.assertEqual(order, ["filter", "teacher"])
        self.assertEqual(result[0].status, Status.COMPLETED)
        self.assertIsNotNone(result[0].teacher_logprobs)
        self.assertIn("teacher_score_time_s", result[0].extra_fields)

    async def test_origin_data_source_routes_groups_to_different_teachers(self):
        math_endpoint = self._endpoint()
        code_endpoint = self._endpoint()
        loop = self._loop(
            _opd_config(
                {"math_teacher": math_endpoint.url, "code_teacher": code_endpoint.url},
                {"math": "math_teacher", "code": "code_teacher"},
            )
        )

        async def generate_sample(state, **kwargs):
            return self._complete(state)

        loop.generate_sample = generate_sample
        await loop.collect_rollout_group([_state(1, data_source="math")])
        await loop.collect_rollout_group([_state(2, data_source="code")])

        self.assertEqual(len(math_endpoint.state.requests), 1)
        self.assertEqual(len(code_endpoint.state.requests), 1)

    async def test_teacher_failure_marks_group_failed(self):
        endpoint = self._endpoint([_HTTPResponse({"error": "down"}, status=500)])
        loop = self._loop(
            _opd_config(
                {"teacher": endpoint.url},
                {"math": "teacher"},
                max_retry_per_sample=0,
            )
        )

        async def generate_sample(state, **kwargs):
            return self._complete(state)

        loop.generate_sample = generate_sample
        result = await loop.collect_rollout_group([_state(1)])

        self.assertEqual(get_group_status(result), Status.FAILED)
        self.assertIsNone(result[0].teacher_logprobs)
        self.assertIn("teacher_score_time_s", result[0].extra_fields)


class TestOPDRolloutData(unittest.IsolatedAsyncioTestCase):
    async def test_rollout_state_round_trip_replay_and_reset_preserve_contract(self):
        state = _state(1)
        state.teacher_tokens = [201, 2]
        state.teacher_logprobs = [-0.2, -0.3]

        restored = RolloutState.model_validate(state.model_dump())
        self.assertEqual(restored.teacher_tokens, state.teacher_tokens)
        self.assertEqual(restored.teacher_logprobs, state.teacher_logprobs)

        replay_buffer = AsyncReplayBufferConfig().build()
        await replay_buffer.put([restored], "math")
        replayed = (await replay_buffer.get(1, "math", Status.COMPLETED))[0][0]
        self.assertEqual(replayed.teacher_tokens, [201, 2])
        self.assertEqual(replayed.teacher_logprobs, [-0.2, -0.3])

        reset_rollout_response(replayed)
        self.assertIsNone(replayed.teacher_tokens)
        self.assertIsNone(replayed.teacher_logprobs)


class TestPGOPDTokenAdvantages(unittest.TestCase):
    @staticmethod
    def _scored_state(
        rollout_id: int,
        *,
        behavior_logprobs: list[float],
        teacher_logprobs: list[float],
        reward: float | None = None,
        response_mask: list[int] | None = None,
    ) -> RolloutState:
        response_ids = list(range(100, 100 + len(behavior_logprobs)))
        state = _state(rollout_id, response_ids=response_ids)
        state.logprobs = behavior_logprobs
        state.teacher_tokens = response_ids
        state.teacher_logprobs = teacher_logprobs
        state.reward = None if reward is None else {"score": reward}
        state.response_mask = response_mask
        return state

    def test_pure_opd_uses_token_delta_and_response_mask_without_reward(self):
        state = self._scored_state(
            1,
            behavior_logprobs=[-1.0, -2.0, -3.0],
            teacher_logprobs=[-0.5, -2.5, -3.0],
            response_mask=[1, 0, 1],
        )

        advantages = compute_pg_opd_token_advantages(
            [state],
            config=_algorithm_config(task_adv_weight=0.0, opd_adv_weight=2.0),
            task_adv_estimator=None,
        )

        torch.testing.assert_close(advantages[0], torch.tensor([1.0, 0.0, 0.0]))
        self.assertFalse(advantages[0].requires_grad)

        state.response_mask = []
        unmasked = compute_pg_opd_token_advantages(
            [state],
            config=_algorithm_config(),
            task_adv_estimator=None,
        )
        torch.testing.assert_close(unmasked[0], torch.tensor([0.5, -0.5, 0.0]))

    def test_mixed_opd_combines_per_sample_task_advantage(self):
        first = self._scored_state(
            1,
            behavior_logprobs=[-1.0],
            teacher_logprobs=[-0.75],
            reward=10.0,
        )
        second = self._scored_state(
            2,
            behavior_logprobs=[-1.0],
            teacher_logprobs=[-1.25],
            reward=10.0,
        )
        third = self._scored_state(
            3,
            behavior_logprobs=[-1.0],
            teacher_logprobs=[-0.5],
            reward=-1.0,
        )
        estimator = _FixedAdvantageEstimator([2.0, 2.0, -2.0])

        advantages = compute_pg_opd_token_advantages(
            [first, second, third],
            config=_algorithm_config(task_adv_weight=0.5, opd_adv_weight=2.0),
            task_adv_estimator=estimator,
        )

        torch.testing.assert_close(torch.cat(advantages), torch.tensor([1.5, 0.5, 0.0]))
        self.assertEqual(estimator.calls[0][0].tolist(), [10.0, 10.0, -1.0])
        self.assertEqual(estimator.calls[0][1], [first, second, third])

    def test_mixed_opd_requires_reward(self):
        state = self._scored_state(
            1,
            behavior_logprobs=[-1.0],
            teacher_logprobs=[-0.5],
        )
        config = _algorithm_config(task_adv_weight=1.0)

        with self.assertRaisesRegex(ValueError, "Reward score is required"):
            compute_pg_opd_token_advantages(
                [state],
                config=config,
                task_adv_estimator=_FixedAdvantageEstimator([1.0]),
            )

    def test_token_advantages_survive_packing_and_define_denominator(self):
        state = self._scored_state(
            1,
            behavior_logprobs=[-1.0, -2.0, -3.0],
            teacher_logprobs=[-0.5, -2.5, -2.0],
            response_mask=[1, 0, 1],
        )
        response_advantages = compute_pg_opd_token_advantages(
            [state],
            config=_algorithm_config(),
            task_adv_estimator=None,
        )[0]
        data_batches = [
            {
                "seq_ctx": SequenceContext.from_input_ids((torch.tensor([[10, 11, 20, 21]]),), device="cpu"),
                "shifted_labels": torch.tensor([[-100, 100, -100, 102]]),
                "advantage": [0.0] + response_advantages.tolist(),
                "rollout_logprobs": torch.tensor([[0.0, -1.0, -2.0, -3.0]]),
            }
        ]

        controller = TrainingController.__new__(TrainingController)
        packed = controller._packing(data_batches, pack_max_length=6, language_cfg=None)

        self.assertEqual(packed[0]["shifted_labels"].shape, packed[0]["advantages"].shape)
        self.assertEqual(packed[0]["advantages"].tolist(), [[0.0, 0.5, 0.0, 1.0, -100.0, -100.0]])

        loss_config = GRPOLossConfig(
            policy_loss_cfg={"loss_type": "vanilla", "cliprange_low": 0.2, "cliprange_high": 0.2}
        )
        loss_context = loss_config.build(
            {
                "shifted_labels": packed[0]["shifted_labels"],
                "advantages": packed[0]["advantages"],
                "old_logprobs": torch.zeros_like(packed[0]["advantages"]),
            }
        )
        assert isinstance(loss_context, GRPOLossContext)
        GRPOLossContext.build_batches([loss_context])
        torch.testing.assert_close(
            loss_context.loss_kwargs.policy_loss_weight.cpu(),
            torch.tensor([[0.0, 0.5, 0.0, 0.5, 0.0, 0.0]]),
        )

    def test_teacher_signal_changes_student_gradient(self):
        def gradient(teacher_logprob: float) -> torch.Tensor:
            state = self._scored_state(
                1,
                behavior_logprobs=[-1.0],
                teacher_logprobs=[teacher_logprob],
            )
            advantage = compute_pg_opd_token_advantages(
                [state],
                config=_algorithm_config(),
                task_adv_estimator=None,
            )[0].unsqueeze(0)
            current_logprob = torch.tensor([[-1.0]], requires_grad=True)
            old_logprob = torch.tensor([[-1.0]])
            loss_config = GRPOLossConfig(
                policy_loss_cfg={"loss_type": "vanilla", "cliprange_low": 0.2, "cliprange_high": 0.2}
            )
            context = loss_config.build(
                {
                    "shifted_labels": torch.tensor([[100]]),
                    "advantages": advantage,
                    "old_logprobs": old_logprob,
                }
            )
            assert isinstance(context, GRPOLossContext)
            GRPOLossContext.build_batches([context])
            device = context.loss_kwargs.advantages.device
            current_logprob = current_logprob.to(device).detach().requires_grad_()
            loss = context.policy_loss_fn(
                current_logprob,
                context.loss_kwargs.old_logprobs,
                context.loss_kwargs.advantages,
                context.loss_kwargs.policy_loss_weight,
                loss_config.policy_loss_cfg,
            )
            loss.backward()
            return current_logprob.grad.detach().cpu().clone()

        positive_signal_gradient = gradient(-0.5)
        stronger_signal_gradient = gradient(0.0)

        self.assertFalse(torch.equal(positive_signal_gradient, stronger_signal_gradient))

    def test_pure_and_mixed_opd_support_rollout_is_off_and_on(self):
        for task_adv_weight in (0.0, 1.0):
            for enable_is in (False, True):
                with self.subTest(task_adv_weight=task_adv_weight, enable_is=enable_is):
                    state = self._scored_state(
                        1,
                        behavior_logprobs=[-1.0, -1.0],
                        teacher_logprobs=[0.0, 0.0],
                        reward=1.0 if task_adv_weight else None,
                    )
                    estimator = _FixedAdvantageEstimator([1.0]) if task_adv_weight else None
                    advantages = compute_pg_opd_token_advantages(
                        [state],
                        config=_algorithm_config(task_adv_weight=task_adv_weight),
                        task_adv_estimator=estimator,
                    )[0].unsqueeze(0)
                    rollout_is = (
                        RolloutImportanceSampling(
                            rollout_is_mode="mask",
                            rollout_is_threshold=(2.0, 0.5),
                            rollout_is_mask_threshold=(2.0, 0.5),
                        )
                        if enable_is
                        else RolloutImportanceSampling()
                    )
                    loss_config = GRPOLossConfig(
                        policy_loss_cfg={"loss_type": "vanilla", "cliprange_low": 0.2, "cliprange_high": 0.2},
                        rollout_is=rollout_is,
                    )
                    context = loss_config.build(
                        {
                            "shifted_labels": torch.tensor([[100, 101]]),
                            "advantages": advantages,
                            "rollout_logprobs": torch.tensor([[-1.0, -1.0]]),
                            "old_logprobs": torch.tensor([[-1.0, -1.0 + math.log(10.0)]]),
                        }
                    )
                    assert isinstance(context, GRPOLossContext)
                    context.compute_rollout_is(None, torch.tensor([2]))  # type: ignore[arg-type]
                    GRPOLossContext.build_batches([context])

                    device = context.loss_kwargs.advantages.device
                    current_logprobs = torch.tensor([[-1.0, -1.0]], device=device, requires_grad=True)
                    loss = context.policy_loss_fn(
                        current_logprobs,
                        context.loss_kwargs.old_logprobs,
                        context.loss_kwargs.advantages,
                        context.loss_kwargs.policy_loss_weight,
                        loss_config.policy_loss_cfg,
                    )
                    loss.backward()

                    if enable_is:
                        self.assertIsNotNone(context.loss_kwargs.is_weights)
                        self.assertEqual(context.loss_kwargs.shifted_labels.tolist(), [[100, -100]])
                        self.assertEqual(current_logprobs.grad[0, 1].item(), 0.0)
                    else:
                        self.assertIsNone(context.loss_kwargs.is_weights)
                        self.assertEqual(context.loss_kwargs.shifted_labels.tolist(), [[100, 101]])
                        self.assertNotEqual(current_logprobs.grad[0, 1].item(), 0.0)


class TestOPDConfig(unittest.TestCase):
    def test_rejects_duplicate_or_unknown_teacher_names(self):
        teacher = OPDTeacherConfig(name="teacher", endpoint="http://127.0.0.1:1")
        with self.assertRaisesRegex(ValueError, "must be unique"):
            OPDConfig(
                teachers=[teacher, teacher],
                data_source_teacher_map={"math": "teacher"},
            )
        with self.assertRaisesRegex(ValueError, "unknown teachers"):
            OPDConfig(
                teachers=[teacher],
                data_source_teacher_map={"math": "missing"},
            )


if __name__ == "__main__":
    unittest.main()
