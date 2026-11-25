"""Behavior tests for PRM judging through public interfaces.

TestPRMScoring
    test_mean_step_score_and_request_body
    test_batch_judge
    test_retry_recovers_from_invalid_response
    test_invalid_response_is_not_silent_zero
    test_transport_and_http_errors_are_classified
    test_out_of_range_and_non_finite_scores_fail
    test_boundary_scores_are_clamped_within_tolerance
    test_n_steps_mismatch_fails
    test_composed_real_outcome_and_prm_process
    test_remote_contract_preserves_payload
TestPRMInputValidation
    test_missing_response_fails_before_request
    test_invalid_constructor_arguments
    test_config_composes_with_existing_judgers
TestPRMServerHelpers
    test_split_steps_ignores_empty_segments
    test_step_end_offsets_track_repeated_steps
    test_locate_step_positions_uses_last_char
"""

import asyncio
import unittest

import httpx

from xtuner.v1.data_proto.rl_data import RolloutState
from xtuner.v1.rl.judger import (
    ComposedJudger,
    ComposedJudgerConfig,
    JudgerConfig,
    PRMJudger,
    PRMJudgerConfig,
    RemoteJudger,
    merge_outcome_and_process,
)
from xtuner.v1.rl.judger.math_rule import MathRuleJudger
from xtuner.v1.rl.judger.prm_server import locate_step_positions, split_steps, step_end_char_offsets


def _body(step_scores=(0.6, 0.4), n_steps=None):
    # 步骤分取值保证均值为精确的 0.5，避免浮点尾差干扰断言。
    return {
        "step_scores": list(step_scores),
        "n_steps": len(step_scores) if n_steps is None else n_steps,
    }


class PRMTestCase(unittest.TestCase):
    """Provide a real rollout state and replace only the external PRM transport."""

    def setUp(self):
        self.state = RolloutState(
            group_id=3,
            rollout_id=7,
            task_name="prm_test",
            message=[{"role": "user", "content": "What is 2/9 + 1/4?"}],
            response="Step one: common denominator 36.\n\nStep two: 8/36 + 9/36 = 17/36.",
            reward_model={"ground_truth": "17/36"},
        )

    def _judger(self, transport, **kwargs):
        return PRMJudger(
            endpoint="http://prm.test/score",
            retry_backoff=0,
            transport=transport,
            **kwargs,
        )

    async def _unused_transport(self, request):
        self.fail("Transport must not be called for invalid input")


class TestPRMScoring(PRMTestCase):
    """Exercise scoring, batching, retries, composition, and remote judging."""

    def test_mean_step_score_and_request_body(self):
        # 验证公开 judge 接口返回步骤分均值，请求体只含回复文本与步骤分隔符。
        requests = []

        async def transport(request):
            requests.append(request)
            return _body()

        judged = asyncio.run(self._judger(transport).judge(self.state))

        self.assertEqual(judged.reward["score"], 0.5)
        self.assertEqual(judged.reward["process_score"], 0.5)
        self.assertEqual(judged.reward["step_scores"], [0.6, 0.4])
        self.assertEqual(judged.reward["n_steps"], 2)
        self.assertEqual(judged.reward["judge_status"], "ok")
        self.assertEqual(judged.reward["judge_attempts"], 1)
        self.assertGreaterEqual(judged.reward["judge_latency_seconds"], 0.0)
        self.assertEqual(judged.reward["rollout_id"], 7)
        request = requests[0]
        self.assertEqual(set(request), {"text", "step_separator"})
        self.assertEqual(request["text"], self.state.response)
        self.assertEqual(request["step_separator"], "\n\n")

    def test_batch_judge(self):
        # 验证批量公开接口逐条返回对应的过程奖励。
        async def transport(request):
            return _body()

        states = [self.state.model_copy(deep=True), self.state.model_copy(deep=True)]
        judged = asyncio.run(self._judger(transport).batch_judge(states))
        self.assertEqual([state.reward["score"] for state in judged], [0.5, 0.5])

    def test_retry_recovers_from_invalid_response(self):
        # 验证首次返回非法响应时重试，第二次合法结果正常计分。
        calls = 0

        async def transport(request):
            nonlocal calls
            calls += 1
            if calls == 1:
                return {"unexpected": "shape"}
            return _body((1.0,))

        judged = asyncio.run(self._judger(transport, max_retries=1).judge(self.state))
        self.assertEqual(judged.reward["score"], 1.0)
        self.assertEqual(judged.reward["judge_attempts"], 2)

    def test_invalid_response_is_not_silent_zero(self):
        # 验证服务返回空步骤分时明确失败，不静默记为 0 分。
        async def transport(request):
            return _body(())

        with self.assertRaises(RuntimeError) as raised:
            asyncio.run(self._judger(transport, max_retries=0).judge(self.state))
        self.assertIn("non-empty step_scores", str(raised.exception))
        self.assertIn('"error_type": "protocol"', str(raised.exception))

    def test_transport_and_http_errors_are_classified(self):
        # 验证网络超时与 HTTP 状态错误分为 transport 和 http，均不发放过程奖励。
        async def timeout_transport(request):
            raise httpx.ReadTimeout("read timeout")

        with self.assertRaises(RuntimeError) as raised:
            asyncio.run(self._judger(timeout_transport, max_retries=0).judge(self.state))
        self.assertIn('"error_type": "transport"', str(raised.exception))
        self.assertIn('"rollout_id": 7', str(raised.exception))
        self.assertIn('"group_id": 3', str(raised.exception))
        self.assertIn('"task_name": "prm_test"', str(raised.exception))

        async def http_error_transport(request):
            http_request = httpx.Request("POST", "http://prm.test/score")
            response = httpx.Response(503, request=http_request)
            raise httpx.HTTPStatusError("service unavailable", request=http_request, response=response)

        with self.assertRaises(RuntimeError) as raised:
            asyncio.run(self._judger(http_error_transport, max_retries=0).judge(self.state))
        self.assertIn('"error_type": "http"', str(raised.exception))

    def test_out_of_range_and_non_finite_scores_fail(self):
        # 验证步骤分越界（超出钳制容差）或非有限值时 fail-closed，不进入奖励融合。
        for step_scores, message in [
            ([1.5, 0.4], "outside [0, 1]"),
            ([-0.2], "outside [0, 1]"),
            ([float("nan")], "finite number"),
            ([float("inf")], "finite number"),
            (["0.5"], "finite number"),
        ]:
            with self.subTest(step_scores=step_scores):

                async def transport(request, step_scores=step_scores):
                    return _body(step_scores)

                with self.assertRaises(RuntimeError) as raised:
                    asyncio.run(self._judger(transport, max_retries=0).judge(self.state))
                self.assertIn(message, str(raised.exception))
                self.assertIn('"error_type": "protocol"', str(raised.exception))

    def test_boundary_scores_are_clamped_within_tolerance(self):
        # 验证 MSE 回归头轻微越界（容差 0.01 内）时钳制到 [0, 1] 后正常计分。
        async def transport(request):
            return _body((1.005, -0.004))

        judged = asyncio.run(self._judger(transport, max_retries=0).judge(self.state))
        self.assertEqual(judged.reward["step_scores"], [1.0, 0.0])
        self.assertEqual(judged.reward["score"], 0.5)

    def test_n_steps_mismatch_fails(self):
        # 验证 n_steps 与步骤分数组长度不一致时明确失败。
        async def transport(request):
            return _body((0.8, 0.4), n_steps=3)

        with self.assertRaisesRegex(RuntimeError, "does not match"):
            asyncio.run(self._judger(transport, max_retries=0).judge(self.state))

    def test_composed_real_outcome_and_prm_process(self):
        # 验证真实数学结果判分与 PRM 过程判分经组合接口得到正确融合分：
        # 过程分 0.5 映射为 2*0.5-1=0，与结果分 1 按 0.3 权重融合为 0.7。
        async def transport(request):
            return _body()

        judge = ComposedJudger(
            branches={
                "outcome": MathRuleJudger(kind="symbolic"),
                "process": self._judger(transport),
            },
            merge_fn=merge_outcome_and_process,
        )
        state = self.state.model_copy(deep=True)
        state.data_source = {"outcome": 1.0, "process": 1.0}
        state.response = "The result is \\boxed{17/36}."
        state.finish_reason = "stop"

        judged = asyncio.run(judge.judge(state))

        self.assertEqual(judged.reward["outcome_score"], 1.0)
        self.assertEqual(judged.reward["process_score"], 0.0)
        self.assertAlmostEqual(judged.reward["score"], 0.7)
        self.assertEqual(judged.reward["process_detail"]["step_scores"], [0.6, 0.4])

    def test_remote_contract_preserves_payload(self):
        # 验证远程代理仍使用 PRMJudger 的预处理与样本身份字段。
        local = self._judger(self._remote_transport)

        class RemoteMethod:
            async def remote(self, payload):
                return await local.judge_payload(payload)

        class Actor:
            judge_payload = RemoteMethod()

        remote = RemoteJudger(Actor(), "remote_prm", preprocess_judger=local)
        judged = asyncio.run(remote.judge(self.state))
        self.assertEqual(judged.reward["score"], 0.5)

    async def _remote_transport(self, request):
        self.assertEqual(request["text"], self.state.response)
        return _body()


class TestPRMInputValidation(PRMTestCase):
    """Reject malformed input and configuration before calling the PRM server."""

    def test_missing_response_fails_before_request(self):
        # 验证缺少 detokenized 回复时在请求前失败。
        state = self.state.model_copy(deep=True)
        state.response = None
        with self.assertRaisesRegex(ValueError, "detokenized response"):
            asyncio.run(self._judger(self._unused_transport, max_retries=0).judge(state))

    def test_invalid_constructor_arguments(self):
        # 验证空 endpoint、空分隔符与非法重试配置在构造时失败。
        with self.assertRaisesRegex(ValueError, "endpoint"):
            PRMJudger(endpoint="")
        with self.assertRaisesRegex(ValueError, "step_separator"):
            PRMJudger(endpoint="http://prm.test/score", step_separator="")
        with self.assertRaisesRegex(ValueError, "timeout or retry"):
            PRMJudger(endpoint="http://prm.test/score", max_retries=-1)

    def test_config_composes_with_existing_judgers(self):
        # 验证 PRMJudgerConfig 可直接进入现有组合配置并保留默认重试次数。
        config = ComposedJudgerConfig(
            branches={
                "outcome": JudgerConfig(judger_name="outcome", reward_handler=lambda **kwargs: {"score": 1.0}),
                "process": PRMJudgerConfig(
                    judger_name="process",
                    endpoint="http://prm.test/score",
                ),
            }
        )
        process = config.branches["process"].build_local()
        self.assertIsInstance(process, PRMJudger)
        self.assertEqual(process.max_retries, 2)
        self.assertEqual(process.request_timeout, 60.0)


class TestPRMServerHelpers(unittest.TestCase):
    """Exercise the serving-side step location helpers without a GPU."""

    def test_split_steps_ignores_empty_segments(self):
        # 验证空段落（连续分隔符、首尾分隔符）不计入步骤。
        text = "first step\n\n\n\nsecond step\n\n"
        self.assertEqual(split_steps(text, "\n\n"), ["first step", "second step"])

    def test_step_end_offsets_track_repeated_steps(self):
        # 验证重复出现的相同步骤各自定位到自己的区间末尾。
        text = "same step\n\nsame step"
        steps = split_steps(text, "\n\n")
        self.assertEqual(steps, ["same step", "same step"])
        ends = step_end_char_offsets(text, steps)
        self.assertEqual(ends, [len("same step"), len(text)])

    def test_locate_step_positions_uses_last_char(self):
        # 验证步骤分读取自覆盖步骤末字符的 token，未被任何 token 覆盖的步骤跳过。
        # 两个 token：token0 覆盖字符 [0, 5)，token1 覆盖字符 [5, 13)。
        offset_mapping = [[0, 5], [5, 13]]
        positions = locate_step_positions(offset_mapping, ends=[5, 13])
        self.assertEqual(positions, [0, 1])
        # 步骤末字符落在所有 token 覆盖范围之外（如结尾空白被 tokenizer 丢弃）时跳过。
        self.assertEqual(locate_step_positions(offset_mapping, ends=[20]), [])


if __name__ == "__main__":
    unittest.main()
