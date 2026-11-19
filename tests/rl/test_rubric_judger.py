"""Behavior tests for rubric judging through public interfaces.

TestRubricScoring
    test_weighted_score_and_multimodal_request
    test_all_met_and_all_unmet
    test_batch_judge
    test_retry_recovers_from_invalid_output
    test_invalid_output_is_not_silent_zero
    test_non_json_error_includes_content_prefix
    test_schema_fragment_without_outer_brace
    test_invalid_met_evidence_fails_closed
    test_composed_real_outcome_and_fake_process
    test_remote_contract_preserves_sample_rubrics
TestRubricInputValidation
    test_invalid_rubrics_fail_before_request
    test_total_weight_mismatch_and_missing_visual_image
TestRubricRewardMerge
    test_single_merge_and_outcome_only_regression
    test_batch_merge_and_length_check
    test_config_composes_with_existing_judgers
"""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import torch

from xtuner.v1.data_proto.rl_data import RolloutState
from xtuner.v1.rl.advantage.gae import terminal_rewards
from xtuner.v1.rl.judger import (
    ComposedJudger,
    ComposedJudgerConfig,
    JudgerConfig,
    RemoteJudger,
    RubricJudger,
    RubricJudgerConfig,
    merge_outcome_and_process,
)
from xtuner.v1.rl.judger.math_rule import MathRuleJudger


def _body(results):
    return {"choices": [{"message": {"content": json.dumps({"results": results})}}]}


def _results(verdicts=("met", "unmet"), met_evidence="The left ratio is 2/9"):
    return [
        {
            "local_id": local_id,
            "verdict": verdict,
            "evidence": met_evidence if verdict == "met" else f"missing {local_id}",
        }
        for local_id, verdict in zip(("a1", "a2"), verdicts)
    ]


class RubricTestCase(unittest.TestCase):
    """Provide a real rollout state and replace only the external judge transport."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.image_path = Path(self.temp_dir.name) / "question.png"
        self.image_path.write_bytes(b"fake-png-for-request-encoding")
        self.state = RolloutState(
            message=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": {"url": "question.png"}},
                        {"type": "text", "text": "What is the ratio?"},
                    ],
                }
            ],
            response="The left ratio is 2/9 and the right ratio is 1/4.",
            reward_model={"ground_truth": "hidden-reference"},
            extra_fields={
                "rubrics": [
                    {
                        "local_id": "a1",
                        "type": "visual_grounding",
                        "criterion": "Identifies the left grid.",
                        "weight": 1.0,
                    },
                    {
                        "local_id": "a2",
                        "type": "answer_coverage",
                        "criterion": "States both ratios.",
                        "weight": 3.0,
                    },
                ],
                "total_weight": 4.0,
                "image_data": [str(self.image_path)],
            },
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _judger(self, transport, **kwargs):
        return RubricJudger(
            endpoint="http://judge.test/v1/chat/completions",
            model="GLM-5.3-Flash",
            retry_backoff=0,
            transport=transport,
            **kwargs,
        )

    async def _unused_transport(self, request):
        self.fail("Transport must not be called for invalid input")


class TestRubricScoring(RubricTestCase):
    """Exercise scoring, batching, retries, composition, and remote judging."""

    def test_weighted_score_and_multimodal_request(self):
        # 验证公开 judge 接口返回加权分，并发送图片、schema 且不泄漏标准答案。
        requests = []

        async def transport(request):
            requests.append(request)
            return _body(_results())

        judged = asyncio.run(self._judger(transport).judge(self.state))

        self.assertEqual(judged.reward["score"], 0.25)
        self.assertEqual([item["score"] for item in judged.reward["rubric_results"]], [1.0, 0.0])
        request = requests[0]
        self.assertEqual(request["model"], "GLM-5.3-Flash")
        self.assertEqual(request["temperature"], 0.0)
        result_schema = request["response_format"]["json_schema"]["schema"]["properties"]["results"]
        self.assertEqual((result_schema["minItems"], result_schema["maxItems"]), (2, 2))
        self.assertEqual(result_schema["items"]["properties"]["verdict"]["enum"], ["met", "unmet"])
        image_parts = [part for part in request["messages"][1]["content"] if part["type"] == "image_url"]
        self.assertEqual(len(image_parts), 1)
        self.assertTrue(image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertNotIn("hidden-reference", json.dumps(request))

    def test_all_met_and_all_unmet(self):
        # 验证全部满足和全部不满足时过程分分别为 1 和 0。
        for verdicts, expected in [(("met", "met"), 1.0), (("unmet", "unmet"), 0.0)]:
            with self.subTest(verdicts=verdicts):

                async def transport(request, verdicts=verdicts):
                    return _body(_results(verdicts))

                judged = asyncio.run(self._judger(transport).judge(self.state.model_copy(deep=True)))
                self.assertEqual(judged.reward["score"], expected)

    def test_batch_judge(self):
        # 验证批量公开接口逐条返回对应的过程奖励。
        async def transport(request):
            return _body(_results())

        states = [self.state.model_copy(deep=True), self.state.model_copy(deep=True)]
        judged = asyncio.run(self._judger(transport).batch_judge(states))
        self.assertEqual([state.reward["score"] for state in judged], [0.25, 0.25])

    def test_retry_recovers_from_invalid_output(self):
        # 验证首次返回非法文本时重试，第二次合法结果正常计分。
        calls = 0

        async def transport(request):
            nonlocal calls
            calls += 1
            if calls == 1:
                return {"choices": [{"message": {"content": "not json"}}]}
            return _body(_results(("met", "met")))

        judged = asyncio.run(self._judger(transport, max_retries=1).judge(self.state))
        self.assertEqual(judged.reward["score"], 1.0)
        self.assertEqual(judged.reward["judge_attempts"], 2)

    def test_invalid_output_is_not_silent_zero(self):
        # 验证 judge 遗漏评分点时明确失败，不把服务异常静默记为 0 分。
        async def transport(request):
            return _body([_results()[0]])

        with self.assertRaisesRegex(RuntimeError, "Judge omitted rubric local_ids"):
            asyncio.run(self._judger(transport, max_retries=0).judge(self.state))

    def test_non_json_error_includes_content_prefix(self):
        # 验证结构化解析失败时保留有长度限制的原始响应摘要。
        async def transport(request):
            return _body("judge returned no JSON")

        with self.assertRaisesRegex(RuntimeError, "judge returned no JSON"):
            asyncio.run(self._judger(transport, max_retries=0).judge(self.state))

    def test_schema_fragment_without_outer_brace(self):
        # 验证服务缺少最外层左括号或左括号与起始引号时仍可解析。
        for prefix_length in (1, 2):
            with self.subTest(prefix_length=prefix_length):

                async def transport(request):
                    content = json.dumps({"results": _results()})[prefix_length:]
                    return {"choices": [{"message": {"content": content}}]}

                judged = asyncio.run(self._judger(transport, max_retries=0).judge(self.state))
                self.assertEqual(judged.reward["score"], 0.25)

    def test_invalid_met_evidence_fails_closed(self):
        # 验证 met 的证据不是回复原文时降级为 unmet，并保留可观测告警。
        async def transport(request):
            results = _results()
            results[0]["evidence"] = "Identifies the left grid."
            return _body(results)

        judged = asyncio.run(self._judger(transport, max_retries=0).judge(self.state))

        self.assertEqual(judged.reward["score"], 0.0)
        self.assertEqual(judged.reward["judge_status"], "ok_with_invalid_evidence")
        self.assertEqual(judged.reward["invalid_evidence_count"], 1)
        self.assertEqual(judged.reward["rubric_results"][0]["judge_verdict"], "met")
        self.assertEqual(judged.reward["rubric_results"][0]["verdict"], "unmet")

    def test_composed_real_outcome_and_fake_process(self):
        # 验证真实数学结果判分与外部过程判分经组合接口得到正确融合分。
        async def transport(request):
            return _body(_results(met_evidence="The result is"))

        judge = ComposedJudger(
            branches={
                "outcome": MathRuleJudger(kind="symbolic"),
                "process": self._judger(transport),
            },
            merge_fn=merge_outcome_and_process,
        )
        state = self.state.model_copy(deep=True)
        state.data_source = {"outcome": 1.0, "process": 1.0}
        state.reward_model = {"ground_truth": "0.09"}
        state.response = "The result is \\boxed{0.09}."
        state.finish_reason = "stop"

        judged = asyncio.run(judge.judge(state))

        self.assertEqual(judged.reward["outcome_score"], 1.0)
        self.assertEqual(judged.reward["process_score"], -0.5)
        self.assertAlmostEqual(judged.reward["score"], 0.55)

    def test_remote_contract_preserves_sample_rubrics(self):
        # 验证远程代理仍使用 RubricJudger 的预处理和样本级 rubric。
        local = self._judger(self._remote_transport)

        class RemoteMethod:
            async def remote(self, payload):
                return await local.judge_payload(payload)

        class Actor:
            judge_payload = RemoteMethod()

        remote = RemoteJudger(Actor(), "remote_rubric", preprocess_judger=local)
        judged = asyncio.run(remote.judge(self.state))
        self.assertEqual(judged.reward["score"], 0.25)

    async def _remote_transport(self, request):
        rubric_text = request["messages"][1]["content"][-1]["text"]
        self.assertIn('"local_id": "a1"', rubric_text)
        self.assertIn('"local_id": "a2"', rubric_text)
        return _body(_results())


class TestRubricInputValidation(RubricTestCase):
    """Reject malformed rubric metadata before calling the external judge."""

    def test_invalid_rubrics_fail_before_request(self):
        # 验证空评分点、非法权重和重复 ID 均在请求前失败。
        cases = [
            ({}, "non-empty extra_info.rubrics"),
            ({"rubrics": []}, "non-empty extra_info.rubrics"),
            ({"rubrics": [{**self.state.extra_fields["rubrics"][0], "weight": 0}]}, "positive weight"),
            (
                {
                    "rubrics": [
                        self.state.extra_fields["rubrics"][0],
                        self.state.extra_fields["rubrics"][0],
                    ]
                },
                "Duplicate rubric local_id",
            ),
        ]
        for extra_fields, message in cases:
            with self.subTest(message=message):
                state = self.state.model_copy(deep=True)
                state.extra_fields = extra_fields
                with self.assertRaisesRegex(ValueError, message):
                    self._judger(self._unused_transport).preprocess(state)

    def test_total_weight_mismatch_and_missing_visual_image(self):
        # 验证权重总数不一致及视觉评分点缺图时均明确拒绝。
        state = self.state.model_copy(deep=True)
        state.extra_fields["total_weight"] = 5
        with self.assertRaisesRegex(ValueError, "does not match rubric weight sum"):
            self._judger(self._unused_transport).preprocess(state)

        state = self.state.model_copy(deep=True)
        state.extra_fields["image_data"] = []
        state.message[0]["content"] = [{"type": "text", "text": "Question without image"}]
        with self.assertRaisesRegex(ValueError, "require at least one input image"):
            self._judger(self._unused_transport).preprocess(state)


class TestRubricRewardMerge(unittest.TestCase):
    """Exercise reward fusion and its compatibility with existing judger configs."""

    def test_single_merge_and_outcome_only_regression(self):
        # 验证融合公式、零过程权重回归和 GAE 终端奖励位置。
        state = RolloutState(message=[], response="answer")
        process = {"score": 0.25, "rubric_results": [{"local_id": "a1"}]}
        merged = merge_outcome_and_process(state, {"outcome": {"score": -1}, "process": process})
        self.assertAlmostEqual(merged.reward["score"], -0.85)
        self.assertEqual(merged.reward["outcome_score"], -1)
        self.assertEqual(merged.reward["process_score"], -0.5)

        state = RolloutState(message=[], response="answer")
        merged = merge_outcome_and_process(
            state,
            {"outcome": {"score": -1}, "process": process},
            process_weight=0,
        )
        self.assertEqual(merged.reward["score"], -1)

        token_rewards = terminal_rewards(
            torch.tensor([merged.reward["score"]]),
            torch.tensor([1, 1, 1]),
            torch.tensor([0, 3]),
        )
        self.assertEqual(token_rewards.tolist(), [0.0, 0.0, -1.0])

    def test_batch_merge_and_length_check(self):
        # 验证批量融合逐条对齐，分支数量不一致时拒绝错位合并。
        states = [RolloutState(message=[]), RolloutState(message=[])]
        judged = {
            "outcome": [{"score": 1}, {"score": -1}],
            "process": [{"score": 1}, {"score": 0}],
        }
        merged = merge_outcome_and_process(states, judged, process_weight=0.3)
        self.assertEqual([state.reward["score"] for state in merged], [1.0, -1.0])
        with self.assertRaisesRegex(ValueError, "output lengths"):
            merge_outcome_and_process(states, {"outcome": [{"score": 1}], "process": [{"score": 1}]})

    def test_config_composes_with_existing_judgers(self):
        # 验证 RubricJudgerConfig 可直接进入现有组合配置并保留默认重试次数。
        config = ComposedJudgerConfig(
            branches={
                "outcome": JudgerConfig(judger_name="outcome", reward_handler=lambda **kwargs: {"score": 1.0}),
                "process": RubricJudgerConfig(
                    judger_name="process",
                    endpoint="http://judge.test/v1/chat/completions",
                    model="GLM-5.3-Flash",
                ),
            }
        )
        process = config.branches["process"].build_local()
        self.assertIsInstance(process, RubricJudger)
        self.assertEqual(process.max_retries, 2)


if __name__ == "__main__":
    unittest.main()
