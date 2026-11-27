"""Behavior tests for rubric judging through public interfaces.

TestRubricScoring
    test_weighted_score_and_multimodal_request
    test_all_met_and_all_unmet
    test_batch_judge
    test_retry_recovers_from_invalid_output
    test_invalid_output_is_not_silent_zero
    test_transport_and_http_errors_are_classified
    test_non_json_error_includes_content_prefix
    test_top_level_results_array_is_accepted_when_complete
    test_junk_prefixed_results_array_is_accepted
    test_results_object_missing_outer_braces
    test_blank_output_error_includes_sample_and_response_metadata
    test_truncated_output_error_is_classified
    test_failure_debug_dump_captures_raw_response
    test_schema_fragment_without_outer_brace
    test_invalid_met_evidence_fails_closed
    test_overlong_met_evidence_fails_closed
    test_control_char_evidence_parses_and_fails_closed
    test_prompt_and_schema_cap_evidence_length
    test_composed_real_outcome_and_fake_process
    test_remote_contract_preserves_sample_rubrics
    test_question_uses_first_user_message
    test_batch_respects_concurrency_limit
    test_chat_template_kwargs_configurable
    test_agent_tool_images_merged_and_deduplicated
    test_chunked_rubric_requests_and_usage_aggregation
    test_truncated_mid_json_recovers_partial_and_continues
    test_on_failure_zero_degrades_without_raising
    test_dodged_bullet_scoring_evidence_guard_and_all_pass
TestRubricInputValidation
    test_invalid_rubrics_fail_before_request
    test_total_weight_mismatch_and_missing_visual_image
    test_dodged_bullet_validation_and_total_weight_compat
TestRubricRewardMerge
    test_single_merge_and_outcome_only_regression
    test_batch_merge_and_length_check
    test_config_composes_with_existing_judgers
"""

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path

import httpx
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


def _body(results, finish_reason="stop", usage=None, wrap_results=True):
    payload = {"results": results} if wrap_results else results
    content = results if isinstance(results, str) else json.dumps(payload)
    return {
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
        "usage": usage or {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    }


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
            group_id=3,
            rollout_id=7,
            task_name="rubric_test",
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
        self.assertEqual(judged.reward["judge_finish_reason"], "stop")
        self.assertEqual(judged.reward["judge_usage"]["completion_tokens"], 20)
        self.assertEqual(judged.reward["judge_attempts"], 1)
        self.assertGreaterEqual(judged.reward["judge_latency_seconds"], 0.0)
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

        with self.assertRaises(RuntimeError) as raised:
            asyncio.run(self._judger(transport, max_retries=0).judge(self.state))
        self.assertIn("Judge omitted rubric local_ids", str(raised.exception))
        self.assertIn('"error_type": "protocol"', str(raised.exception))

    def test_transport_and_http_errors_are_classified(self):
        # 验证网络超时与 HTTP 状态错误分为 transport 和 http，均不生成过程奖励。
        async def timeout_transport(request):
            raise httpx.ReadTimeout("read timeout")

        with self.assertRaises(RuntimeError) as raised:
            asyncio.run(self._judger(timeout_transport, max_retries=0).judge(self.state))
        self.assertIn('"error_type": "transport"', str(raised.exception))

        async def http_error_transport(request):
            http_request = httpx.Request("POST", "http://judge.test/v1/chat/completions")
            response = httpx.Response(503, request=http_request)
            raise httpx.HTTPStatusError("service unavailable", request=http_request, response=response)

        with self.assertRaises(RuntimeError) as raised:
            asyncio.run(self._judger(http_error_transport, max_retries=0).judge(self.state))
        self.assertIn('"error_type": "http"', str(raised.exception))

    def test_non_json_error_includes_content_prefix(self):
        # 验证结构化解析失败时保留有长度限制的原始响应摘要。
        async def transport(request):
            return _body("judge returned no JSON")

        with self.assertRaisesRegex(RuntimeError, "judge returned no JSON"):
            asyncio.run(self._judger(transport, max_retries=0).judge(self.state))

    def test_top_level_results_array_is_accepted_when_complete(self):
        # 验证 judge 返回裸 results 数组时仍按同一套 ID、verdict 和证据规则校验。
        async def transport(request):
            return _body(_results(), wrap_results=False)

        judged = asyncio.run(self._judger(transport, max_retries=0).judge(self.state))
        self.assertEqual(judged.reward["score"], 0.25)
        self.assertEqual(judged.reward["judge_status"], "ok")

    def test_junk_prefixed_results_array_is_accepted(self):
        # 验证裸数组前带垃圾前缀（长输入实测为 ":\n"）时仍按完整数组解析并校验。
        async def transport(request):
            return _body(":\n" + json.dumps(_results()))

        judged = asyncio.run(self._judger(transport, max_retries=0).judge(self.state))
        self.assertEqual(judged.reward["score"], 0.25)
        self.assertEqual(judged.reward["judge_status"], "ok")

    def test_results_object_missing_outer_braces(self):
        # 验证 results 键值完好但成对外层花括号缺失时可修复（长输入实测形态）。
        async def transport(request):
            content = json.dumps({"results": _results()})
            return _body(content[1:-1])

        judged = asyncio.run(self._judger(transport, max_retries=0).judge(self.state))
        self.assertEqual(judged.reward["score"], 0.25)
        self.assertEqual(judged.reward["judge_status"], "ok")

    def test_blank_output_error_includes_sample_and_response_metadata(self):
        # 验证空白输出保留样本身份、结束原因、token 用量和内容长度，且不发放过程奖励。
        async def transport(request):
            return _body(
                "  \n  \n",
                usage={"prompt_tokens": 8000, "completion_tokens": 1024, "total_tokens": 9024},
            )

        with self.assertRaises(RuntimeError) as raised:
            asyncio.run(self._judger(transport, max_retries=0).judge(self.state))

        message = str(raised.exception)
        self.assertIn('"error_type": "model_output"', message)
        self.assertIn('"rollout_id": 7', message)
        self.assertIn('"group_id": 3', message)
        self.assertIn('"task_name": "rubric_test"', message)
        self.assertIn('"finish_reason": "stop"', message)
        self.assertIn('"completion_tokens": 1024', message)
        self.assertIn('"content_chars": 6', message)

    def test_truncated_output_error_is_classified(self):
        # 验证 max_tokens 截断与普通协议错误可区分，并保留输出用量。
        async def transport(request):
            return _body(
                '{"results":[',
                finish_reason="length",
                usage={"prompt_tokens": 8000, "completion_tokens": 1024, "total_tokens": 9024},
            )

        with self.assertRaises(RuntimeError) as raised:
            asyncio.run(self._judger(transport, max_retries=0).judge(self.state))

        message = str(raised.exception)
        self.assertIn('"error_type": "output_truncated"', message)
        self.assertIn('"finish_reason": "length"', message)
        self.assertIn('"completion_tokens": 1024', message)

    def test_failure_debug_dump_captures_raw_response(self):
        # 验证设置 RUBRIC_JUDGE_DEBUG_DIR 后失败样本落盘完整原始响应，成功样本不落盘。
        async def bad_transport(request):
            return _body('{"results":[', finish_reason="length")

        with tempfile.TemporaryDirectory() as debug_dir:
            os.environ["RUBRIC_JUDGE_DEBUG_DIR"] = debug_dir
            try:
                with self.assertRaises(RuntimeError) as raised:
                    asyncio.run(self._judger(bad_transport, max_retries=0).judge(self.state))
                dumps = list(Path(debug_dir).glob("*.json"))
                self.assertEqual(len(dumps), 1)
                record = json.loads(dumps[0].read_text())
                self.assertEqual(record["raw_response"]["choices"][0]["message"]["content"], '{"results":[')
                self.assertEqual(record["payload"]["rollout_id"], 7)
                self.assertIn("debug_dump", str(raised.exception))

                async def ok_transport(request):
                    return _body(_results())

                asyncio.run(self._judger(ok_transport).judge(self.state))
                self.assertEqual(len(list(Path(debug_dir).glob("*.json"))), 1)
            finally:
                del os.environ["RUBRIC_JUDGE_DEBUG_DIR"]

    def test_schema_fragment_without_outer_brace(self):
        # 验证 results 标记在开头丢失任意数量字符时均可按已知结构修复。
        for prefix_length in range(1, len('{"results"')):
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

    def test_overlong_met_evidence_fails_closed(self):
        # 验证 met 的证据虽是原文但超过长度上限时同样降级为 unmet，
        # 防止后端不强制 schema maxLength 时长证据挤占输出预算。
        long_quote = "The left ratio is 2/9 and the right ratio is 1/4. " * 8
        self.state.response = long_quote

        async def transport(request):
            results = _results(met_evidence=long_quote[:300])
            results[1]["verdict"] = "met"
            results[1]["evidence"] = "The left ratio is 2/9"
            return _body(results)

        judged = asyncio.run(self._judger(transport, max_retries=0).judge(self.state))

        self.assertEqual(judged.reward["score"], 0.75)
        self.assertEqual(judged.reward["invalid_evidence_count"], 1)
        overlong = judged.reward["rubric_results"][0]
        self.assertEqual(overlong["judge_verdict"], "met")
        self.assertEqual(overlong["verdict"], "unmet")
        self.assertIn("exceeds", overlong["validation_error"])

    def test_control_char_evidence_parses_and_fails_closed(self):
        # 验证 judge 引用回复中的 LaTeX 时输出的字面控制字符（实测 \boldsymbol ->
        # \x08oldsymbol）不会使整份 JSON 解析失败：宽松解析后证据无法逐字匹配回复，
        # 该评分点 fail-closed 为 unmet，其余评分点正常计分。
        content = (
            '{\n"results": [\n'
            '{\n"local_id": "a1",\n"verdict": "met",\n'
            '"evidence": "The text says $B = \x08oldsymbol{\x08igemptyset}$."\n},\n'
            '{"local_id": "a2", "verdict": "met", "evidence": "The left ratio is 2/9"}\n]\n}'
        )

        async def transport(request):
            return _body(content)

        judged = asyncio.run(self._judger(transport, max_retries=0).judge(self.state))

        self.assertEqual(judged.reward["score"], 0.75)
        self.assertEqual(judged.reward["judge_status"], "ok_with_invalid_evidence")
        self.assertEqual(judged.reward["invalid_evidence_count"], 1)
        corrupted = judged.reward["rubric_results"][0]
        self.assertEqual(corrupted["judge_verdict"], "met")
        self.assertEqual(corrupted["verdict"], "unmet")

    def test_prompt_and_schema_cap_evidence_length(self):
        # 验证 evidence 长度上限同时落在 system prompt 和请求 schema 中。
        requests = []

        async def transport(request):
            requests.append(request)
            return _body(_results())

        asyncio.run(self._judger(transport).judge(self.state))

        request = requests[0]
        self.assertIn("50 words", request["messages"][0]["content"])
        evidence_schema = request["response_format"]["json_schema"]["schema"]["properties"]["results"]["items"][
            "properties"
        ]["evidence"]
        self.assertEqual(evidence_schema["maxLength"], 256)

    def test_lenient_evidence_accepts_paraphrase_prefix(self):
        # Qwen judge 常输出 "The response states that <原文>"；lenient 模式剥前缀后应判 met。
        async def transport(request):
            return _body(
                [
                    {"local_id": "a1", "verdict": "met", "evidence": "The left ratio is 2/9"},
                    {
                        "local_id": "a2",
                        "verdict": "met",
                        "evidence": "The response states that the left ratio is 2/9 and the right ratio is 1/4.",
                    },
                ]
            )

        strict = asyncio.run(self._judger(transport).judge(self.state))
        self.assertEqual(strict.reward["score"], 0.25)
        self.assertEqual(strict.reward["invalid_evidence_count"], 1)

        lenient = asyncio.run(self._judger(transport, evidence_match_mode="lenient").judge(self.state))
        self.assertEqual(lenient.reward["score"], 1.0)
        self.assertEqual(lenient.reward["invalid_evidence_count"], 0)
        a2 = lenient.reward["rubric_results"][1]
        self.assertEqual(a2["verdict"], "met")
        self.assertIn("The response states that", a2["judge_evidence"])
        self.assertIn("left ratio is 2/9 and the right ratio is 1/4.", a2["evidence"])

    def test_lenient_evidence_rejects_fabricated_quote(self):
        # 重叠不足的编造证据仍应 fail-closed。
        async def transport(request):
            return _body(
                [
                    {"local_id": "a1", "verdict": "met", "evidence": "The left ratio is 2/9"},
                    {
                        "local_id": "a2",
                        "verdict": "met",
                        "evidence": "The response states that the moon is made of cheese and every ratio equals seven elevenths.",
                    },
                ]
            )

        judged = asyncio.run(self._judger(transport, evidence_match_mode="lenient").judge(self.state))
        self.assertEqual(judged.reward["invalid_evidence_count"], 1)
        self.assertEqual(judged.reward["rubric_results"][1]["verdict"], "unmet")

    def test_lenient_mode_keeps_visual_grounding_strict(self):
        # 页图断言不允许转述证据，即使 lenient 模式也保持严格。
        async def transport(request):
            return _body(
                [
                    {
                        "local_id": "a1",
                        "verdict": "met",
                        "evidence": "The response states that the left ratio is 2/9.",
                    },
                    {
                        "local_id": "a2",
                        "verdict": "met",
                        "evidence": "The left ratio is 2/9 and the right ratio is 1/4.",
                    },
                ]
            )

        judged = asyncio.run(self._judger(transport, evidence_match_mode="lenient").judge(self.state))
        self.assertEqual(judged.reward["rubric_results"][0]["verdict"], "unmet")
        self.assertEqual(judged.reward["rubric_results"][1]["verdict"], "met")

    def test_evidence_match_config_validation(self):
        with self.assertRaises(ValueError):
            self._judger(self._unused_transport, evidence_match_mode="loose")
        with self.assertRaises(ValueError):
            self._judger(self._unused_transport, evidence_min_overlap=0)

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

    def test_question_uses_first_user_message(self):
        # 验证 agentic 多轮轨迹下判分上下文只取首个 user 消息，后续 user 角色的
        # 工具输出/观察结果不污染 judge 输入；reasoning RL 单轮行为不变。
        requests = []

        async def transport(request):
            requests.append(request)
            return _body(_results())

        state = self.state.model_copy(deep=True)
        state.message = [
            {"role": "user", "content": "Read the PDF and answer: what is the ratio?"},
            {"role": "assistant", "content": "<tool_call>get_page_text ...</tool_call>"},
            {"role": "user", "content": "<tool_response>page-1 text ...</tool_response>"},
            {"role": "assistant", "content": "The left ratio is 2/9 and the right ratio is 1/4."},
        ]

        judged = asyncio.run(self._judger(transport, max_retries=0).judge(state))

        self.assertEqual(judged.reward["score"], 0.25)
        question_text = requests[0]["messages"][1]["content"][0]["text"]
        self.assertIn("Read the PDF and answer", question_text)
        self.assertNotIn("tool_response", question_text)

    def test_batch_respects_concurrency_limit(self):
        # 验证批量判分的并发度不超过 max_concurrent_requests。
        in_flight = 0
        max_in_flight = 0

        async def transport(request):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return _body(_results())

        states = [self.state.model_copy(deep=True) for _ in range(4)]
        judged = asyncio.run(self._judger(transport, max_concurrent_requests=2).batch_judge(states))

        self.assertEqual([state.reward["score"] for state in judged], [0.25] * 4)
        self.assertLessEqual(max_in_flight, 2)

    def test_chat_template_kwargs_configurable(self):
        # 验证 chat_template_kwargs 默认沿用 GLM 值、可覆盖、传 {} 时整体省略。
        requests = []

        async def transport(request):
            requests.append(request)
            return _body(_results())

        asyncio.run(self._judger(transport).judge(self.state.model_copy(deep=True)))
        self.assertEqual(
            requests[-1]["chat_template_kwargs"],
            {"reasoning_effort": "low", "clear_thinking": False},
        )

        asyncio.run(
            self._judger(transport, chat_template_kwargs={"reasoning_effort": "high"}).judge(
                self.state.model_copy(deep=True)
            )
        )
        self.assertEqual(requests[-1]["chat_template_kwargs"], {"reasoning_effort": "high"})

        asyncio.run(self._judger(transport, chat_template_kwargs={}).judge(self.state.model_copy(deep=True)))
        self.assertNotIn("chat_template_kwargs", requests[-1])

    def test_agent_tool_images_merged_and_deduplicated(self):
        # 验证 agentic 工具返回的 get_page_image 页图透传进 judge 请求，且与 image_data 保序去重。
        requests = []

        async def transport(request):
            requests.append(request)
            return _body(_results(("met", "met")))

        state = self.state.model_copy(deep=True)
        tool_uri_1 = "data:image/png;base64,dG9vbC1wYWdlLTE="
        tool_uri_2 = "data:image/png;base64,dG9vbC1wYWdlLTI="
        state.extra_fields["agent_tool_images"] = [
            {"page": "1", "image_data": str(self.image_path)},  # 与首轮 image_data 重复
            {"page": "2", "image_data": tool_uri_1},
            {"page": "2", "image_data": tool_uri_1},  # 多轮重复取同页
            tool_uri_2,  # 兼容纯字符串条目
        ]

        judged = asyncio.run(self._judger(transport, max_retries=0).judge(state))
        self.assertEqual(judged.reward["score"], 1.0)
        image_urls = [
            part["image_url"]["url"] for part in requests[0]["messages"][1]["content"] if part["type"] == "image_url"
        ]
        self.assertEqual(len(image_urls), 3)
        self.assertTrue(image_urls[0].startswith("data:image/png;base64,"))
        self.assertEqual(image_urls[1:], [tool_uri_1, tool_uri_2])

        # 首轮无内联图、仅靠 agent_tool_images 供图时 visual_grounding 也应通过校验
        tool_only_state = self.state.model_copy(deep=True)
        tool_only_state.extra_fields["image_data"] = []
        tool_only_state.message = [{"role": "user", "content": "Inspect page 1 with get_page_image."}]
        tool_only_state.extra_fields["agent_tool_images"] = [{"page": "1", "image_data": tool_uri_1}]
        payload = self._judger(self._unused_transport).preprocess(tool_only_state)
        self.assertEqual(payload["image_sources"], [tool_uri_1])

    def test_chunked_rubric_requests_and_usage_aggregation(self):
        # 验证多 rubric 样本按 max_rubrics_per_request 分批请求，避免单次输出撑爆 max_tokens。
        state = self.state.model_copy(deep=True)
        state.response = "Part c1 ok. Part c2 ok. Part c3 ok. Part c4 ok. Part c5 ok."
        rubrics = [
            {"local_id": f"c{i}", "type": "answer_coverage", "criterion": f"Check c{i}", "weight": 1.0}
            for i in range(1, 6)
        ]
        state.extra_fields = {
            "rubrics": rubrics,
            "total_weight": 5.0,
            "image_data": [str(self.image_path)],
        }
        seen_batches = []

        async def transport(request):
            enum_ids = request["response_format"]["json_schema"]["schema"]["properties"]["results"]["items"][
                "properties"
            ]["local_id"]["enum"]
            seen_batches.append(enum_ids)
            return _body(
                [
                    {
                        "local_id": cid,
                        "verdict": "met" if cid in {"c1", "c3", "c5"} else "unmet",
                        "evidence": f"Part {cid} ok." if cid in {"c1", "c3", "c5"} else f"missing {cid}",
                    }
                    for cid in enum_ids
                ],
                usage={"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
            )

        judged = asyncio.run(self._judger(transport, max_rubrics_per_request=2, max_retries=0).judge(state))
        self.assertEqual(seen_batches, [["c1", "c2"], ["c3", "c4"], ["c5"]])
        self.assertAlmostEqual(judged.reward["score"], 0.6)
        self.assertFalse(judged.reward["all_pass"])
        self.assertEqual(judged.reward["judge_attempts"], 3)
        self.assertEqual(
            judged.reward["judge_usage"], {"prompt_tokens": 150, "completion_tokens": 30, "total_tokens": 180}
        )

    def test_truncated_mid_json_recovers_partial_and_continues(self):
        # 复现 20261126g 真实截断形态：首轮 finish_reason=length 在第 2 条中间截断，
        # 判分器回收已闭合的 c1 并仅对剩余 c2..c3 发起续判，0 重试浪费完成判分。
        state = self.state.model_copy(deep=True)
        state.response = "Total Rolls needed: 4. Grease amount is 12g."
        state.extra_fields = {
            "rubrics": [
                {"local_id": "c1", "type": "answer_coverage", "criterion": "First op rolls = 6", "weight": 1.0},
                {"local_id": "c2", "type": "answer_coverage", "criterion": "Grease = 12g", "weight": 1.0},
                {"local_id": "c3", "type": "dodged_bullet", "criterion": "Avoid modifying shims", "weight": 0.6},
            ],
            "total_weight": 2.0,
            "image_data": [str(self.image_path)],
        }
        seen_enums = []

        async def transport(request):
            enum_ids = request["response_format"]["json_schema"]["schema"]["properties"]["results"]["items"][
                "properties"
            ]["local_id"]["enum"]
            seen_enums.append(enum_ids)
            if len(seen_enums) == 1:
                truncated = (
                    '{"results":[{"local_id":"c1","verdict":"unmet","evidence":'
                    "\"The response states: 'Total Rolls needed: 4.' It fails to identify the number "
                    'of First Operation seaming rolls required as 6."},{"local_id":"'
                )
                return _body(
                    truncated,
                    finish_reason="length",
                    usage={"prompt_tokens": 8638, "completion_tokens": 4096, "total_tokens": 12734},
                )
            return _body(
                [
                    {"local_id": "c2", "verdict": "met", "evidence": "Grease amount is 12g."},
                    {"local_id": "c3", "verdict": "met", "evidence": "Avoided."},
                ],
                finish_reason="stop",
                usage={"prompt_tokens": 8500, "completion_tokens": 80, "total_tokens": 8580},
            )

        judged = asyncio.run(self._judger(transport, max_rubrics_per_request=None, max_retries=0).judge(state))
        self.assertEqual(seen_enums, [["c1", "c2", "c3"], ["c2", "c3"]])
        self.assertAlmostEqual(judged.reward["score"], 0.5)
        self.assertFalse(judged.reward["all_pass"])
        self.assertFalse(judged.reward["judge_failed"])
        self.assertEqual(judged.reward["judge_attempts"], 2)
        self.assertEqual(judged.reward["judge_usage"]["completion_tokens"], 4176)

    def test_on_failure_zero_degrades_without_raising(self):
        # 验证长窗口训练开启 on_failure="zero" 时重试耗尽返回 0 分并记录 judge_failed 与 error_type，不崩训练。
        async def transport(request):
            return _body('{"results":[', finish_reason="length")

        cfg = RubricJudgerConfig(
            judger_name="process",
            endpoint="http://judge.test/v1/chat/completions",
            model="Qwen3.5-9B",
            max_retries=1,
            retry_backoff=0,
            max_tokens=2048,
            max_evidence_chars=200,
            max_rubrics_per_request=6,
            on_failure="zero",
        )
        judger = cfg.build_local()
        judger._transport = transport
        state = self.state.model_copy(deep=True)
        judged = asyncio.run(judger.judge(state))

        self.assertEqual(judged.reward["score"], 0.0)
        self.assertEqual(judged.reward["process_score"], 0.0)
        self.assertTrue(judged.reward["judge_failed"])
        self.assertEqual(judged.reward["error_type"], "output_truncated")
        self.assertEqual(judged.reward["judge_status"], "failed")
        self.assertEqual(judged.reward["judge_attempts"], 2)
        self.assertFalse(judged.reward["all_pass"])
        self.assertTrue(judged.extra_fields["judge_failed"])
        self.assertEqual(judged.extra_fields["error_type"], "output_truncated")

    def test_dodged_bullet_scoring_evidence_guard_and_all_pass(self):
        # 验证 dodged_bullet 口径：
        # 1) PI 全错 + DB 全守 -> score = 0.0（不靠"没犯错"拿正分，且 DB met 不因非子串证据被误杀）；
        # 2) PI 全对 + DB 全守 -> score = 1.0, all_pass = True；
        # 3) PI 得分 - DB 踩坑扣分 -> max(0, pos - neg) / pos_sum，且 DB 踩坑不因概括性证据被漏扣分。
        base_state = self.state.model_copy(deep=True)
        base_state.response = "The ratio is 2/9 and recommended drug is Heparin."
        base_state.extra_fields = {
            "rubrics": [
                {"local_id": "p1", "type": "answer_coverage", "criterion": "States ratio 2/9", "weight": 1.0},
                {"local_id": "p2", "type": "answer_coverage", "criterion": "Recommends Heparin", "weight": 0.6},
                {
                    "local_id": "d1",
                    "type": "dodged_bullet",
                    "criterion": "The response should avoid recommending Warfarin.",
                    "weight": 0.6,
                },
                {
                    "local_id": "d2",
                    "type": "dodged_bullet",
                    "criterion": "The response should avoid citing external guidelines.",
                    "weight": 0.6,
                },
            ],
            "total_weight": 1.6,
            "image_data": [str(self.image_path)],
        }

        # Case 1: PI 全错 + DB 全守（非原文摘要证据 "Avoided."）-> score = 0.0, invalid_evidence_count = 0
        async def pi_wrong_db_kept(request):
            return _body(
                [
                    {"local_id": "p1", "verdict": "unmet", "evidence": "Missing ratio."},
                    {"local_id": "p2", "verdict": "unmet", "evidence": "Missing drug."},
                    {"local_id": "d1", "verdict": "met", "evidence": "Avoided recommending Warfarin."},
                    {"local_id": "d2", "verdict": "met", "evidence": "Avoided."},
                ]
            )

        s1 = asyncio.run(self._judger(pi_wrong_db_kept, max_retries=0).judge(base_state.model_copy(deep=True)))
        self.assertEqual(s1.reward["score"], 0.0)
        self.assertFalse(s1.reward["all_pass"])
        self.assertEqual(s1.reward["invalid_evidence_count"], 0)

        # Case 2: PI 全对 + DB 全守 -> score = 1.0, all_pass = True
        async def all_good(request):
            return _body(
                [
                    {"local_id": "p1", "verdict": "met", "evidence": "The ratio is 2/9"},
                    {"local_id": "p2", "verdict": "met", "evidence": "recommended drug is Heparin."},
                    {"local_id": "d1", "verdict": "met", "evidence": "Avoided."},
                    {"local_id": "d2", "verdict": "met", "evidence": "Avoided."},
                ]
            )

        s2 = asyncio.run(self._judger(all_good, max_retries=0).judge(base_state.model_copy(deep=True)))
        self.assertEqual(s2.reward["score"], 1.0)
        self.assertTrue(s2.reward["all_pass"])
        self.assertTrue(s2.extra_fields["all_pass"])

        # Case 3: p1 met (1.0), p2 unmet (0.0), d1 violated (-0.6, 概括性证据不漏扣), d2 met -> (1.0 - 0.6)/1.6 = 0.25
        async def one_violation(request):
            return _body(
                [
                    {"local_id": "p1", "verdict": "met", "evidence": "The ratio is 2/9"},
                    {"local_id": "p2", "verdict": "unmet", "evidence": "Missing Heparin."},
                    {"local_id": "d1", "verdict": "unmet", "evidence": "Response recommends Warfarin instead."},
                    {"local_id": "d2", "verdict": "met", "evidence": "Avoided."},
                ]
            )

        s3 = asyncio.run(self._judger(one_violation, max_retries=0).judge(base_state.model_copy(deep=True)))
        self.assertAlmostEqual(s3.reward["score"], 0.25)
        self.assertFalse(s3.reward["all_pass"])

        # Case 4: 扣分超过正分时截断到 0.0（p2 met=0.6, d1+d2 均违规=-1.2 -> max(0, 0.6-1.2)/1.6 = 0.0）
        async def clamped_zero(request):
            return _body(
                [
                    {"local_id": "p1", "verdict": "unmet", "evidence": "Missing ratio."},
                    {"local_id": "p2", "verdict": "met", "evidence": "recommended drug is Heparin."},
                    {"local_id": "d1", "verdict": "unmet", "evidence": "Violated d1."},
                    {"local_id": "d2", "verdict": "unmet", "evidence": "Violated d2."},
                ]
            )

        s4 = asyncio.run(self._judger(clamped_zero, max_retries=0).judge(base_state.model_copy(deep=True)))
        self.assertEqual(s4.reward["score"], 0.0)
        self.assertFalse(s4.reward["all_pass"])


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

    def test_dodged_bullet_validation_and_total_weight_compat(self):
        # 验证含 dodged_bullet 时：至少需要一条正向 rubric；total_weight 同时兼容正向分母与全权重和。
        only_db = self.state.model_copy(deep=True)
        only_db.extra_fields = {
            "rubrics": [{"local_id": "d1", "type": "dodged_bullet", "criterion": "Avoid X", "weight": 0.6}]
        }
        with self.assertRaisesRegex(ValueError, "at least one positive rubric"):
            self._judger(self._unused_transport).preprocess(only_db)

        mixed_rubrics = [
            {"local_id": "p1", "type": "answer_coverage", "criterion": "State X", "weight": 1.0},
            {"local_id": "d1", "type": "dodged_bullet", "criterion": "Avoid Y", "weight": 0.6},
        ]
        for valid_total in (None, 1.0, 1.6):
            state = self.state.model_copy(deep=True)
            state.extra_fields = {"rubrics": mixed_rubrics, "total_weight": valid_total}
            self.assertEqual(len(self._judger(self._unused_transport).preprocess(state)["rubrics"]), 2)

        bad = self.state.model_copy(deep=True)
        bad.extra_fields = {"rubrics": mixed_rubrics, "total_weight": 2.5}
        with self.assertRaisesRegex(ValueError, "does not match rubric weight sum"):
            self._judger(self._unused_transport).preprocess(bad)


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
