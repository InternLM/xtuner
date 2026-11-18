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


def _results(verdicts=("met", "unmet")):
    return [
        {"local_id": local_id, "verdict": verdict, "evidence": f"evidence for {local_id}"}
        for local_id, verdict in zip(("a1", "a2"), verdicts)
    ]


class TestRubricJudger(unittest.TestCase):
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

    def test_weighted_score_and_multimodal_request(self):
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
        for verdicts, expected in [(("met", "met"), 1.0), (("unmet", "unmet"), 0.0)]:
            with self.subTest(verdicts=verdicts):

                async def transport(request, verdicts=verdicts):
                    return _body(_results(verdicts))

                judged = asyncio.run(self._judger(transport).judge(self.state.model_copy(deep=True)))
                self.assertEqual(judged.reward["score"], expected)

    def test_batch_judge(self):
        async def transport(request):
            return _body(_results())

        states = [self.state.model_copy(deep=True), self.state.model_copy(deep=True)]
        judged = asyncio.run(self._judger(transport).batch_judge(states))
        self.assertEqual([state.reward["score"] for state in judged], [0.25, 0.25])

    def test_retry_recovers_from_invalid_output(self):
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
        async def transport(request):
            return _body([_results()[0]])

        with self.assertRaisesRegex(RuntimeError, "Judge omitted rubric local_ids"):
            asyncio.run(self._judger(transport, max_retries=0).judge(self.state))

    def test_composed_real_outcome_and_fake_process(self):
        async def transport(request):
            return _body(_results())

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
        self.assertEqual(judged.reward["process_score"], 0.25)
        self.assertAlmostEqual(judged.reward["score"], 0.775)

    def test_remote_contract_preserves_sample_rubrics(self):
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

    def test_invalid_rubrics_fail_before_request(self):
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
        state = self.state.model_copy(deep=True)
        state.extra_fields["total_weight"] = 5
        with self.assertRaisesRegex(ValueError, "does not match rubric weight sum"):
            self._judger(self._unused_transport).preprocess(state)

        state = self.state.model_copy(deep=True)
        state.extra_fields["image_data"] = []
        state.message[0]["content"] = [{"type": "text", "text": "Question without image"}]
        with self.assertRaisesRegex(ValueError, "require at least one input image"):
            self._judger(self._unused_transport).preprocess(state)

    async def _unused_transport(self, request):
        self.fail("Transport must not be called for invalid input")


class TestRubricRewardMerge(unittest.TestCase):
    def test_single_merge_and_outcome_only_regression(self):
        state = RolloutState(message=[], response="answer")
        process = {"score": 0.25, "rubric_results": [{"local_id": "a1"}]}
        merged = merge_outcome_and_process(state, {"outcome": {"score": -1}, "process": process})
        self.assertAlmostEqual(merged.reward["score"], -0.625)
        self.assertEqual(merged.reward["outcome_score"], -1)
        self.assertEqual(merged.reward["process_score"], 0.25)

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
        states = [RolloutState(message=[]), RolloutState(message=[])]
        judged = {
            "outcome": [{"score": 1}, {"score": -1}],
            "process": [{"score": 1}, {"score": 0}],
        }
        merged = merge_outcome_and_process(states, judged, process_weight=0.3)
        self.assertEqual([state.reward["score"] for state in merged], [1.0, -0.7])
        with self.assertRaisesRegex(ValueError, "output lengths"):
            merge_outcome_and_process(states, {"outcome": [{"score": 1}], "process": [{"score": 1}]})

    def test_config_composes_with_existing_judgers(self):
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
