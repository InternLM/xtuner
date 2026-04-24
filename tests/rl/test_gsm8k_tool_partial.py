import re
import unittest

from xtuner.v1.data_proto import RolloutState, SampleParams, Status
from xtuner.v1.rl.agent_loop.gsm8k_with_tool import GSM8KToolAgentLoop
from xtuner.v1.rl.agent_loop.utils import PartialRolloutHandler


def _encode(text: str) -> list[int]:
    return [ord(char) for char in text]


class _FakeTokenizer:
    def __init__(self) -> None:
        self.tool_messages: list[list[dict]] = []

    def decode(self, ids: list[int] | None) -> str:
        return "".join(chr(token_id) for token_id in ids or [])

    def apply_chat_template(self, messages: list[dict], remove_system_prompt: bool = True) -> list[int]:
        self.tool_messages.append(messages)
        content = "".join(message["content"] for message in messages)
        return _encode(f"<tool>{content}</tool>")


class _FakeGenerate:
    def __init__(self, responses: list[tuple[str, Status]]) -> None:
        self.responses = list(responses)
        self.requests: list[dict] = []

    async def remote(self, rollout_state: RolloutState) -> RolloutState:
        self.requests.append(
            {
                "tokens": list(rollout_state.tokens or []),
                "max_tokens": rollout_state.sample_params.max_tokens,
            }
        )
        text, status = self.responses.pop(0)
        rollout_state.response = text
        rollout_state.response_ids = _encode(text)
        rollout_state.logprobs = [0.0] * len(rollout_state.response_ids)
        rollout_state.status = status
        rollout_state.finish_reason = "stop" if status == Status.COMPLETED else "abort"
        return rollout_state


class _FakeRolloutController:
    def __init__(self, responses: list[tuple[str, Status]]) -> None:
        self.generate = _FakeGenerate(responses)


class _FakeJudger:
    def __init__(self) -> None:
        self.calls = 0

    async def judge(self, rollout_state: RolloutState) -> RolloutState:
        self.calls += 1
        rollout_state.reward = {"score": 1.0}
        return rollout_state


class TestGSM8KToolPartialRollout(unittest.IsolatedAsyncioTestCase):
    def _build_loop(
        self,
        responses: list[tuple[str, Status]],
        *,
        max_tokens: int = 16,
        max_turns: int = 2,
        judger: _FakeJudger | None = None,
    ) -> GSM8KToolAgentLoop:
        loop = GSM8KToolAgentLoop.__new__(GSM8KToolAgentLoop)
        loop.max_turns = max_turns
        loop.rollout_ctl = _FakeRolloutController(responses)
        loop.sample_params = SampleParams(max_tokens=max_tokens)
        loop.max_tokens = max_tokens
        loop.partial_rollout_handler = PartialRolloutHandler(max_tokens=max_tokens)
        loop.tokenizer = _FakeTokenizer()
        loop.judger = judger
        loop.tool_call_pattern = re.compile(r"\n*<tool_call>(.*?)</tool_call>", re.DOTALL)
        loop.tool_call_start_token = "<tool_call>"
        loop.tool_call_end_token = "</tool_call>"
        loop.calc_gsm8k_reward = lambda answer, ground_truth: 1.0
        return loop

    def _make_aborted_state(self, response_text: str, *, max_tokens: int = 16) -> RolloutState:
        response_ids = _encode(response_text)
        return RolloutState(
            uid=1,
            message=[{"role": "user", "content": "question"}],
            prompt_ids=[7],
            sample_params=SampleParams(max_tokens=max_tokens),
            status=Status.ABORTED,
            response_ids=response_ids,
            response=response_text,
            logprobs=[0.0] * len(response_ids),
            response_mask=[1] * len(response_ids),
            response_rollout_steps=[1] * len(response_ids),
            reward_model={"ground_truth": "#### 42"},
            extra_fields={},
        )

    async def test_partial_rollout_appends_history_and_updates_staleness(self):
        judger = _FakeJudger()
        loop = self._build_loop([("c", Status.COMPLETED)], max_tokens=5, judger=judger)
        state = self._make_aborted_state("ab", max_tokens=5)

        result = await loop.generate_sample(state, enable_partial_rollout=True, rollout_step=3)

        self.assertEqual(loop.rollout_ctl.generate.requests[0]["tokens"], [7] + _encode("ab"))
        self.assertEqual(loop.rollout_ctl.generate.requests[0]["max_tokens"], 3)
        self.assertEqual(loop.tokenizer.decode(result.response_ids), "abc")
        self.assertEqual(result.response_mask, [1, 1, 1])
        self.assertEqual(result.response_rollout_steps, [1, 1, 3])
        self.assertEqual(result.seq_staleness, 2)
        tool_state = result.extra_fields["gsm8k_tool_agent_loop_state"]
        self.assertEqual(tool_state["cur_turn"], 1)
        self.assertEqual(tool_state["current_turn_response_start_idx"], 0)
        self.assertEqual(judger.calls, 1)

    async def test_non_partial_path_does_not_write_partial_state(self):
        first_turn = '<tool_call>{"name":"calc_gsm8k_reward","arguments":{"answer":"42"}}</tool_call>'
        loop = self._build_loop([(first_turn, Status.COMPLETED), ("done", Status.COMPLETED)], max_tokens=128)
        state = RolloutState(
            uid=2,
            message=[{"role": "user", "content": "question"}],
            prompt_ids=[7],
            sample_params=SampleParams(max_tokens=128),
            reward_model={"ground_truth": "#### 42"},
            extra_fields={},
        )

        result = await loop.generate_sample(state)

        self.assertNotIn("gsm8k_tool_agent_loop_state", result.extra_fields)
        self.assertEqual(len(loop.tokenizer.tool_messages), 1)
        self.assertIn("done", loop.tokenizer.decode(result.response_ids))

    async def test_aborted_partial_rollout_is_not_judged(self):
        judger = _FakeJudger()
        loop = self._build_loop([("c", Status.ABORTED)], max_tokens=5, judger=judger)
        state = self._make_aborted_state("ab", max_tokens=5)

        result = await loop.generate_sample(state, enable_partial_rollout=True, rollout_step=3)

        self.assertEqual(result.status, Status.ABORTED)
        self.assertEqual(loop.tokenizer.decode(result.response_ids), "abc")
        self.assertEqual(result.response_rollout_steps, [1, 1, 3])
        tool_state = result.extra_fields["gsm8k_tool_agent_loop_state"]
        self.assertEqual(tool_state["cur_turn"], 0)
        self.assertEqual(tool_state["current_turn_response_start_idx"], 0)
        self.assertEqual(judger.calls, 0)

    async def test_tool_call_can_complete_across_partial_boundary(self):
        history = '<tool_call>{"name":"calc_gsm8k_reward","arguments":{"answer":"42"}}'
        loop = self._build_loop([("</tool_call>", Status.COMPLETED)], max_tokens=128, max_turns=1)
        state = self._make_aborted_state(history, max_tokens=128)

        result = await loop.generate_sample(state, enable_partial_rollout=True, rollout_step=2)

        decoded = loop.tokenizer.decode(result.response_ids)
        self.assertIn(history + "</tool_call>", decoded)
        self.assertIn("<tool>", decoded)
        self.assertEqual(len(loop.tokenizer.tool_messages), 1)
        tool_response_ids = _encode('<tool>{"result": 1.0}</tool>')
        self.assertEqual(result.response_mask[-len(tool_response_ids) :], [0] * len(tool_response_ids))
        tool_state = result.extra_fields["gsm8k_tool_agent_loop_state"]
        self.assertEqual(tool_state["current_turn_response_start_idx"], len(result.response_ids))

    async def test_does_not_reparse_previous_completed_turn(self):
        history = (
            '<tool_call>{"name":"calc_gsm8k_reward","arguments":{"answer":"42"}}</tool_call>'
            '<tool>{"result": 1.0}</tool>'
        )
        loop = self._build_loop([("final answer", Status.COMPLETED)], max_tokens=128, max_turns=2)
        state = self._make_aborted_state(history, max_tokens=128)
        state.extra_fields["gsm8k_tool_agent_loop_state"] = {
            "cur_turn": 1,
            "current_turn_response_start_idx": len(_encode(history)),
        }

        result = await loop.generate_sample(state, enable_partial_rollout=True, rollout_step=2)

        self.assertEqual(loop.tokenizer.tool_messages, [])
        self.assertIn("final answer", loop.tokenizer.decode(result.response_ids))


if __name__ == "__main__":
    unittest.main()
