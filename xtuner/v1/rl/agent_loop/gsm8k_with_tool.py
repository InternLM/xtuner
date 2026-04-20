import copy
import json
import re
from typing import cast

from pydantic import BaseModel, ConfigDict

from xtuner.v1.data_proto import RolloutState, SampleParams, Status
from xtuner.v1.rl.agent_loop import AgentLoop, AgentLoopConfig
from xtuner.v1.rl.agent_loop.utils import PartialRolloutHandler
from xtuner.v1.rl.judger import Judger
from xtuner.v1.rl.rollout import RolloutController
from xtuner.v1.utils import get_logger


logger = get_logger()

_TOOL_LOOP_STATE_KEY = "gsm8k_tool_agent_loop_state"


class GSM8KToolAgentLoopConfig(AgentLoopConfig):
    max_turns: int

    def build_local(self, rollout_controller, judger: Judger | None = None, logger=None) -> "GSM8KToolAgentLoop":
        return GSM8KToolAgentLoop(
            max_turns=self.max_turns,
            rollout_ctl=rollout_controller,
            hf_checkpoint=self.hf_checkpoint,
            sample_params=self.sample_params,
            judger=judger,
        )


class FunctionCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: dict


class GSM8KToolAgentLoop(AgentLoop):
    def __init__(
        self,
        max_turns: int,
        rollout_ctl: RolloutController,
        hf_checkpoint: str,
        sample_params: SampleParams,
        judger: Judger | None = None,
    ):
        super().__init__(
            rollout_ctl=rollout_ctl, hf_checkpoint=hf_checkpoint, sample_params=sample_params, judger=judger
        )
        self.max_turns = max_turns
        self.max_tokens = self.sample_params.max_tokens
        self.partial_rollout_handler = PartialRolloutHandler(max_tokens=self.max_tokens)
        self.tool_call_pattern = re.compile(r"\n*<tool_call>(.*?)</tool_call>", re.DOTALL)
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"

    def calc_gsm8k_reward(self, answer: dict, ground_truth: str) -> float:
        from xtuner.v1.rl.judger.gsm8k import compute_reward

        extra_info = {"score": 1.0, "format_score": 0}
        actual_answer = answer.get("answer", "")
        if not actual_answer.startswith("#### "):
            actual_answer = "#### " + actual_answer
        return compute_reward(actual_answer, ground_truth, extra_info)

    def extract_tool_calls_from_text(self, text: str) -> tuple[str, list[FunctionCall]]:
        if self.tool_call_start_token not in text or self.tool_call_end_token not in text:
            return text, []

        matches = self.tool_call_pattern.findall(text)
        function_calls = []
        for match in matches:
            try:
                function_call = json.loads(match)
                name, arguments = function_call["name"], function_call["arguments"]
                function_calls.append(FunctionCall(name=name, arguments=arguments))
            except Exception as e:
                logger.error(f"Error parsing tool call JSON: {e}")
                continue

        content = self.tool_call_pattern.sub("", text)
        return content, function_calls

    def _save_tool_loop_state(
        self, rollout_state: RolloutState, cur_turn: int, current_turn_response_start_idx: int
    ) -> None:
        tool_loop_state = rollout_state.extra_fields.setdefault(_TOOL_LOOP_STATE_KEY, {})
        tool_loop_state["cur_turn"] = cur_turn
        tool_loop_state["current_turn_response_start_idx"] = current_turn_response_start_idx

    async def generate_sample(self, rollout_state: RolloutState, **kwargs) -> RolloutState:
        enable_partial_rollout = kwargs.get("enable_partial_rollout", False)
        rollout_step = kwargs.get("rollout_step", 0)
        if enable_partial_rollout:
            return await self._generate_sample_with_partial_rollout(rollout_state, rollout_step)
        return await self._generate_sample_without_partial_rollout(rollout_state)

    async def _judge_completed_sample(self, rollout_state: RolloutState) -> RolloutState:
        if rollout_state.status != Status.COMPLETED:
            return rollout_state
        if self.judger is not None:
            rollout_state = await self.judger.judge(rollout_state)
        return rollout_state

    async def _generate_sample_without_partial_rollout(self, rollout_state: RolloutState) -> RolloutState:
        if not rollout_state.tokens:
            rollout_state.tokens = rollout_state.prompt_ids

        base_sample_params = copy.deepcopy(rollout_state.sample_params or self.sample_params)
        final_response_mask: list[int] = []
        final_response_ids: list[int] = []
        final_logprobs: list[float] = []

        max_len = base_sample_params.max_tokens
        cur_turn_tokens = list(rollout_state.tokens or rollout_state.prompt_ids or [])
        remaining_max_tokens = max_len - len(final_response_ids)
        cur_turn = 0

        while True:
            if cur_turn >= self.max_turns or len(final_response_ids) >= max_len or remaining_max_tokens <= 0:
                break

            rollout_state.tokens = cur_turn_tokens
            rollout_state.sample_params = copy.deepcopy(base_sample_params)
            rollout_state.sample_params.max_tokens = remaining_max_tokens

            rollout_state = await self.rollout_ctl.generate.remote(rollout_state)  # type: ignore[attr-defined]
            response_ids = list(rollout_state.response_ids or [])
            cur_turn_tokens.extend(response_ids)

            # 处理 rollout_controller 的输出
            final_response_ids.extend(response_ids)
            final_logprobs.extend(list(rollout_state.logprobs or []))
            final_response_mask.extend([1] * len(response_ids))
            # TODO: 处理 routed_experts, 要注意这里涉及到是否要解引用的问题

            if rollout_state.status != Status.COMPLETED:
                break

            cur_turn += 1
            if len(final_response_ids) >= max_len:
                break

            response_text = self.tokenizer.decode(response_ids)
            _, function_calls = self.extract_tool_calls_from_text(response_text)
            if not function_calls:
                break

            tool_messages = []
            for function_call in function_calls:
                tool_name = function_call.name
                tool_args = function_call.arguments
                if tool_name == "calc_gsm8k_reward":
                    answer = tool_args
                    ground_truth = cast(dict, rollout_state.reward_model).get("ground_truth", "")
                    function_results = self.calc_gsm8k_reward(answer, ground_truth)
                    tool_message = {
                        "role": "tool",
                        "content": json.dumps({"result": function_results}, ensure_ascii=False),
                    }
                    tool_messages.append(tool_message)

            # 处理工具调用的输出
            tools_response_ids = self.tokenizer.apply_chat_template(tool_messages, remove_system_prompt=True)
            tool_token_budget = max_len - len(final_response_ids)
            tools_response_ids = tools_response_ids[:tool_token_budget]
            final_response_ids.extend(tools_response_ids)
            final_logprobs.extend([0.0] * len(tools_response_ids))
            final_response_mask.extend([0] * len(tools_response_ids))

            # 处理下一轮生成的输入
            cur_turn_tokens.extend(tools_response_ids)
            remaining_max_tokens = max_len - len(final_response_ids)

        final_response_ids = final_response_ids[:max_len]
        final_response_mask = final_response_mask[:max_len]
        final_logprobs = final_logprobs[:max_len]

        rollout_state.response_ids = final_response_ids
        rollout_state.response_mask = final_response_mask
        rollout_state.logprobs = final_logprobs
        rollout_state.response = self.tokenizer.decode(rollout_state.response_ids)
        assert len(rollout_state.response_ids) == len(rollout_state.response_mask) == len(rollout_state.logprobs), (
            f"{len(rollout_state.response_ids)} vs {len(rollout_state.response_mask)} vs {len(rollout_state.logprobs)}"
        )
        return await self._judge_completed_sample(rollout_state)

    async def _generate_sample_with_partial_rollout(
        self, rollout_state: RolloutState, rollout_step: int
    ) -> RolloutState:
        original_status = rollout_state.status
        if original_status == Status.EXPIRED:
            rollout_state.extra_fields.pop(_TOOL_LOOP_STATE_KEY, None)

        rollout_state = self.partial_rollout_handler.preprocess(rollout_state, True)
        if not rollout_state.tokens:
            rollout_state.tokens = rollout_state.prompt_ids

        base_sample_params = copy.deepcopy(rollout_state.sample_params or self.sample_params)
        final_response_mask: list[int] = []
        final_response_ids: list[int] = []
        final_logprobs: list[float] = []

        history_dict = rollout_state.extra_fields.get("history_response_dict") or {}
        history_response_ids = list(history_dict.get("response_ids", []))
        # This state is agent-loop specific. It lets a resumed sample know which
        # GSM8K tool turn it is in, and where the current assistant turn starts
        # inside the full response stream.
        tool_loop_state = rollout_state.extra_fields.setdefault(_TOOL_LOOP_STATE_KEY, {})
        cur_turn = int(tool_loop_state.get("cur_turn", 0))
        current_turn_response_start_idx = int(tool_loop_state.get("current_turn_response_start_idx", 0))

        max_len = base_sample_params.max_tokens
        cur_turn_tokens = list(rollout_state.tokens or rollout_state.prompt_ids or [])
        remaining_max_tokens = max_len - len(final_response_ids)

        if remaining_max_tokens <= 0:
            rollout_state.response_ids = []
            rollout_state.response_mask = []
            rollout_state.logprobs = []
            rollout_state.response = ""
            rollout_state.finish_reason = "length"
            rollout_state.status = Status.COMPLETED

        while True:
            if cur_turn >= self.max_turns or len(final_response_ids) >= max_len or remaining_max_tokens <= 0:
                break

            rollout_state.tokens = cur_turn_tokens
            rollout_state.sample_params = copy.deepcopy(base_sample_params)
            rollout_state.sample_params.max_tokens = remaining_max_tokens

            rollout_state = await self.rollout_ctl.generate.remote(rollout_state)  # type: ignore[attr-defined]
            response_ids = list(rollout_state.response_ids or [])
            cur_turn_tokens.extend(response_ids)

            final_response_ids.extend(response_ids)
            final_logprobs.extend(list(rollout_state.logprobs or []))
            final_response_mask.extend([1] * len(response_ids))
            # TODO: 处理 routed_experts, 要注意这里涉及到是否要解引用的问题

            if rollout_state.status != Status.COMPLETED:
                # The current assistant turn was interrupted. Save the same
                # cur_turn/start index so the next partial rollout resumes this
                # unfinished turn instead of treating it as a new turn.
                self._save_tool_loop_state(rollout_state, cur_turn, current_turn_response_start_idx)
                break

            cur_turn += 1
            # The model finished one assistant turn. Persist that before any
            # possible later pause, so resume continues from the correct turn.
            self._save_tool_loop_state(rollout_state, cur_turn, current_turn_response_start_idx)
            if len(final_response_ids) >= max_len:
                break

            full_response_ids = history_response_ids + final_response_ids
            # Parse only the current assistant turn. Previous turns may already
            # contain executed tool calls and must not be parsed again.
            response_text = self.tokenizer.decode(full_response_ids[current_turn_response_start_idx:])
            _, function_calls = self.extract_tool_calls_from_text(response_text)
            if not function_calls:
                break

            tool_messages = []
            for function_call in function_calls:
                tool_name = function_call.name
                tool_args = function_call.arguments
                if tool_name == "calc_gsm8k_reward":
                    answer = tool_args
                    ground_truth = cast(dict, rollout_state.reward_model).get("ground_truth", "")
                    function_results = self.calc_gsm8k_reward(answer, ground_truth)
                    tool_message = {
                        "role": "tool",
                        "content": json.dumps({"result": function_results}, ensure_ascii=False),
                    }
                    tool_messages.append(tool_message)

            tools_response_ids = self.tokenizer.apply_chat_template(tool_messages, remove_system_prompt=True)
            tool_token_budget = max_len - len(final_response_ids)
            tools_response_ids = tools_response_ids[:tool_token_budget]
            final_response_ids.extend(tools_response_ids)
            final_logprobs.extend([0.0] * len(tools_response_ids))
            final_response_mask.extend([0] * len(tools_response_ids))

            cur_turn_tokens.extend(tools_response_ids)
            remaining_max_tokens = max_len - len(final_response_ids)
            # Tool output closes the current turn. The next model generation
            # starts after all response tokens accumulated so far.
            current_turn_response_start_idx = len(history_response_ids) + len(final_response_ids)
            self._save_tool_loop_state(rollout_state, cur_turn, current_turn_response_start_idx)

        final_response_ids = final_response_ids[:max_len]
        final_response_mask = final_response_mask[:max_len]
        final_logprobs = final_logprobs[:max_len]

        rollout_state.response_ids = final_response_ids
        rollout_state.response_mask = final_response_mask
        rollout_state.logprobs = final_logprobs
        rollout_state.response = self.tokenizer.decode(rollout_state.response_ids)
        # postprocess appends current-step rollout ids before concatenating
        # historical response tokens, so response_ids must still be only the new
        # segment here.
        rollout_state = self.partial_rollout_handler.postprocess(rollout_state, rollout_step)
        assert len(rollout_state.response_ids) == len(rollout_state.response_mask) == len(rollout_state.logprobs), (
            f"{len(rollout_state.response_ids)} vs {len(rollout_state.response_mask)} vs {len(rollout_state.logprobs)}"
        )
        return await self._judge_completed_sample(rollout_state)
