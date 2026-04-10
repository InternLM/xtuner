import copy
import json
import re
from typing import cast

from pydantic import BaseModel, ConfigDict

from xtuner.v1.data_proto import RolloutState, SampleParams
from xtuner.v1.rl.agent_loop import AgentLoop, AgentLoopConfig, JudgerSpec
from xtuner.v1.rl.rollout import RolloutController
from xtuner.v1.utils import get_logger


logger = get_logger()


class GSM8KToolAgentLoopConfig(AgentLoopConfig):
    max_turns: int

    def build(self, rollout_controller, logger=None) -> "GSM8KToolAgentLoop":
        return GSM8KToolAgentLoop(
            max_turns=self.max_turns,
            rollout_ctl=rollout_controller,
            hf_checkpoint=self.hf_checkpoint,
            sample_params=self.sample_params,
            judger=self.build_judger(),
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
        judger: JudgerSpec = None,
    ):
        super().__init__(
            rollout_ctl=rollout_ctl, hf_checkpoint=hf_checkpoint, sample_params=sample_params, judger=judger
        )
        self.max_turns = max_turns
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

    def extract_tool_calls(self, rollout_state: RolloutState) -> tuple[str, list[FunctionCall]]:
        text = self.tokenizer.decode(rollout_state.response_ids)
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

    async def generate_sample(self, rollout_state: RolloutState, **kwargs) -> RolloutState:
        # Respect state passed from preprocess for partial rollout continuation.
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
            cur_turn += 1
            response_ids = cast(list[int], rollout_state.response_ids)
            cur_turn_tokens.extend(response_ids)

            # 处理 rollout_controller 的输出
            final_response_ids.extend(response_ids)
            final_logprobs.extend(cast(list[float], rollout_state.logprobs))
            final_response_mask.extend([1] * len(response_ids))
            # TODO: 处理 routed_experts, 要注意这里涉及到是否要解引用的问题

            if len(final_response_ids) >= max_len:
                break

            _, function_calls = self.extract_tool_calls(rollout_state)
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
        rollout_state = await self.judge_sample(rollout_state)
        return rollout_state
