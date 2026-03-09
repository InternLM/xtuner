import json
import re
from typing import Any, cast

from pydantic import BaseModel

from xtuner.v1.data_proto import RolloutState, SampleParams
from xtuner.v1.ray.rollout import RolloutController
from xtuner.v1.rl.base.agent_loop import AgentLoop, AgentLoopConfig
from xtuner.v1.utils import get_logger


logger = get_logger()

gsm8k_tools = [
    {
        "type": "function",
        "function": {
            "name": "calc_gsm8k_reward",
            "description": "A tool for calculating the reward of gsm8k. (1.0 if parsed answer is correct, 0.0 if parsed answer is incorrect or not correctly parsed)",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The model's answer to the GSM8K math problem, must be a digits",
                    },
                    "required": ["answer"],
                },
            },
        },
    }
]


class GSM8KToolAgentLoopConfig(AgentLoopConfig):
    max_turns: int
    tools: list[dict[str, Any]] | None = gsm8k_tools  # TODO: 明确tools如何定义，目前先写死

    def build(self, rollout_controller, judger=None) -> "GSM8KToolAgentLoop":
        return GSM8KToolAgentLoop(
            max_turns=self.max_turns,
            tools=self.tools,
            rollout_ctl=rollout_controller,
            hf_checkpoint=self.hf_checkpoint,
            sample_params=self.sample_params,
            judger=judger,
        )


class FunctionCall(BaseModel):
    name: str
    arguments: dict


class GSM8KToolAgentLoop(AgentLoop):
    def __init__(
        self,
        max_turns: int,
        tools: list[dict[str, Any]] | None,
        rollout_ctl: RolloutController,
        hf_checkpoint: str,
        sample_params: SampleParams,
        judger=None,
    ):
        super().__init__(
            rollout_ctl=rollout_ctl, hf_checkpoint=hf_checkpoint, sample_params=sample_params, judger=judger
        )
        self.max_turns = max_turns
        self.tools = tools
        self.tool_call_pattern = re.compile(r"\n*<tool_call>(.*?)</tool_call>", re.DOTALL)
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"

    def apply_chat_template(self, rollout_state: RolloutState) -> RolloutState:
        tokens_with_tools = self.tokenizer.apply_chat_template(
            rollout_state.message, tools=self.tools, tokenize=True, add_generation_prompt=False
        )
        rollout_state.tokens = tokens_with_tools
        return rollout_state

    def calc_gsm8k_reward(self, answer: dict, ground_truth: str) -> float:
        from xtuner.v1.ray.judger.gsm8k import compute_reward

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

    async def generate_sample(self, rollout_state: RolloutState) -> RolloutState:
        rollout_state.tools = self.tools
        # NOTE: 使用过程中发现很容易忘了给rollout_state传sample_params
        rollout_state.sample_params = self.sample_params
        rollout_state = self.apply_chat_template(rollout_state)

        final_response_mask = []
        final_response_ids = []
        final_logprobs = []

        max_len = self.sample_params.max_tokens
        cur_turn_tokens = cast(list[int], rollout_state.tokens)
        init_tokens = list(cur_turn_tokens)  # 深拷贝以保护原始Prompt
        cur_turn = 0
        while True:
            if cur_turn >= self.max_turns:
                break

            rollout_state.tokens = cur_turn_tokens
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
            cur_turn_tokens.extend(tools_response_ids)
            final_response_ids.extend(tools_response_ids)
            final_logprobs.extend([0.0] * len(tools_response_ids))
            final_response_mask.extend([0] * len(tools_response_ids))

            if len(final_response_ids) >= max_len:
                break

        final_response_ids = final_response_ids[:max_len]
        final_response_mask = final_response_mask[:max_len]
        final_logprobs = final_logprobs[:max_len]

        rollout_state.tokens = init_tokens
        rollout_state.response_ids = final_response_ids
        rollout_state.response_mask = final_response_mask
        rollout_state.logprobs = final_logprobs
        rollout_state.response = self.tokenizer.decode(rollout_state.response_ids)
        assert len(rollout_state.response_ids) == len(rollout_state.response_mask) == len(rollout_state.logprobs), (
            f"{len(rollout_state.response_ids)} vs {len(rollout_state.response_mask)} vs {len(rollout_state.logprobs)}"
        )
        rollout_state = await self.judge_sample(rollout_state)
        return rollout_state
