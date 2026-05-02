import asyncio
import copy
import json
from dataclasses import asdict
from typing import Dict, List, Literal, Optional, Union

from lagent.agents import AsyncAgent
from lagent.agents.fc_agent import EnvAgent as BaseEnvAgent
from lagent.hooks import Hook
from lagent.schema import ActionStatusCode, ActionValidCode, AgentStatusCode
from lagent.utils import create_object, truncate_text

from xtuner.v1.ray.environment.lagent.schema import AgentMessage


def finish_condition_func(selection_message, env_message):
    return (env_message.extra_info or {}).get("finish", False)


class EnvAgent(BaseEnvAgent):
    def __init__(
        self,
        actions: list,
        judger: Union[Dict, AsyncAgent],
        stateful_tools: Optional[List[str]] = None,
        max_turn: Optional[int] = None,
        max_tool_response_length: Optional[int] = None,
        max_tool_calls_per_turn: int = 5,
        tool_response_truncate_side: Literal["left", "right", "middle"] = "middle",
        enable_no_thinking_penalty: bool = True,
        lower_tool_turn_bound: Optional[int] = None,
        enable_repeated_tool_call_penalty: bool = False,
        action_hooks: Optional[List[Union[dict, Hook]]] = None,
        name: Optional[str] = None,
    ):
        super().__init__(
            actions, stateful_tools, max_tool_response_length, tool_response_truncate_side, action_hooks, name
        )
        self.judger: AsyncAgent = create_object(judger)
        # scoring rule settings
        self.max_turn = max_turn
        self.enable_no_thinking_penalty = enable_no_thinking_penalty
        self.max_tool_calls_per_turn = max_tool_calls_per_turn
        self.lower_tool_turn_bound = lower_tool_turn_bound
        self.enable_repeated_tool_call_penalty = enable_repeated_tool_call_penalty

    async def forward(self, assistant_message: AgentMessage, **kwargs):
        extra_info = {}
        current_turn = len(self.memory.get_memory()) // 2
        num_turns = current_turn * 2
        if assistant_message.stream_state == AgentStatusCode.SESSION_OUT_OF_LIMIT:
            extra_info["finish"] = True
            return AgentMessage(
                sender=self.name, content="Session Length Out Of Limit", extra_info=extra_info, reward=0.0
            )
        if self.max_turn is not None and current_turn > self.max_turn:
            extra_info["finish"] = True
            return AgentMessage(sender=self.name, content="Reach Max turn", extra_info=extra_info, reward=0.0)
        if self.enable_no_thinking_penalty and not assistant_message.thinking:
            extra_info["finish"] = True
            return AgentMessage(sender=self.name, content="Format Error", extra_info=extra_info, reward=0.0)
        if not assistant_message.tool_calls:
            extra_info["finish"] = True
            # Preserve full in-session trace for judger/reward preprocessing.
            # This path runs before AgentEnvironment postprocess_func, so we must
            # attach trace data directly on assistant_message.extra_info.
            assistant_message.extra_info.update(
                {
                    'agent_trace': [msg.model_dump() for msg in self.memory.get_memory()],
                    'num_turns': num_turns,
                }
            )
            if "<tool_call>" in (assistant_message.raw_content or ""):
                return AgentMessage(sender=self.name, content="Format Error", extra_info=extra_info, reward=-1.0)
            # assume the first message contains meta info to help judge
            set_env_message = self.memory.get_memory()[0]
            message = (await self.judger(assistant_message, set_env_message, **kwargs)).model_copy(
                update={"sender": self.name}, deep=True
            )
            self.judger.reset(recursive=True)
            message.extra_info.update(extra_info)
            # 惩罚工具调用轮数低于下限的样本
            if self.lower_tool_turn_bound is not None and current_turn < self.lower_tool_turn_bound:
                if isinstance(message.reward, (int, float)):
                    message.reward = min(message.reward, 0.0)
                elif isinstance(message.reward, dict) and "score" in message.reward:
                    message.reward["score"] = min(message.reward["score"], 0.0)
            return message
        if (
            self.max_tool_calls_per_turn is not None
            and len(assistant_message.tool_calls) > self.max_tool_calls_per_turn
        ):
            extra_info["finish"] = True
            return AgentMessage(
                sender=self.name, content="Exceed Max Tool Calls Per Turn", extra_info=extra_info, reward=0.0
            )
        # 惩罚冗余工具调用
        if self.enable_repeated_tool_call_penalty:
            previous_tool_calls = set()
            for msg in self.memory.get_memory()[:-1]:
                for call in msg.tool_calls or []:
                    try:
                        if isinstance(call["arguments"], str):
                            args = json.loads(call["arguments"])
                        previous_tool_calls.add((call["name"], tuple(sorted(args.items()))))
                    except Exception:
                        continue
            for call in assistant_message.tool_calls:
                try:
                    if isinstance(call["arguments"], str):
                        args = json.loads(call["arguments"])
                    if (call["name"], tuple(sorted(args.items()))) in previous_tool_calls:
                        extra_info["finish"] = True
                        return AgentMessage(
                            sender=self.name,
                            content=f"Repeated Tool Call: {call['name']}",
                            extra_info=extra_info,
                            reward=-1,
                        )
                except Exception:
                    continue

        tool_responses = await asyncio.gather(
            *[self._retry_mechanism(self.execute_tool)(tool_call) for tool_call in assistant_message.tool_calls]
        )
        for i, tool_response in enumerate(tool_responses):
            if tool_response.valid != ActionValidCode.OPEN:
                extra_info["finish"] = True
                return AgentMessage(
                    sender=self.name,
                    content=f"Tool Call Error: {tool_response.errmsg} in tool call "
                    f"{json.dumps(assistant_message.tool_calls[i], ensure_ascii=False)}",
                    extra_info=extra_info,
                    reward=-1,
                )
            if tool_response.state != ActionStatusCode.SUCCESS:
                extra_info["finish"] = True
                return AgentMessage(
                    sender=self.name,
                    content=f"Tool Call Error: {tool_response.errmsg} in tool call "
                    f"{json.dumps(assistant_message.tool_calls[i], ensure_ascii=False)}",
                    extra_info=extra_info,
                    reward=-1 if tool_response.state == ActionStatusCode.ARGS_ERROR else 0,
                )
            res = tool_response.format_result()
            if self.max_tool_response_length is not None and len(res) > self.max_tool_response_length:
                res = truncate_text(res, max_num=self.max_tool_response_length, side=self.tool_response_truncate_side)
                tool_response.result = [{"type": "text", "content": res}]
        extra_info["finish"] = False
        return_message = AgentMessage(
            sender=self.name, content=[asdict(resp) for resp in tool_responses], extra_info=extra_info, reward=0.0
        )
        return return_message
