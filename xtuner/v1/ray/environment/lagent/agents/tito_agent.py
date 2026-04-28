import copy

from lagent.agents import Agent

from xtuner.v1.data_proto.rl_data import RLRolloutResponseItem
from xtuner.v1.ray.environment.lagent.schema import AgentMessage


class AsyncTokenInOutAgentMixin:
    async def __call__(self, *message: AgentMessage, **kwargs) -> AgentMessage:  # type: ignore[override]
        message = [AgentMessage(sender="user", content=m) if isinstance(m, str) else copy.deepcopy(m) for m in message]  # type: ignore[assignment]
        for hook in self._hooks.values():  # type: ignore[attr-defined]
            result = hook.before_agent(self, message)
            if result:
                message = result

        # resume aborted rollout
        _message = self._scroll_buffer(message[-1])  # type: ignore[attr-defined]
        if _message is not None:
            if _message.finish_reason != "abort":
                _message = copy.deepcopy(_message)
                for hook in self._hooks.values():  # type: ignore[attr-defined]
                    result = hook.after_agent(self, _message)
                    if result:
                        _message = result
                return _message
            message[-1].extra_info["partial_response"] = _message
        else:
            self.memory and self.memory.add(message)  # type: ignore[attr-defined]
        response_message = await self.forward(*message, **kwargs)
        if _message and _message.finish_reason == "abort":
            message[-1].extra_info.pop("partial_response", None)
        if not isinstance(response_message, AgentMessage):
            assert isinstance(response_message, RLRolloutResponseItem), (
                f"Expected response to be of type AgentMessage or RLRolloutResponseItem, but got {type(response_message)}"
            )
            response_message = AgentMessage.from_model_response(response_message, self.name)  # type: ignore[attr-defined]
        self.memory and self.memory.add(response_message)  # type: ignore[attr-defined]
        response_message = copy.deepcopy(response_message)
        for hook in self._hooks.values():  # type: ignore[attr-defined]
            result = hook.after_agent(self, response_message)
            if result:
                response_message = result
        return response_message

    async def forward(self, *message: AgentMessage, **kwargs) -> AgentMessage:
        partial_response: AgentMessage = message[-1].extra_info.get("partial_response")
        if partial_response and partial_response.raw_content:
            self.memory and self.memory.add(partial_response)  # type: ignore[attr-defined]
        formatted_messages, tools = self.aggregator.aggregate(  # type: ignore[attr-defined]
            self.memory,  # type: ignore[attr-defined]
            self.name,  # type: ignore[attr-defined]
            self.output_format,  # type: ignore[attr-defined]
            self.template,  # type: ignore[attr-defined]
        )
        response_message = await self.llm.chat(formatted_messages, tools=tools, **kwargs)  # type: ignore[attr-defined]
        if isinstance(response_message, AgentMessage):
            response_message.sender = self.name  # type: ignore[attr-defined]
        if partial_response and partial_response.raw_content:
            response_message = partial_response.merge_with(response_message)
            response_message = self.llm.parse_response(response_message)  # type: ignore[attr-defined]
            # remove the partial response from memory, since it's merged into the final response
            self.memory.delete(-1)  # type: ignore[attr-defined]
        return response_message


class AsyncTokenInOutAgent(AsyncTokenInOutAgentMixin, Agent):
    pass
