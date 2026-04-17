import copy

from lagent.agents import Agent

from xtuner.v1.data_proto.rl_data import RLRolloutResponseItem
from xtuner.v1.ray.environment.lagent.schema import AgentMessage


class AsyncTokenInOutAgentMixin:
    async def __call__(self, *message: AgentMessage, session_id=0, **kwargs) -> AgentMessage:
        message = [AgentMessage(sender='user', content=m) if isinstance(m, str) else copy.deepcopy(m) for m in message]
        for hook in self._hooks.values():
            result = hook.before_agent(self, message, session_id)
            if result:
                message = result

        # resume aborted rollout
        _message = self._scroll_buffer(message[-1], session_id)
        if _message is not None:
            if _message.finish_reason != 'abort':
                _message = copy.deepcopy(_message)
                for hook in self._hooks.values():
                    result = hook.after_agent(self, _message, session_id)
                    if result:
                        _message = result
                return _message
            message[-1].extra_info['partial_response'] = _message
        else:
            self.update_memory(message, session_id=session_id)
        response_message = await self.forward(*message, session_id=session_id, **kwargs)
        if _message and _message.finish_reason == 'abort':
            message[-1].extra_info.pop('partial_response', None)
        if not isinstance(response_message, AgentMessage):
            assert isinstance(
                response_message, RLRolloutResponseItem
            ), f"Expected response to be of type AgentMessage or RLRolloutResponseItem, but got {type(response_message)}"
            response_message = AgentMessage.from_model_response(response_message, self.name)
        self.update_memory(response_message, session_id=session_id)
        response_message = copy.deepcopy(response_message)
        for hook in self._hooks.values():
            result = hook.after_agent(self, response_message, session_id)
            if result:
                response_message = result
        return response_message

    async def forward(self, *message: AgentMessage, session_id=0, **kwargs) -> AgentMessage:
        partial_response: AgentMessage = message[-1].extra_info.get('partial_response')
        if partial_response and partial_response.raw_content:
            self.update_memory(partial_response, session_id=session_id)
        formatted_messages = self.aggregator.aggregate(
            self.memory.get(session_id), self.name, self.output_format, self.template
        )
        response_message = await self.llm.chat(formatted_messages, session_id, **kwargs)
        if isinstance(response_message, AgentMessage):
            response_message.sender = self.name
        if partial_response and partial_response.raw_content:
            response_message = partial_response.merge_with(response_message)
            response_message = self.llm.parse_response(response_message)
            # remove the partial response from memory, since it's merged into the final response
            self.memory.get(session_id).delete(-1)
        return response_message


class AsyncTokenInOutAgent(AsyncTokenInOutAgentMixin, Agent):
    pass
