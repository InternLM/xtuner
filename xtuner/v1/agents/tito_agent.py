import copy

from lagent.agents import Agent

from xtuner.v1.data_proto.messages.agent import AgentMessage
from xtuner.v1.data_proto.rl_data import RLRolloutResponseItem


class AsyncTokenInOutAgentMixin:
    async def __call__(self, *message: AgentMessage, session_id=0, **kwargs) -> AgentMessage:
        message = [AgentMessage(sender='user', content=m) if isinstance(m, str) else copy.deepcopy(m) for m in message]
        for hook in self._hooks.values():
            result = hook.before_agent(self, message, session_id)
            if result:
                message = result
        self.update_memory(message, session_id=session_id)
        response_message = await self.forward(*message, session_id=session_id, **kwargs)
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
        formatted_messages = self.aggregator.aggregate(
            self.memory.get(session_id), self.name, self.output_format, self.template
        )
        response_message = await self.llm.chat(formatted_messages, session_id, **kwargs)
        if isinstance(response_message, AgentMessage):
            response_message.sender = self.name
        return response_message


class AsyncTokenInOutAgent(AsyncTokenInOutAgentMixin, Agent):
    pass
