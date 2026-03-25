from typing import List, Optional

from lagent.schema import AgentMessage as BaseAgentMessage
from lagent.schema import AgentStatusCode
from pydantic import Field

from xtuner.v1.data_proto.rl_data import RLRolloutResponseItem, RolloutState


class AgentMessage(BaseAgentMessage):
    """Extends the base AgentMessage to include RL model response conversion."""

    content_ids: Optional[List[int]] = Field(default=None, repr=False)
    content_logprobs: Optional[List[float]] = Field(default=None, repr=False)
    thinking_ids: Optional[List[int]] = Field(default=None, repr=False)
    thinking_logprobs: Optional[List[float]] = Field(default=None, repr=False)
    reward: Optional[float] = None
    raw_content: Optional[str] = None
    raw_content_ids: Optional[List[int]] = Field(default=None, repr=False)
    raw_content_logprobs: Optional[List[float]] = Field(default=None, repr=False)

    def merge_with(self, other: 'AgentMessage') -> 'AgentMessage':
        assert self.finish_reason == 'abort', f"Cannot merge with non-aborted message: {self.finish_reason}"
        self.raw_content += other.raw_content or ""
        self.content = self.raw_content
        self.raw_content_ids.extend(other.raw_content_ids or [])
        self.content_ids = self.raw_content_ids
        self.raw_content_logprobs.extend(other.raw_content_logprobs or [])
        self.content_logprobs = self.raw_content_logprobs
        self.reward = other.reward
        self.finish_reason = other.finish_reason
        self.stream_state = other.stream_state
        return self

    @classmethod
    def from_model_response(cls, model_response: RLRolloutResponseItem, sender: str) -> "AgentMessage":
        """Convert model response dict to AgentMessage."""
        return cls(
            sender=sender,
            content=model_response.response or "",
            content_ids=model_response.response_ids,
            content_logprobs=model_response.logprobs,
            thinking=None,
            thinking_ids=None,
            tool_calls=None,
            tool_calls_ids=None,
            raw_content=model_response.response or "",
            raw_content_ids=model_response.response_ids,
            raw_content_logprobs=model_response.logprobs,
            extra_info=model_response.extra_info,
            stream_state=(
                AgentStatusCode.END
                if model_response.state in [RolloutState.COMPLETED, RolloutState.ABORTED]
                else (
                    AgentStatusCode.SESSION_OUT_OF_LIMIT
                    if model_response.finish_reason == 'length'
                    else AgentStatusCode.SERVER_ERR
                )
            ),
            finish_reason=model_response.finish_reason,
        )

    def to_model_request(self, role: str = 'assistant') -> dict:
        """Convert AgentMessage to model request dict."""
        return {
            "role": role,
            "content": self.content,
            "content_ids": self.content_ids,
            "content_logprobs": self.content_logprobs,
            "thinking": self.thinking,
            "thinking_ids": self.thinking_ids,
            "tool_calls": self.tool_calls,
            "tool_calls_ids": self.tool_calls_ids,
            "raw_content": self.raw_content,
            "raw_content_ids": self.raw_content_ids,
            "raw_content_logprobs": self.raw_content_logprobs,
            "extra_info": self.extra_info,
            "stream_state": self.stream_state,
            "finish_reason": self.finish_reason,
        }
