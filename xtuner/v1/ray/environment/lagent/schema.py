import time
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from lagent.schema import AgentMessage as BaseAgentMessage
from lagent.schema import AgentStatusCode, ModelStatusCode
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel, Field

from xtuner.v1.data_proto.rl_data import (
    RLRolloutResponseItem,
    RolloutState,
    SampleParams,
)


class AgentMessage(BaseAgentMessage):
    """Extends the base AgentMessage to include RL model response
    conversion."""

    content_ids: Optional[List[int]] = Field(default=None, repr=False)
    content_logprobs: Optional[List[float]] = Field(default=None, repr=False)
    thinking_ids: Optional[List[int]] = Field(default=None, repr=False)
    thinking_logprobs: Optional[List[float]] = Field(default=None, repr=False)
    reward: Optional[float | dict] = None
    raw_content: Optional[str] = None
    raw_content_ids: Optional[List[int]] = Field(default=None, repr=False)
    raw_content_logprobs: Optional[List[float]] = Field(default=None, repr=False)

    def merge_with(self, other: "AgentMessage") -> "AgentMessage":
        assert self.finish_reason == "abort", f"Cannot merge with non-aborted message: {self.finish_reason}"  # type: ignore[has-type]
        self.raw_content = (self.raw_content or "") + (other.raw_content or "")
        self.content = self.raw_content
        if self.raw_content_ids is not None:
            self.raw_content_ids.extend(other.raw_content_ids or [])
        self.content_ids = self.raw_content_ids
        if self.raw_content_logprobs is not None:
            self.raw_content_logprobs.extend(other.raw_content_logprobs or [])
        self.content_logprobs = self.raw_content_logprobs
        self.reward = other.reward
        self.finish_reason = other.finish_reason  # type: ignore[has-type, assignment]
        self.stream_state = other.stream_state  # type: ignore[has-type, assignment]
        self.extra_info = other.extra_info  # type: ignore[has-type, assignment]
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
                    if model_response.finish_reason == "length"
                    else AgentStatusCode.SERVER_ERR
                )
            ),
            finish_reason=model_response.finish_reason,
        )

    def to_model_request(self, role: str = "assistant") -> dict:
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


# ---------------------------------------------------------------------------
# Extended OpenAI-compatible request / response types
# ---------------------------------------------------------------------------
class LagentChatCompletionMessage(ChatCompletionMessage):
    """Extends OpenAI's ChatCompletionMessage with RL-specific fields that are
    populated after parsing (reasoning / raw token-level info)."""

    model_config = {"extra": "allow"}

    # Reasoning / thinking text (e.g. <think>…</think> extracted by reasoning_parser)
    reasoning_content: Optional[str] = None
    reasoning_content_ids: Optional[List[int]] = Field(default=None, repr=False)
    reasoning_content_logprobs: Optional[List[float]] = Field(default=None, repr=False)
    # Unprocessed model output before parser stripping
    raw_content: Optional[str] = None
    raw_content_ids: Optional[List[int]] = Field(default=None, repr=False)
    raw_content_logprobs: Optional[List[float]] = Field(default=None, repr=False)
    # Token ids / logprobs for the final content after parsing
    content_ids: Optional[List[int]] = Field(default=None, repr=False)
    content_logprobs: Optional[List[float]] = Field(default=None, repr=False)
    # Arbitrary extra information from the rollout worker (e.g. routed_experts)
    extra_info: Optional[Dict[str, Any]] = None

    @classmethod
    def from_agent_message(cls, message: "AgentMessage") -> "LagentChatCompletionMessage":
        """Build an LagentChatCompletionMessage from a parsed AgentMessage."""
        # Convert lagent tool_calls to OpenAI ChatCompletionMessageToolCall if present
        tool_calls = None
        if message.tool_calls:
            import json

            from openai.types.chat.chat_completion_message_tool_call import (
                ChatCompletionMessageToolCall,
                Function,
            )

            tool_calls = []
            for i, tc in enumerate(message.tool_calls):
                if isinstance(tc, dict):
                    # Normalise the three possible tool_call dict shapes:
                    # 1. OpenAI-nested : {'id': ..., 'function': {'name': ..., 'arguments': ...}}
                    # 2. lagent flat   : {'name': ..., 'arguments': ...}
                    # 3. other flat    : {'name': ..., 'parameters': ...}
                    if "function" in tc:
                        func = tc["function"]
                        name = func.get("name", "")
                        raw_args = func.get("arguments", {})
                    else:
                        name = tc.get("name", "")
                        raw_args = tc.get("arguments", tc.get("parameters", {}))

                    arguments = json.dumps(raw_args, ensure_ascii=False) if not isinstance(raw_args, str) else raw_args
                    tool_calls.append(
                        ChatCompletionMessageToolCall(
                            id=tc.get("id", f"call_{i}"),
                            type="function",
                            function=Function(name=name, arguments=arguments),
                        )
                    )

        return cls(
            role="assistant",
            content=message.content if isinstance(message.content, str) else None,
            tool_calls=tool_calls,
            # reasoning / thinking fields populated by reasoning_parser
            reasoning_content=message.thinking,
            reasoning_content_ids=message.thinking_ids,
            reasoning_content_logprobs=message.thinking_logprobs,
            # raw (pre-parse) output fields
            raw_content=message.raw_content,
            raw_content_ids=message.raw_content_ids,
            raw_content_logprobs=message.raw_content_logprobs,
            # final content fields
            content_ids=message.content_ids,
            content_logprobs=message.content_logprobs,
            extra_info=message.extra_info,
        )


class LagentChoice(BaseModel):
    """Extends OpenAI Choice with the richer message type and RL metadata."""

    index: int
    message: LagentChatCompletionMessage
    finish_reason: Optional[str] = None
    # RL-specific choice-level metadata
    reward: Optional[float] = None
    stream_state: Union[ModelStatusCode, AgentStatusCode] = AgentStatusCode.END


class LagentChatCompletion(BaseModel):
    """OpenAI-compatible ChatCompletion response with Lagent extensions."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: List[LagentChoice]
    usage: Optional[CompletionUsage] = None


class LagentChatCompletionRequest(BaseModel):
    """OpenAI-compatible ChatCompletion request extended with XTuner params."""

    model: str = ""
    messages: List[Dict[str, Any]]
    tools: List[Any] = Field(default_factory=list)
    tool_choice: str = "auto"
    # XTuner-specific generation parameters
    sample_params: SampleParams = Field(default_factory=SampleParams)
    extra_params: Dict[str, Any] = Field(default_factory=dict)
