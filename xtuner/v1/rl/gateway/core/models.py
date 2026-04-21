from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GatewayCoreModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CanonicalToolDefinition(GatewayCoreModel):
    name: str
    description: str | None = None
    parameters_json_schema: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CanonicalToolChoice(GatewayCoreModel):
    type: Literal["auto", "none", "required", "specific"] = "auto"
    tool_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_specific_choice(self) -> CanonicalToolChoice:
        if self.type == "specific" and not self.tool_name:
            raise ValueError("tool_name is required when tool choice type is 'specific'.")
        return self


class CanonicalToolCall(GatewayCoreModel):
    id: str
    name: str
    arguments: Any = None
    raw_arguments_text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CanonicalToolResult(GatewayCoreModel):
    tool_call_id: str
    name: str | None = None
    output: Any = None
    output_text: str | None = None
    is_error: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class CanonicalReasoningStep(GatewayCoreModel):
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class CanonicalReasoning(GatewayCoreModel):
    steps: list[CanonicalReasoningStep] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CanonicalTextBlock(GatewayCoreModel):
    type: Literal["text"] = "text"
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class CanonicalToolCallBlock(GatewayCoreModel):
    type: Literal["tool_call"] = "tool_call"
    tool_call: CanonicalToolCall


class CanonicalToolResultBlock(GatewayCoreModel):
    type: Literal["tool_result"] = "tool_result"
    tool_result: CanonicalToolResult


class CanonicalReasoningBlock(GatewayCoreModel):
    type: Literal["reasoning"] = "reasoning"
    reasoning: CanonicalReasoning


CanonicalContentBlock: TypeAlias = Annotated[
    CanonicalTextBlock | CanonicalToolCallBlock | CanonicalToolResultBlock | CanonicalReasoningBlock,
    Field(discriminator="type"),
]


class CanonicalMessage(GatewayCoreModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: list[CanonicalContentBlock] = Field(default_factory=list)
    name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CanonicalGenerateRequest(GatewayCoreModel):
    request_id: str
    model: str
    messages: list[CanonicalMessage] = Field(default_factory=list)
    tools: list[CanonicalToolDefinition] = Field(default_factory=list)
    tool_choice: CanonicalToolChoice | None = None
    parallel_tool_calls: bool | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: list[str] = Field(default_factory=list)
    stream: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class CanonicalUsage(GatewayCoreModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CanonicalAssistantTurn(GatewayCoreModel):
    role: Literal["assistant"] = "assistant"
    content: list[CanonicalContentBlock] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CanonicalGenerateResponse(GatewayCoreModel):
    request_id: str
    model: str
    output: CanonicalAssistantTurn
    finish_reason: str = "stop"
    usage: CanonicalUsage = Field(default_factory=CanonicalUsage)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelCard(GatewayCoreModel):
    id: str
    backend: str
    context_length: int | None = None
    owned_by: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelCapabilities(GatewayCoreModel):
    model: str
    backend: str
    context_length: int | None = None
    supports_stream: bool = False
    supports_tools: bool = False
    supports_cancel: bool = False
    supports_parallel_tool_calls: bool = False
    supports_reasoning: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class BackendHealth(GatewayCoreModel):
    ready: bool
    status: Literal["ready", "degraded", "unavailable"]
    details: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
