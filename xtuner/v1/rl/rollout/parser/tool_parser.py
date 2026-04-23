from __future__ import annotations

import ast
import json
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from xtuner.v1.data_proto.rl_data import RolloutFunctionCall, RolloutState, RolloutToolCall


class ParsedToolCallResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_calls: list[RolloutToolCall] = Field(default_factory=list)
    remaining_text: str = ""


class ToolCallParser(ABC):
    def parse(self, rollout_state: RolloutState) -> ParsedToolCallResult:
        return self.parse_text(rollout_state.response or "")

    def should_reject_unparsed_markup(
        self,
        *,
        has_tools: bool,
        text: str | None,
        parsed_tool_calls: list[Any] | None,
    ) -> bool:
        """Whether the remaining assistant text should be rejected as a
        malformed tool call.

        Most parsers do not use textual tool-call markup, so the default behavior is to accept the text. Parsers with
        format-specific markup can override this and reject outputs that still contain unparsed tool-call fragments.
        """
        return False

    @abstractmethod
    def parse_text(self, text: str) -> ParsedToolCallResult:
        raise NotImplementedError


def extract_tokenizer_token_contents(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | Any,
) -> set[str]:
    token_contents: set[str] = set()

    for token in getattr(tokenizer, "additional_special_tokens", []) or []:
        if isinstance(token, str):
            token_contents.add(token)

    added_tokens_decoder = getattr(tokenizer, "added_tokens_decoder", None)
    if isinstance(added_tokens_decoder, dict):
        for token_info in added_tokens_decoder.values():
            if isinstance(token_info, str):
                token_contents.add(token_info)
            elif isinstance(token_info, dict):
                content = token_info.get("content")
                if isinstance(content, str):
                    token_contents.add(content)
            else:
                content = getattr(token_info, "content", None)
                if isinstance(content, str):
                    token_contents.add(content)

    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        try:
            vocab = get_vocab()
        except Exception:
            vocab = None
        if isinstance(vocab, dict):
            token_contents.update(token for token in vocab if isinstance(token, str))

    return token_contents


def parse_json_or_python_mapping(raw_payload: str) -> Any:
    try:
        return json.loads(raw_payload)
    except Exception:
        try:
            return ast.literal_eval(raw_payload)
        except Exception:
            return None


def coerce_parameter_value(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return ""
    try:
        return json.loads(stripped)
    except Exception:
        try:
            return ast.literal_eval(stripped)
        except Exception:
            return stripped


def build_rollout_tool_call(
    *,
    name: str,
    arguments: Any,
    call_id: str,
) -> RolloutToolCall:
    raw_arguments_text = arguments if isinstance(arguments, str) else None
    parsed_arguments = arguments
    if isinstance(arguments, str):
        decoded = parse_json_or_python_mapping(arguments)
        parsed_arguments = decoded if decoded is not None else {"raw": arguments}
    return RolloutToolCall(
        id=call_id,
        function=RolloutFunctionCall(
            name=name,
            arguments=parsed_arguments,
            raw_arguments_text=raw_arguments_text,
        ),
    )
