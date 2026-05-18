from __future__ import annotations

from typing import Literal

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from .qwen3_reasoning_parser import Qwen3ReasoningParser, extract_qwen3_reasoning_strip_tokens
from .qwen3_tool_parser import Qwen3ToolCallParser
from .qwen3p5_tool_parser import Qwen3p5ToolCallParser
from .reasoning_parser import ReasoningParser
from .tool_parser import ToolCallParser


ToolCallParserName = Literal["none", "qwen3", "qwen3p5"]
ReasoningParserName = Literal["none", "qwen3"]


def build_tool_call_parser(parser_name: ToolCallParserName) -> ToolCallParser | None:
    if parser_name == "none":
        return None
    if parser_name == "qwen3":
        return Qwen3ToolCallParser()
    if parser_name == "qwen3p5":
        return Qwen3p5ToolCallParser()
    raise ValueError(f"Unsupported tool_call_parser: {parser_name}")


def build_reasoning_parser(
    parser_name: ReasoningParserName,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
) -> ReasoningParser | None:
    if parser_name == "none":
        return None
    if parser_name == "qwen3":
        return Qwen3ReasoningParser(strip_tokens=extract_qwen3_reasoning_strip_tokens(tokenizer))
    raise ValueError(f"Unsupported reasoning_parser: {parser_name}")
