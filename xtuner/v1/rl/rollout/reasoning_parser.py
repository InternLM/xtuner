from __future__ import annotations

import re
from abc import ABC, abstractmethod

from xtuner.v1.data_proto import ParsedReasoningResult, RolloutState


class ReasoningParser(ABC):
    @abstractmethod
    def parse(self, rollout_state: RolloutState) -> ParsedReasoningResult:
        """Return parsed reasoning and remaining text for a rollout
        response."""


class ThinkTagReasoningParser(ReasoningParser):
    _reasoning_pattern = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)

    def __init__(self, strip_tokens: list[str] | None = None):
        self._strip_tokens = strip_tokens or []

    def parse(self, rollout_state: RolloutState) -> ParsedReasoningResult:
        text = rollout_state.response or ""
        if not text:
            return ParsedReasoningResult()
        cleaned = text
        for token in self._strip_tokens:
            cleaned = cleaned.replace(token, "")
        reasoning_chunks = [
            match.group(1).strip() for match in self._reasoning_pattern.finditer(cleaned) if match.group(1).strip()
        ]
        content = self._reasoning_pattern.sub("", cleaned).strip()
        if not reasoning_chunks and "<think>" in cleaned:
            prefix, suffix = cleaned.split("<think>", 1)
            content = prefix.strip()
            truncated_reasoning = suffix.replace("</think>", "").strip()
            if truncated_reasoning:
                reasoning_chunks.append(truncated_reasoning)
        reasoning = "\n".join(reasoning_chunks).strip() or None
        return ParsedReasoningResult(reasoning_text=reasoning, remaining_text=content or None)
