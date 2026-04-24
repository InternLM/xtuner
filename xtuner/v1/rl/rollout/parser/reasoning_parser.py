from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict

from xtuner.v1.data_proto.rl_data import RolloutState


class ParsedReasoningResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning_text: str | None = None
    remaining_text: str | None = None


class ReasoningParser(ABC):
    @abstractmethod
    def parse(self, rollout_state: RolloutState) -> ParsedReasoningResult:
        """Return parsed reasoning and remaining text for a rollout
        response."""
