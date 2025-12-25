# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

from pydantic import BaseModel, ConfigDict, field_validator


class ChatTemplate(BaseModel):
    """Define a Pydantic data model for a hybrid chat with attributes for
    system, user and assistant chat as well as function and interpreter calls
    and results."""

    # Normal Chat
    model_config = ConfigDict(extra="forbid")
    meta_instruction: str = ""
    system: str  # System message format
    user: str  # User message format
    assistant: str  # Assistant message format
    stop_words: List[str]  # List of stop words
    sep: str = "\n"
    thinking: str | None = None  # Thinking message format, not role
    default_system: str | None = None
    tool_prompt: str | None = None  # Tool prompt format

    # only compute loss on the last assistant response ignoring the multiple rounds of assistant
    only_last_assistant_loss: bool = False  # gpt_oss is True
    loss_assistant_format_mapping: Dict[str, str] | None = None  # gpt_oss is {'<|end|>': '<|return|>'}

    def decorate_system(self, text: str) -> str:
        """Decorate text with the `system` template."""
        return self.system.format(system=text)

    def decorate_assistant(self, text: str) -> str:
        """Decorate text with the `assistant` template."""
        return self.assistant.format(assistant=text)

    def decorate_thinking(self, text: str) -> str:
        """Decorate text with the `thinking` template."""
        if self.thinking is None:
            raise ValueError("thinking template is not defined.")
        return self.thinking.format(assistant=text)

    def decorate_user(self, text: str) -> str:
        """Decorate text with the `user` template."""
        return self.user.format(user=text)

    @field_validator("system")
    def check_system(cls, v: str) -> str:
        """Validate that `system` contains '{system}'.

        If not, raises a ValueError.
        """
        if v is not None and "{system}" not in v:
            raise ValueError("system must contain the keyword '{system}'")
        return v

    @field_validator("user")
    def check_user(cls, v: str) -> str:
        """Validate that `user` contains '{user}'.

        If not, raises a ValueError.
        """
        if v is not None and "{user}" not in v:
            raise ValueError("user must contain the keyword '{user}'")
        return v

    @field_validator("assistant")
    def check_assistant(cls, v: str) -> str:
        """Validate that `assistant` contains '{assistant}'.

        If not, raises a ValueError.
        """
        if v is not None and "{assistant}" not in v:
            raise ValueError("assistant must contain the keyword '{assistant}'")
        return v
