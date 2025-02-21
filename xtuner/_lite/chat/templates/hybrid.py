# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional

from pydantic import BaseModel, field_validator


class HybridChatTemplate(BaseModel):
    """Define a Pydantic data model for a hybrid chat with attributes for
    system, user and assistant chat as well as function and interpreter calls
    and results."""

    # Normal Chat
    system: str  # System message format
    user: str  # User message format
    assistant: str  # Assistant message format
    stop_words: List[str]  # List of stop words
    sep: str = "\n"

    # Multimodal Chat
    # Predefined token and index for images
    image_token: str = "<image>"
    image_token_index: int = -100

    # Agent Chat

    # Interpreter and function related strings
    files: Optional[str] = None

    functions: Optional[str] = None  # Function description format
    function_call: Optional[str] = None  # Function call format
    function_result: Optional[str] = None  # Function result format

    code_interpreter: Optional[str] = None
    code_interpreter_call: Optional[str] = None  # Interpreter call format
    code_interpreter_result: Optional[str] = None  # Interpreter result format

    function_token: Optional[str] = None
    code_interpreter_token: Optional[str] = None
    action_start_token: Optional[str] = None
    action_end_token: Optional[str] = None

    @property
    def mm_token_maps(self) -> Dict[str, int]:
        """Return a dictionary that maps multimodal tokens to corresponding
        token indexes."""
        return {self.image_token: self.image_token_index}

    def decorate_system(self, text: str) -> str:
        """Decorate text with the `system` template."""
        return self.system.format(system=text)

    def decorate_assistant(self, text: str) -> str:
        """Decorate text with the `assistant` template."""
        return self.assistant.format(assistant=text)

    def decorate_user(self, text: str) -> str:
        """Decorate text with the `user` template."""
        return self.user.format(user=text)

    def decorate_files(self, text: str) -> str:
        """Decorate text with the `functions` template."""
        return self.files.format(files=text)

    def decorate_functions(self, text: str) -> str:
        """Decorate text with the `functions` template."""
        return self.functions.format(functions=text)

    def decorate_function_call(self, text: str, func: str) -> str:
        """Decorate text with the `function_call` template."""
        return self.function_call.format(assistant=text, function_call=func)

    def decorate_function_result(self, text: str) -> str:
        """Decorate text with the `function_result` template."""
        return self.function_result.format(function_result=text)

    def decorate_code_interpreter(self, text: str) -> str:
        """Decorate text with the `code_interpreter` template."""
        return self.code_interpreter.format(code_interpreter=text)

    def decorate_code_interpreter_call(self, text: str, func: str) -> str:
        """Decorate text with the `code_interpreter_call` template."""
        return self.code_interpreter_call.format(
            assistant=text, code_interpreter_call=func
        )

    def decorate_code_interpreter_result(self, text: str) -> str:
        """Decorate text with the `code_interpreter_result` template."""
        return self.code_interpreter_result.format(code_interpreter_result=text)

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

    @field_validator("function_call")
    def check_function_call(cls, v: str) -> str:
        """Validate that `function_call` contains '{function_call}'.

        If not, raises a ValueError.
        """
        if v is not None and "{function_call}" not in v and "{assistant}" not in v:
            raise ValueError(
                "function_call must contain the keywords '{function_call}'"
            )
        if v is not None and "{assistant}" not in v:
            raise ValueError(
                "function_call must contain the keyword '{assistant}' and "
                "'{function_call}'"
            )
        return v

    @field_validator("function_result")
    def check_function_result(cls, v: str) -> str:
        """Validate that `function_result` contains '{function_result}'.

        If not, raises a ValueError.
        """
        if v is not None and "{function_result}" not in v:
            raise ValueError(
                "function_result must contain the keyword '{function_result}'"
            )
        return v

    @field_validator("functions")
    def check_functions(cls, v: str) -> str:
        """Validate that `functions` contains '{functions}'.

        If not, raises a ValueError.
        """
        if v is not None and "{functions}" not in v:
            raise ValueError("functions must contain the keyword '{functions}'")
        return v

    @field_validator("code_interpreter")
    def check_code_interpreter(cls, v: str) -> str:
        """Validate that `code_interpreter` contains '{code_interpreter}'.

        If not, raises a ValueError.
        """
        if v is not None and "{code_interpreter}" not in v:
            raise ValueError(
                "code_interpreter must contain the keyword " "'{code_interpreter}'"
            )
        return v

    @field_validator("code_interpreter_call")
    def check_code_interpreter_call(cls, v: str) -> str:
        """Validate that `code_interpreter_call` contains
        '{code_interpreter_call}'.

        If not, raises a ValueError.
        """
        if (
            v is not None
            and "{code_interpreter_call}" not in v
            and "{assistant}" not in v
        ):
            raise ValueError(
                "code_interpreter_call must contain the keywords "
                "'{assistant}' and '{code_interpreter_call}'"
            )
        if v is not None and "{assistant}" not in v:
            raise ValueError(
                "code_interpreter_call must contain the keywords "
                "'{assistant}' and '{code_interpreter_call}'"
            )
        return v

    @field_validator("code_interpreter_result")
    def check_code_interpreter_result(cls, v: str) -> str:
        """Validate that `code_interpreter_result` contains
        '{code_interpreter_result}'.

        If not, raises a ValueError.
        """
        if v is not None and "{code_interpreter_result}" not in v:
            raise ValueError(
                "code_interpreter_result must contain the keyword "
                "'{code_interpreter_result}'"
            )
        return v
