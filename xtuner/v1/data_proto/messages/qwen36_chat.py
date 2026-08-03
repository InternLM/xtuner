# Copyright (c) OpenMMLab. All rights reserved.
import json
from typing import Dict, List

from xtuner.v1.data_proto.messages.qwen35_chat import (
    Qwen35ChatMessages,
    qwen35_tokenize_fn_fastspeed,
    qwen35_tokenize_fn_slowspeed,
)


def _render_tool_call_args(arguments: dict) -> str:
    """Render tool arguments with the Qwen3.6 JSON scalar semantics."""
    parts = ""
    for key, value in arguments.items():
        parts += f"<parameter={key}>\n"
        if isinstance(value, str):
            parts += value
        else:
            parts += json.dumps(value, ensure_ascii=False)
        parts += "\n</parameter>\n"
    return parts


def qwen36_tokenize_fn_fastspeed(
    messages,
    tokenizer=None,
    tools=None,
    add_generation_prompt=False,
    add_vision_id=False,
    return_labels=True,
):
    """Use the Qwen3.5 training renderer with Qwen3.6 tool serialization."""
    return qwen35_tokenize_fn_fastspeed(
        messages,
        tokenizer=tokenizer,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        add_vision_id=add_vision_id,
        return_labels=return_labels,
        _tool_call_args_renderer=_render_tool_call_args,
    )


def qwen36_tokenize_fn_slowspeed(
    tokenizer,
    messages: List[Dict[str, str]],
    tools=None,
    add_vision_id=False,
    **kwargs,
):
    """Reference renderer for comparison with the modified HF template."""
    kwargs["preserve_thinking"] = True
    return qwen35_tokenize_fn_slowspeed(
        tokenizer,
        messages,
        tools=tools,
        add_vision_id=add_vision_id,
        **kwargs,
    )


class Qwen36ChatMessages(Qwen35ChatMessages):
    """Qwen3.6 messages; all behavior except tool serialization is Qwen3.5."""

    def _tokenize_chat(self, tokenizer, add_vision_id):
        return qwen36_tokenize_fn_fastspeed(
            self.messages,
            tokenizer,
            self.tools,
            add_vision_id=add_vision_id,
            return_labels=True,
        )
