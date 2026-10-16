# Copyright (c) OpenMMLab. All rights reserved.
"""GLM-5.3-Flash rendering, ordered tool results and SFT supervision.

User, observation and system boundaries belong to the preceding assistant's loss.
Consecutive assistant headers remain unsupervised, and only a final SFT assistant
receives endoftext. Clearing historical thinking also masks historical assistant
outputs. Media use GLM-5.3 placeholders; assistant reply images are unsupported.
"""

import json
from collections.abc import Mapping
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict

from transformers import PreTrainedTokenizer
from xtuner.v1.utils import IGNORE_INDEX

from .glm52_chat import _render_tools, _tokenize_with_loss_mask, _tool_to_json


_END_OF_TEXT = "<|endoftext|>"
_NEXT_ROLE_STOP_TOKENS = {"user": "<|user|>", "tool": "<|observation|>", "system": "<|system|>"}
_MEDIA_PLACEHOLDER = {
    "image": "<|begin_of_image|><|image|><|end_of_image|>",
    "image_url": "<|begin_of_image|><|image|><|end_of_image|>",
    "video": "<|begin_of_video|><|video|><|end_of_video|>",
    "video_url": "<|begin_of_video|><|video|><|end_of_video|>",
    "audio": "<|begin_of_audio|><|end_of_audio|>",
    "audio_url": "<|begin_of_audio|><|end_of_audio|>",
    "input_audio": "<|begin_of_audio|><|end_of_audio|>",
}


def _visible_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = ""
        for item in content:
            if isinstance(item, str):
                text += item
                continue
            if not isinstance(item, Mapping):
                continue
            item_type = item.get("type")
            if item_type == "text":
                text += item.get("text", "")
            elif item_type in _MEDIA_PLACEHOLDER:
                text += _MEDIA_PLACEHOLDER[item_type]
        return text
    return str(content)


def _assistant_content_and_reasoning(message: Mapping[str, Any]) -> tuple[str, str | None]:
    content_parts = message.get("content", "")
    if isinstance(content_parts, list) and any(
        isinstance(part, Mapping) and part.get("type") in ("image", "image_url") for part in content_parts
    ):
        raise NotImplementedError("GLM-5.3-Flash does not support images in assistant replies.")
    content = _visible_text(content_parts)
    reasoning_content = message.get("reasoning_content")
    if isinstance(reasoning_content, str):
        return content, reasoning_content
    if "</think>" in content:
        return content.split("</think>")[-1], content.split("</think>")[0].split("<think>")[-1]
    return content, None


def _render_tool_calls(tool_calls: List[Dict[str, Any]]) -> str:
    text = ""
    for tool_call in tool_calls:
        function = tool_call.get("function", tool_call)
        arguments = function["arguments"]
        if not isinstance(arguments, Mapping):
            raise ValueError(
                f"GLM-5.3-Flash tool {function['name']!r} arguments must be an object; "
                f"got {type(arguments).__name__}. Parse JSON strings before tokenization."
            )
        text += f"<tool_call>{function['name']}"
        for key, value in arguments.items():
            rendered_value = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            text += f"<arg_key>{key}</arg_key><arg_value>{rendered_value}</arg_value>"
        text += "</tool_call>"
    return text


def _tool_result_id(value: Mapping[str, Any]) -> str:
    result_id = value.get("tool_call_id") or value.get("id")
    return str(result_id) if result_id else ""


def _is_output_list(content: Any) -> bool:
    return isinstance(content, list) and bool(content) and isinstance(content[0], Mapping) and "output" in content[0]


def order_glm53_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ordered = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.get("role") != "tool":
            ordered.append(message)
            index += 1
            continue
        block_start = index
        while index < len(messages) and messages[index].get("role") == "tool":
            index += 1
        block = messages[block_start:index]
        previous = messages[block_start - 1] if block_start else {}
        calls = previous.get("tool_calls", []) if previous.get("role") == "assistant" else []
        call_ids = [_tool_result_id(call) for call in calls]
        entries = []
        for result in block:
            content = result.get("content", "")
            if _is_output_list(content):
                entries.extend((_tool_result_id(entry), {**result, "content": [entry]}) for entry in content)
            else:
                entries.append((_tool_result_id(result), result))
        result_ids = [result_id for result_id, result in entries]
        can_sort = (
            bool(call_ids)
            and all(call_ids)
            and len(set(call_ids)) == len(call_ids)
            and all(result_ids)
            and len(set(result_ids)) == len(result_ids)
            and set(result_ids).issubset(call_ids)
        )
        if can_sort:
            positions = {call_id: position for position, call_id in enumerate(call_ids)}
            ordered.extend(result for result_id, result in sorted(entries, key=lambda entry: positions[entry[0]]))
        else:
            ordered.extend(block)
    return ordered


def _render_tool_references(references: list, tools: Optional[List[Dict[str, Any]]]) -> str:
    text = "<tool_response><tools>\n"
    for reference in references:
        for tool in tools or []:
            function = tool.get("function", tool)
            if function.get("name") == reference.get("name"):
                text += _tool_to_json(function) + "\n"
    return text + "</tools></tool_response>"


def _render_tool_output(content: Any, tools: Optional[List[Dict[str, Any]]]) -> str:
    if isinstance(content, list) and content and isinstance(content[0], Mapping):
        if content[0].get("type") == "tool_reference":
            return _render_tool_references(content, tools)
    return f"<tool_response>{_visible_text(content)}</tool_response>"


def _render_tool_result(content: Any, tools: Optional[List[Dict[str, Any]]]) -> str:
    if _is_output_list(content):
        return "".join(_render_tool_output(entry["output"], tools) for entry in content)
    return _render_tool_output(content, tools)


def glm53_media_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    media_messages = []
    for message in order_glm53_messages(messages):
        role = message.get("role")
        if role == "assistant":
            _assistant_content_and_reasoning(message)
            continue
        if role not in ("user", "system", "tool", "pretrain"):
            continue
        content = message.get("content", "")
        outputs = [entry["output"] for entry in content] if role == "tool" and _is_output_list(content) else [content]
        for output in outputs:
            if isinstance(output, list):
                if output and isinstance(output[0], Mapping) and output[0].get("type") == "tool_reference":
                    continue
                media = [
                    part for part in output
                    if isinstance(part, Mapping) and part.get("type") in ("image", "image_url", "video", "video_url")
                ]
                if media:
                    media_messages.append({"role": "user", "content": media})
    return media_messages


def render_glm53_chat(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    add_generation_prompt: bool = False,
    reasoning_effort: Literal["low", "high", "max"] = "max",
    clear_thinking: bool = False,
) -> tuple[str, list[bool]]:
    """渲染 GLM-5.3-Flash 对话；结构与 GLM-5.2 相同，媒体渲染为两段式 placeholder 标记。"""
    messages = order_glm53_messages(messages)
    text = ""
    loss_mask: list[bool] = []

    def append(value: str, loss: bool) -> None:
        nonlocal text
        text += value
        loss_mask.extend([loss] * len(value))

    append("[gMASK]<sop>", False)
    effective_reasoning_effort = reasoning_effort if reasoning_effort in ("low", "high") else "max"
    append(f"<|system|>Reasoning Effort: {effective_reasoning_effort.capitalize()}", False)
    if tools:
        append(_render_tools(tools), False)

    last_user_index = -1
    for index, message in enumerate(messages):
        if message.get("role") == "user":
            last_user_index = index

    previous_assistant_loss = False
    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "user":
            boundary_loss = index > 0 and messages[index - 1].get("role") == "assistant" and previous_assistant_loss
            append(_NEXT_ROLE_STOP_TOKENS[role], boundary_loss)
            append(_visible_text(message.get("content", "")), False)
        elif role == "system":
            boundary_loss = index > 0 and messages[index - 1].get("role") == "assistant" and previous_assistant_loss
            append("<|system|>", boundary_loss)
            append(_visible_text(message.get("content", "")), False)
        elif role == "assistant":
            content, reasoning_content = _assistant_content_and_reasoning(message)
            loss = bool(message.get("loss", True)) and (not clear_thinking or index > last_user_index)
            render_reasoning = reasoning_content is not None and (not clear_thinking or index > last_user_index)

            append("<|assistant|>", False)
            if render_reasoning:
                assert reasoning_content is not None
                append("<think>", False)
                append(reasoning_content + "</think>", loss)
            else:
                append("<think>", False)
                append("</think>", loss)
            if content.strip():
                append(content.strip(), loss)
            if message.get("tool_calls"):
                append(_render_tool_calls(message["tool_calls"]), loss)
            previous_assistant_loss = loss
            next_role = messages[index + 1].get("role") if index + 1 < len(messages) else None
            if next_role is None and not add_generation_prompt:
                append(_END_OF_TEXT, loss)
        elif role == "tool":
            if index == 0 or messages[index - 1].get("role") != "tool":
                boundary_loss = (
                    index > 0 and messages[index - 1].get("role") == "assistant" and previous_assistant_loss
                )
                append(_NEXT_ROLE_STOP_TOKENS[role], boundary_loss)
            append(_render_tool_result(message.get("content", ""), tools), False)

    if add_generation_prompt:
        append("<|assistant|>", False)
        append("<think>", False)

    return text, loss_mask


def glm53_tokenize_fn_fastspeed(
    tokenizer: PreTrainedTokenizer,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    add_generation_prompt: bool = False,
    reasoning_effort: Literal["low", "high", "max"] = "max",
    clear_thinking: bool = False,
    **kwargs,
) -> tuple[list[int], list[int]]:
    text, loss_mask = render_glm53_chat(
        messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        reasoning_effort=reasoning_effort,
        clear_thinking=clear_thinking,
    )
    return _tokenize_with_loss_mask(tokenizer, text, loss_mask)


def glm53_tokenize_fn_slowspeed(
    tokenizer: PreTrainedTokenizer,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    add_generation_prompt: bool = False,
    reasoning_effort: Literal["low", "high", "max"] = "max",
    clear_thinking: bool = False,
    **kwargs,
) -> tuple[list[int], list[int]]:
    """慢速 golden 参考实现，见 ``glm52_tokenize_fn_slowspeed`` 的 docstring；算法相同。"""
    full_text, _ = render_glm53_chat(
        messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        reasoning_effort=reasoning_effort,
        clear_thinking=clear_thinking,
    )
    total_ids = tokenizer.encode(full_text, add_special_tokens=False)
    labels = [IGNORE_INDEX] * len(total_ids)

    messages = order_glm53_messages(messages)
    last_user_index = max((index for index, message in enumerate(messages) if message.get("role") == "user"), default=-1)
    for index, message in enumerate(messages):
        if (
            message.get("role") != "assistant"
            or not message.get("loss", True)
            or (clear_thinking and index < last_user_index)
        ):
            continue
        content, reasoning = _assistant_content_and_reasoning(message)
        prefix_text, _ = render_glm53_chat(
            messages[:index],
            tools=tools,
            add_generation_prompt=True,
            reasoning_effort=reasoning_effort,
            clear_thinking=clear_thinking,
        )
        generated = reasoning + "</think>" if reasoning is not None else "</think>"
        generated += content.strip()
        if message.get("tool_calls"):
            generated += _render_tool_calls(message["tool_calls"])
        next_role = messages[index + 1].get("role") if index + 1 < len(messages) else None
        if next_role in _NEXT_ROLE_STOP_TOKENS:
            generated += _NEXT_ROLE_STOP_TOKENS[next_role]
        elif next_role is None and not add_generation_prompt:
            generated += _END_OF_TEXT
        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        content_ids = tokenizer.encode(prefix_text + generated, add_special_tokens=False)[len(prefix_ids) :]
        if not content_ids:
            continue
        start = len(prefix_ids)
        if total_ids[:start] != prefix_ids or total_ids[start : start + len(content_ids)] != content_ids:
            raise ValueError("Could not align assistant message in GLM-5.3-Flash chat template rendering.")
        labels[start : start + len(content_ids)] = content_ids

    return total_ids, labels


class Glm53ChatMessages(BaseModel):
    model_config = ConfigDict(extra="forbid")
    messages: List[Dict[str, Any]]
    tools: Optional[List[Dict[str, Any]]] = None

    def tokenize(
        self,
        tokenizer: PreTrainedTokenizer,
        chat_template=None,
        add_generation_prompt: bool = False,
        reasoning_effort: Literal["low", "high", "max"] = "max",
        clear_thinking: bool = False,
        **kwargs,
    ) -> Dict:
        if chat_template is not None and chat_template.default_system is not None:
            if self.messages[0]["role"] == "system":
                self.messages[0]["content"] = chat_template.default_system
            else:
                self.messages.insert(0, {"role": "system", "content": chat_template.default_system})

        input_ids, labels = glm53_tokenize_fn_fastspeed(
            tokenizer,
            self.messages,
            tools=self.tools,
            add_generation_prompt=add_generation_prompt,
            reasoning_effort=reasoning_effort,
            clear_thinking=clear_thinking,
            **kwargs,
        )
        return {"input_ids": input_ids, "labels": labels}
