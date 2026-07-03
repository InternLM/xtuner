# Copyright (c) OpenMMLab. All rights reserved.
import json
from collections.abc import Mapping
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from transformers import PreTrainedTokenizer
from xtuner.v1.utils import IGNORE_INDEX


_MEDIA_TYPES = {"image", "image_url", "video", "video_url", "audio", "audio_url", "input_audio"}


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
            elif item_type in _MEDIA_TYPES:
                media_type = item_type.replace("_url", "").replace("input_", "")
                text += (
                    f"<reminder>You are unable to process this {media_type} because you don't have multi-modal input "
                    "ability. Try different methods.</reminder>"
                )
        return text
    return str(content)


def _tool_to_json(tool: Mapping[str, Any]) -> str:
    parts = []
    for key, value in tool.items():
        if key in {"defer_loading", "strict"}:
            continue
        parts.append(f'"{key}": {json.dumps(value, ensure_ascii=False)}')
    return "{" + ", ".join(parts) + "}"


def _render_tools(tools: List[Dict[str, Any]]) -> str:
    text = (
        "<|system|>\n"
        "# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        "<tools>\n"
    )
    for tool in tools:
        tool = tool["function"] if "function" in tool else tool
        if tool.get("defer_loading"):
            continue
        text += _tool_to_json(tool) + "\n"
    text += (
        "</tools>\n\n"
        "For each function call, output the function name and arguments within the following XML format:\n"
        "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>"
        "<arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>"
    )
    return text


def _assistant_content_and_reasoning(message: Mapping[str, Any]) -> tuple[str, str | None]:
    content = _visible_text(message.get("content", ""))
    reasoning_content = message.get("reasoning_content")
    if isinstance(reasoning_content, str):
        return content, reasoning_content
    if "</think>" in content:
        reasoning_content = content.split("</think>")[0].split("<think>")[-1]
        content = content.split("</think>")[-1]
        return content, reasoning_content
    return content, None


def _tool_call_args(arguments: Mapping[str, Any]) -> str:
    text = ""
    for key, value in arguments.items():
        text += f"<arg_key>{key}</arg_key><arg_value>"
        text += value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        text += "</arg_value>"
    return text


def _render_tool_calls(tool_calls: List[Dict[str, Any]]) -> str:
    text = ""
    for tool_call in tool_calls:
        tool_call = tool_call["function"] if "function" in tool_call else tool_call
        text += f"<tool_call>{tool_call['name']}"
        text += _tool_call_args(tool_call["arguments"])
        text += "</tool_call>"
    return text


def _render_tool_result(content: Any, tools: Optional[List[Dict[str, Any]]]) -> str:
    if isinstance(content, str):
        return f"<tool_response>{content}</tool_response>"
    if isinstance(content, list) and content and isinstance(content[0], Mapping):
        if content[0].get("type") == "tool_reference":
            text = "<tool_response><tools>\n"
            for tool_ref in content:
                for tool in tools or []:
                    tool = tool["function"] if "function" in tool else tool
                    if tool.get("name") == tool_ref.get("name"):
                        text += _tool_to_json(tool) + "\n"
            return text + "</tools></tool_response>"
        if "output" in content[0]:
            return "".join(f"<tool_response>{tool_result['output']}</tool_response>" for tool_result in content)
    return f"<tool_response>{_visible_text(content)}</tool_response>"


def _assistant_generated_segments(
    message: Mapping[str, Any],
    message_index: int,
    last_user_index: int,
    clear_thinking: bool | None,
) -> list[str]:
    if not message.get("loss", True):
        return []

    content, reasoning_content = _assistant_content_and_reasoning(message)
    segments = []
    if reasoning_content is not None and (clear_thinking is False or message_index > last_user_index):
        # The opening <think> is template scaffolding. The trace and its closing tag are model-generated.
        segments.append(reasoning_content + "</think>")
    if content.strip():
        segments.append(content.strip())
    if message.get("tool_calls"):
        segments.append(_render_tool_calls(message["tool_calls"]))
    return segments


def render_glm52_chat(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    add_generation_prompt: bool = False,
    enable_thinking: bool | None = None,
    reasoning_effort: str | None = None,
    clear_thinking: bool | None = None,
) -> tuple[str, list[bool]]:
    text = ""
    loss_mask: list[bool] = []

    def append(value: str, loss: bool) -> None:
        nonlocal text
        text += value
        loss_mask.extend([loss] * len(value))

    append("[gMASK]<sop>", False)
    if enable_thinking is not False:
        effective_reasoning_effort = "high" if reasoning_effort == "high" else "max"
        append(f"<|system|>Reasoning Effort: {effective_reasoning_effort.capitalize()}", False)
    if tools:
        append(_render_tools(tools), False)

    last_user_index = -1
    for index, message in enumerate(messages):
        if message.get("role") == "user":
            last_user_index = index

    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "user":
            append(f"<|user|>{_visible_text(message.get('content', ''))}", False)
        elif role == "system":
            append(f"<|system|>{_visible_text(message.get('content', ''))}", False)
        elif role == "assistant":
            content, reasoning_content = _assistant_content_and_reasoning(message)
            loss = bool(message.get("loss", True))

            append("<|assistant|>", False)
            if reasoning_content is not None and (clear_thinking is False or index > last_user_index):
                append("<think>", False)
                append(reasoning_content + "</think>", loss)
            else:
                append("<think></think>", False)
            if content.strip():
                append(content.strip(), loss)
            if message.get("tool_calls"):
                append(_render_tool_calls(message["tool_calls"]), loss)
        elif role == "tool":
            if index == 0 or messages[index - 1].get("role") != "tool":
                append("<|observation|>", False)
            append(_render_tool_result(message.get("content", ""), tools), False)

    if add_generation_prompt:
        append("<|assistant|>", False)
        append("<think></think>" if enable_thinking is False else "<think>", False)

    return text, loss_mask


def _tokenize_with_loss_mask(
    tokenizer: PreTrainedTokenizer, text: str, loss_mask: list[bool]
) -> tuple[list[int], list[int]]:
    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    input_ids = encoded["input_ids"]
    labels = []
    for token_id, (start, end) in zip(input_ids, encoded["offset_mapping"]):
        if start == end or not any(loss_mask[start:end]):
            labels.append(IGNORE_INDEX)
        else:
            labels.append(token_id)
    return input_ids, labels


def glm52_tokenize_fn_slowspeed(
    tokenizer: PreTrainedTokenizer,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    add_generation_prompt: bool = False,
    enable_thinking: bool | None = None,
    reasoning_effort: str | None = None,
    clear_thinking: bool | None = None,
    **kwargs,
) -> tuple[list[int], list[int]]:
    hf_kwargs: dict[str, Any] = dict(
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        tools=tools,
    )
    if enable_thinking is not None:
        hf_kwargs["enable_thinking"] = enable_thinking
    if reasoning_effort is not None:
        hf_kwargs["reasoning_effort"] = reasoning_effort
    if clear_thinking is not None:
        hf_kwargs["clear_thinking"] = clear_thinking
    hf_kwargs.update(kwargs)

    text = tokenizer.apply_chat_template(messages, **hf_kwargs)
    loss_mask = [False] * len(text)

    last_user_index = -1
    for index, message in enumerate(messages):
        if message.get("role") == "user":
            last_user_index = index

    cursor = 0
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        assistant_start = text.find("<|assistant|>", cursor)
        if assistant_start == -1:
            raise ValueError("Could not align assistant message in GLM-5.2 chat template rendering.")
        cursor = assistant_start + len("<|assistant|>")
        for segment in _assistant_generated_segments(message, index, last_user_index, clear_thinking):
            if not segment:
                continue
            start = text.find(segment, cursor)
            if start == -1:
                raise ValueError(f"Could not align assistant-generated segment in GLM-5.2 chat template: {segment!r}")
            end = start + len(segment)
            loss_mask[start:end] = [True] * (end - start)
            cursor = end

    return _tokenize_with_loss_mask(tokenizer, text, loss_mask)


class Glm52ChatMessages(BaseModel):
    model_config = ConfigDict(extra="forbid")
    messages: List[Dict[str, Any]]
    tools: Optional[List[Dict[str, Any]]] = None

    def tokenize(
        self,
        tokenizer: PreTrainedTokenizer,
        chat_template=None,
        add_generation_prompt: bool = False,
        enable_thinking: bool | None = None,
        reasoning_effort: str | None = None,
        clear_thinking: bool | None = None,
        **kwargs,
    ) -> Dict:
        # Keep default-system semantics aligned with the Qwen3.5 message path.
        if chat_template is not None and chat_template.default_system is not None:
            if self.messages[0]["role"] == "system":
                self.messages[0]["content"] = chat_template.default_system
            else:
                self.messages.insert(0, {"role": "system", "content": chat_template.default_system})

        text, loss_mask = render_glm52_chat(
            self.messages,
            tools=self.tools,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            clear_thinking=clear_thinking,
        )
        input_ids, labels = _tokenize_with_loss_mask(tokenizer, text, loss_mask)
        return {"input_ids": input_ids, "labels": labels}
