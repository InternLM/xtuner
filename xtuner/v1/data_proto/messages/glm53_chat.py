# Copyright (c) OpenMMLab. All rights reserved.
"""GLM-5.3-Flash chat template.

Same SFT loss-mask semantics as :mod:`glm52_chat` (``<|user|>``/``<|observation|>`` are the
stop targets whose loss belongs to the previous assistant turn; a trailing assistant with no
following role boundary gets an explicit ``<|endoftext|>``). Matches these GLM-5.3 details
in the official ``chat_template.jinja``:

1. ``reasoning_effort`` domain is ``{low, high, max}`` instead of ``{high, max}``; any value
   outside ``{low, high}`` (including "max" itself) renders as "Max", matching the jinja's
   ``if ... in ['low', 'high'] else 'max'`` fallback.
2. media renders as the GLM-5.3 two-stage placeholder protocol (see
   ``glm53_vl_tokenize_fn.py`` for the expansion step that turns these single markers into the
   real per-patch token spans): ``<|begin_of_image|><|image|><|end_of_image|>`` /
   ``<|begin_of_video|><|video|><|end_of_video|>`` / ``<|begin_of_audio|><|end_of_audio|>``,
   instead of GLM-5.2's ``<reminder>...</reminder>`` text (GLM-5.2 has no multimodal input).
3. ``enable_thinking`` is accepted for API compatibility but, like the official template,
   does not remove the reasoning-effort header or close the generation-time ``<think>`` prefix.
   Because the flag has no effect, ``False`` is rejected with an ``AssertionError`` instead of
   being silently ignored.
4. A contiguous block of tool messages is rendered in the preceding assistant's ``tool_calls``
   order when the official template's can_sort rule holds (every result id is present, unique
   within the block, and matches a unique tool_call id); otherwise the message order is kept.
   A block containing media always keeps the message order: media payloads pair to markers
   positionally (the data collector walks messages in order and
   ``glm53_vl_tokenize_fn._expand_marker`` pairs the i-th payload with the i-th marker), so
   reordering would silently swap payloads. The official template reorders media blocks too,
   but the official inference stack pairs images to markers positionally as well
   (``transformers.processing_utils.get_text_with_replacements``), so that combination is
   ill-defined upstream rather than a behavior training must mirror.
5. Structured ``{"output": ...}`` entries render their output through ``visible_text`` and a
   nested ``tool_reference`` list expands the matching tool definitions into
   ``<tool_response><tools>...</tools></tool_response>`` (GLM-5.2's official template has
   neither behavior, so these live here rather than in :mod:`glm52_chat`).
6. ``<|endoftext|>`` is appended only after the FINAL assistant message (the training stop
   suffix). The official template never emits EOS mid-conversation, so assistant→system and
   assistant→assistant boundaries get no extra token; only ``<|user|>``/``<|observation|>``
   act as supervised stop targets.
"""

from collections.abc import Mapping
from typing import Any, Dict, List, Literal, Optional, TypeGuard

from pydantic import BaseModel, ConfigDict

from transformers import PreTrainedTokenizer
from xtuner.v1.utils import IGNORE_INDEX

from .glm52_chat import (
    _NEXT_ROLE_STOP_TOKENS,
    _assistant_content_and_reasoning,
    _render_tool_calls,
    _render_tools,
    _tokenize_with_loss_mask,
    _tool_to_json,
)


_END_OF_TEXT = "<|endoftext|>"
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


def _id_of(obj: Mapping[str, Any]) -> Any:
    return obj.get("tool_call_id") or obj.get("id")


def _is_list_of_outputs(message: Mapping[str, Any]) -> bool:
    content = message.get("content")
    return (
        isinstance(content, list) and len(content) > 0 and isinstance(content[0], Mapping) and "output" in content[0]
    )


def _contains_media(content: Any) -> bool:
    """Media items anywhere in ``content``: top-level parts or nested ``{output: ...}`` entries."""
    if isinstance(content, Mapping):
        if content.get("type") in _MEDIA_PLACEHOLDER:
            return True
        return _contains_media(content.get("output"))
    if isinstance(content, list):
        return any(_contains_media(item) for item in content)
    return False


def _is_tool_reference_list(value: Any) -> TypeGuard[List[Mapping[str, Any]]]:
    return (
        isinstance(value, list)
        and len(value) > 0
        and isinstance(value[0], Mapping)
        and value[0].get("type") == "tool_reference"
    )


def _tool_references_to_response(refs: List[Mapping[str, Any]], tools: Optional[List[Dict[str, Any]]]) -> str:
    text = "<tool_response><tools>\n"
    for ref in refs:
        for tool in tools or []:
            tool = tool["function"] if "function" in tool else tool
            if tool.get("name") == ref.get("name"):
                text += _tool_to_json(tool) + "\n"
    return text + "</tools></tool_response>"


def _render_output_entry(entry: Mapping[str, Any], tools: Optional[List[Dict[str, Any]]]) -> str:
    output = entry.get("output")
    if _is_tool_reference_list(output):
        return _tool_references_to_response(output, tools)
    return f"<tool_response>{_visible_text(output)}</tool_response>"


def _render_tool_response(message: Mapping[str, Any], tools: Optional[List[Dict[str, Any]]]) -> str:
    """GLM-5.3's ``render_tool_response`` macro: structured outputs render
    their visible content and nested ``tool_reference`` lists expand tool
    definitions (GLM-5.2's official template stringifies both, so the shared
    glm52 helper cannot be reused here)."""
    content = message.get("content", "")
    if isinstance(content, str):
        return f"<tool_response>{content}</tool_response>"
    if _is_tool_reference_list(content):
        return _tool_references_to_response(content, tools)
    if _is_list_of_outputs(message):
        return "".join(_render_output_entry(entry, tools) for entry in content)
    return f"<tool_response>{_visible_text(content)}</tool_response>"


def _render_tool_block(messages: List[Dict[str, Any]], start: int, tools: Optional[List[Dict[str, Any]]]) -> str:
    """Render the contiguous tool-message block starting at ``start``.

    Results are emitted in the preceding assistant's ``tool_calls`` order when the official
    template's can_sort rule holds: the block follows an assistant with tool_calls, every
    result id is present, unique within the block, and matches a tool_call id, and the
    tool_call ids themselves are present and unique. Any violation keeps the message order,
    as does any block containing media: media payloads pair to markers positionally (the
    collector walks messages in order), so reordering such a block would attach payloads to
    the wrong markers.
    """
    end = start
    while end + 1 < len(messages) and messages[end + 1].get("role") == "tool":
        end += 1
    block = messages[start : end + 1]

    tool_calls = None
    if start > 0 and messages[start - 1].get("role") == "assistant":
        tool_calls = messages[start - 1].get("tool_calls") or None

    can_sort = bool(tool_calls)
    if can_sort:
        result_ids: list[Any] = []
        for message in block:
            if _is_list_of_outputs(message):
                result_ids.extend(_id_of(entry) for entry in message["content"])
            else:
                result_ids.append(_id_of(message))
        call_ids = [_id_of(tool_call) for tool_call in tool_calls or []]
        if (
            any(not result_id for result_id in result_ids)
            or len(set(result_ids)) != len(result_ids)
            or any(not call_id for call_id in call_ids)
            or len(set(call_ids)) != len(call_ids)
            or any(result_id not in call_ids for result_id in result_ids)
        ):
            can_sort = False
    if can_sort and any(_contains_media(message.get("content")) for message in block):
        # 媒体 payload 与 marker 是位置式配对：collector 按 messages 顺序收集，
        # glm53_vl_tokenize_fn._expand_marker 把第 i 个 payload 换给第 i 个 marker。
        # 重排只改 marker 顺序，会把图像换到别的 tool 结果上，且占位符计数校验发现不了。
        # 官方模板无视媒体照样重排，但官方推理栈的媒体配对同样按位置
        # （transformers processing_utils.get_text_with_replacements），该组合在上游无自洽定义。
        can_sort = False

    if not can_sort:
        return "".join(_render_tool_response(message, tools) for message in block)

    text = ""
    for call_id in call_ids:
        for message in block:
            if _is_list_of_outputs(message):
                for entry in message["content"]:
                    if _id_of(entry) == call_id:
                        text += _render_output_entry(entry, tools)
            elif _id_of(message) == call_id:
                text += _render_tool_response(message, tools)
    return text


def render_glm53_chat(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    add_generation_prompt: bool = False,
    enable_thinking: bool = True,
    reasoning_effort: Literal["low", "high", "max"] = "max",
    clear_thinking: bool = False,
) -> tuple[str, list[bool]]:
    """渲染 GLM-5.3-Flash 对话；结构与 GLM-5.2 相同，媒体渲染为两段式 placeholder 标记。"""
    assert enable_thinking, (
        "GLM-5.3's official chat template ignores enable_thinking: the Reasoning Effort header "
        "and the generation-time <think> prefix are always emitted, so enable_thinking=False "
        "would be silently ignored. Pass True (the default) or drop the argument."
    )
    text = ""
    loss_mask: list[bool] = []

    def append(value: str, loss: bool) -> None:
        nonlocal text
        text += value
        loss_mask.extend([loss] * len(value))

    append("[gMASK]<sop>", False)
    # The official GLM53 template always emits these, including when enable_thinking=False.
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
            append(f"<|system|>{_visible_text(message.get('content', ''))}", False)
        elif role == "assistant":
            content, reasoning_content = _assistant_content_and_reasoning(message)
            loss = bool(message.get("loss", True))
            render_reasoning = reasoning_content is not None and (not clear_thinking or index > last_user_index)
            loss = loss and (reasoning_content is None or render_reasoning)

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
            if next_role is None:
                # 训练用 EOS 只追加在对话末尾；官方模板从不在对话中间输出 EOS。
                append(_END_OF_TEXT, loss)
        elif role == "tool":
            if index == 0 or messages[index - 1].get("role") != "tool":
                boundary_loss = (
                    index > 0 and messages[index - 1].get("role") == "assistant" and previous_assistant_loss
                )
                append(_NEXT_ROLE_STOP_TOKENS[role], boundary_loss)
                append(_render_tool_block(messages, index, tools), False)
            # 块内后续 tool 消息已由块起点统一渲染（含按 tool_calls 顺序的重排）。

    if add_generation_prompt:
        append("<|assistant|>", False)
        append("<think>", False)

    return text, loss_mask


def glm53_tokenize_fn_fastspeed(
    tokenizer: PreTrainedTokenizer,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    add_generation_prompt: bool = False,
    enable_thinking: bool = True,
    reasoning_effort: Literal["low", "high", "max"] = "max",
    clear_thinking: bool = False,
    **kwargs,
) -> tuple[list[int], list[int]]:
    text, loss_mask = render_glm53_chat(
        messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
        clear_thinking=clear_thinking,
    )
    return _tokenize_with_loss_mask(tokenizer, text, loss_mask)


def glm53_tokenize_fn_slowspeed(
    tokenizer: PreTrainedTokenizer,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    add_generation_prompt: bool = False,
    enable_thinking: bool = True,
    reasoning_effort: Literal["low", "high", "max"] = "max",
    clear_thinking: bool = False,
    **kwargs,
) -> tuple[list[int], list[int]]:
    """慢速 golden 参考实现，见 ``glm52_tokenize_fn_slowspeed`` 的 docstring；算法相同。"""
    full_text, _ = render_glm53_chat(
        messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
        clear_thinking=clear_thinking,
    )
    total_ids = tokenizer.encode(full_text, add_special_tokens=False)
    labels = [IGNORE_INDEX] * len(total_ids)

    curr_ptr = 0
    for index, message in enumerate(messages):
        if message.get("role") != "assistant" or not message.get("loss", True):
            continue

        prefix_text, _ = render_glm53_chat(
            messages[:index],
            tools=tools if index == 0 else None,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            clear_thinking=clear_thinking,
        )
        message_text, _ = render_glm53_chat(
            [m.copy() for m in messages[: index + 1]],
            tools=tools if index == 0 else None,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            clear_thinking=clear_thinking,
        )

        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        message_ids = tokenizer.encode(message_text, add_special_tokens=False)
        content_ids = message_ids[len(prefix_ids) :]
        if not content_ids:
            continue

        next_role = messages[index + 1].get("role") if index + 1 < len(messages) else None
        if next_role in _NEXT_ROLE_STOP_TOKENS:
            content_ids[-1] = tokenizer.convert_tokens_to_ids(_NEXT_ROLE_STOP_TOKENS[next_role])

        for start in range(curr_ptr, len(total_ids) - len(content_ids) + 1):
            if total_ids[start : start + len(content_ids)] == content_ids:
                labels[start : start + len(content_ids)] = content_ids
                curr_ptr = start + len(content_ids)
                break
        else:
            if index == len(messages) - 1:
                raise ValueError("Could not align final assistant message in GLM-5.3-Flash chat template rendering.")

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
        enable_thinking: bool = True,
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
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            clear_thinking=clear_thinking,
            **kwargs,
        )
        return {"input_ids": input_ids, "labels": labels}
