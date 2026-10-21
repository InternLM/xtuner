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
   Media blocks sort as well. The official template's ordering unit is the result entry, not
   the message: under can_sort it emits every entry matching a tool_call id in ``tool_calls``
   order, pulling entries out of aggregated ``{"output": ...}`` messages. Media payloads pair
   to markers positionally (the collector walks messages in order and
   ``glm53_vl_tokenize_fn._expand_marker`` pairs the i-th payload with the i-th marker), so
   the VLM tokenize fn first flattens the messages the same way via :func:`sort_tool_messages`,
   making the payload collection order follow the rendered marker order.
5. ``<|endoftext|>`` is appended only after the FINAL assistant message (the training stop
   suffix). The official template never emits EOS mid-conversation, so assistant→system and
   assistant→assistant boundaries get no extra token; only ``<|user|>``/``<|observation|>``
   act as supervised stop targets.
"""

from collections.abc import Mapping
from typing import Any, Dict, List, Literal, Optional

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


def _result_ids_of(message: Mapping[str, Any]) -> List[Any]:
    if _is_list_of_outputs(message):
        return [_id_of(entry) for entry in message["content"]]
    return [_id_of(message)]


def _sortable_call_ids(messages: List[Dict[str, Any]], start: int, end: int) -> Optional[List[Any]]:
    """``tool_calls`` ids to sort the block ``messages[start:end+1]`` by, or
    ``None``.

    The official template's can_sort rule holds when the block follows an assistant with tool_calls, every result id is
    present, unique within the block, and matches a tool_call id, and the tool_call ids themselves are present and
    unique.
    """
    tool_calls = None
    if start > 0 and messages[start - 1].get("role") == "assistant":
        tool_calls = messages[start - 1].get("tool_calls") or None
    if not tool_calls:
        return None
    result_ids: List[Any] = []
    for message in messages[start : end + 1]:
        result_ids.extend(_result_ids_of(message))
    call_ids = [_id_of(tool_call) for tool_call in tool_calls]
    if (
        any(not result_id for result_id in result_ids)
        or len(set(result_ids)) != len(result_ids)
        or any(not call_id for call_id in call_ids)
        or len(set(call_ids)) != len(call_ids)
        or any(result_id not in call_ids for result_id in result_ids)
    ):
        return None
    return call_ids


def sort_tool_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return ``messages`` with every sortable contiguous tool block flattened
    into the preceding assistant's ``tool_calls`` order (the official
    template's can_sort rule).

    The official template's ordering unit is the result entry, not the message: under can_sort it emits each entry
    matching a tool_call id in ``tool_calls`` order, pulling entries out of aggregated ``{"output": ...}`` messages.
    Media payloads pair to markers positionally, so the messages are flattened the same way up front: aggregated
    messages are split into single-entry shallow copies (other keys kept), plain messages are reused unchanged, and
    every entry lands in ``tool_calls`` order. Blocks that fail the can_sort rule are returned unchanged.

    数据级别示例（排序发生在第 4 级 entry，而非第 2 级 message）::

        # ── 级别 1: conversation = messages + tools(整条样本) ──
        tools = [
            {"type": "function", "function": {"name": "a", "parameters": {...}}},
            {"type": "function", "function": {"name": "b", "parameters": {...}}},
            {"type": "function", "function": {"name": "c", "parameters": {...}}},
        ]
        messages = [
            # ── 级别 2: message(messages 列表的元素,带 role) ──
            {"role": "user", "content": "查一下"},
            {"role": "assistant", "content": "", "tool_calls": [调用a, 调用b, 调用c]},  # 顺序: a, b, c

            # ── 级别 3: tool block(下面两条连续 tool message,渲染为一个 <|observation|> 块) ──
            {"role": "tool", "tool_call_id": "c", "content": [  # 分散形态: 这条 message 整体是一个 entry
                {"type": "text", "text": "RESULT_C"},           # ← 级别 5: content part
            ]},
            # ↓ 聚合形态: content 是 entry 数组,内含 2 个 entry,顺序 b, a(与调用序相反)
            #   测试用例 tool_aggregate_media_reversed 用的就是这种形态
            {"role": "tool", "content": [
                {"id": "b", "output": [                         # ← 级别 4: entry(id 在元素上)
                    {"type": "text", "text": "RESULT_B"},       # ← 级别 5: content part
                    {"type": "image_url", "image_url": {"url": "IMG_B"}},
                ]},
                {"id": "a", "output": [
                    {"type": "text", "text": "RESULT_A"},
                    {"type": "image_url", "image_url": {"url": "IMG_A"}},
                ]},
            ]},
            {"role": "assistant", "content": "ASSISTANT_ANSWER"},
        ]

        # 拉平结果: [entry a 的 message, entry b 的 message, entry c 的 message](顺序 == tool_calls)
    """
    out: List[Dict[str, Any]] = []
    index = 0
    while index < len(messages):
        if messages[index].get("role") != "tool":
            out.append(messages[index])
            index += 1
            continue
        end = index
        while end + 1 < len(messages) and messages[end + 1].get("role") == "tool":
            end += 1
        block = messages[index : end + 1]
        call_ids = _sortable_call_ids(messages, index, end)
        if call_ids is None:
            out.extend(block)
        else:
            position = {call_id: i for i, call_id in enumerate(call_ids)}
            units: List[Dict[str, Any]] = []
            for message in block:
                if _is_list_of_outputs(message):
                    units.extend({**message, "content": [entry]} for entry in message["content"])
                else:
                    units.append(message)
            out.extend(sorted(units, key=lambda unit: position[_result_ids_of(unit)[0]]))
        index = end + 1
    return out


def _render_tool_output(output: Any, tools: Optional[List[Dict[str, Any]]]) -> str:
    """Render one aggregated ``{output: ...}`` entry the way the official template's
    render_tool_response does: a tool_reference list expands to the referenced tool
    schemas, anything else renders through ``_visible_text`` (media parts become the
    two-stage placeholder markers)."""
    if (
        isinstance(output, list)
        and output
        and isinstance(output[0], Mapping)
        and output[0].get("type") == "tool_reference"
    ):
        return _render_tool_result(output, tools)
    return f"<tool_response>{_visible_text(output)}</tool_response>"


def _render_tool_result(content: Any, tools: Optional[List[Dict[str, Any]]]) -> str:
    """GLM-5.3 version of the tool-result renderer.

    GLM-5.2 has no multimodal input, so its renderer emits media as a ``<reminder>`` text; the official GLM-5.3
    template renders media parts in tool content as the two-stage placeholder markers instead, so this version goes
    through GLM-5.3's ``_visible_text``.
    """
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
            return "".join(_render_tool_output(entry["output"], tools) for entry in content)
    return f"<tool_response>{_visible_text(content)}</tool_response>"


def _render_tool_block(messages: List[Dict[str, Any]], start: int, tools: Optional[List[Dict[str, Any]]]) -> str:
    """Render the contiguous tool-message block starting at ``start``.

    Results are emitted in the preceding assistant's ``tool_calls`` order when the official
    template's can_sort rule holds (see ``_sortable_call_ids``); any violation keeps the
    message order. Media blocks sort as well, matching the official template and the
    inference stack: ``glm53_vl_tokenize_fn`` reorders the messages themselves via
    ``sort_tool_messages`` before media collection, so payloads collected in message order
    follow the sorted marker order.
    """
    end = start
    while end + 1 < len(messages) and messages[end + 1].get("role") == "tool":
        end += 1
    block = messages[start : end + 1]

    call_ids = _sortable_call_ids(messages, start, end)
    if call_ids is None:
        return "".join(_render_tool_result(message.get("content", ""), tools) for message in block)

    text = ""
    for call_id in call_ids:
        for message in block:
            if _is_list_of_outputs(message):
                for entry in message["content"]:
                    if _id_of(entry) == call_id:
                        text += _render_tool_output(entry["output"], tools)
            elif _id_of(message) == call_id:
                text += _render_tool_result(message.get("content", ""), tools)
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
