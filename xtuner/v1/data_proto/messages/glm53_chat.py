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
"""

from collections.abc import Mapping
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict

from transformers import PreTrainedTokenizer
from xtuner.v1.utils import IGNORE_INDEX

from .glm52_chat import (
    _NEXT_ROLE_STOP_TOKENS,
    _append_literal,
    _assistant_content_and_reasoning,
    _encode_glm_text,
    _render_tool_calls,
    _render_tool_result,
    _render_tools,
    _tokenize_with_loss_mask,
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


def _visible_text(content: Any, _literal_spans=None, _offset: int = 0) -> str:
    if isinstance(content, str):
        return _append_literal("", content, _literal_spans, _offset)
    if isinstance(content, list):
        text = ""
        for item in content:
            if isinstance(item, str):
                text = _append_literal(text, item, _literal_spans, _offset)
                continue
            if not isinstance(item, Mapping):
                continue
            item_type = item.get("type")
            if item_type == "text":
                text = _append_literal(text, item.get("text", ""), _literal_spans, _offset)
            elif item_type in _MEDIA_PLACEHOLDER:
                text += _MEDIA_PLACEHOLDER[item_type]
        return text
    return _append_literal("", str(content), _literal_spans, _offset)


def render_glm53_chat(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    add_generation_prompt: bool = False,
    enable_thinking: bool = True,
    reasoning_effort: Literal["low", "high", "max"] = "max",
    clear_thinking: bool = False,
    _literal_spans: list[tuple[int, int]] | None = None,
) -> tuple[str, list[bool]]:
    """渲染 GLM-5.3-Flash 对话；结构与 GLM-5.2 相同，媒体渲染为两段式 placeholder 标记。"""
    assert enable_thinking, (
        "GLM-5.3's official chat template ignores enable_thinking: the Reasoning Effort header "
        "and the generation-time <think> prefix are always emitted, so enable_thinking=False "
        "would be silently ignored. Pass True (the default) or drop the argument."
    )
    text = ""
    loss_mask: list[bool] = []

    def append(value: str, loss: bool, *, literal: bool = False) -> None:
        nonlocal text
        text = _append_literal(text, value, _literal_spans) if literal else text + value
        loss_mask.extend([loss] * len(value))

    append("[gMASK]<sop>", False)
    # The official GLM53 template always emits these, including when enable_thinking=False.
    effective_reasoning_effort = reasoning_effort if reasoning_effort in ("low", "high") else "max"
    append(f"<|system|>Reasoning Effort: {effective_reasoning_effort.capitalize()}", False)
    if tools:
        append(_render_tools(tools, _literal_spans, len(text)), False)

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
            append(_visible_text(message.get("content", ""), _literal_spans, len(text)), False)
        elif role == "system":
            append("<|system|>", False)
            append(_visible_text(message.get("content", ""), _literal_spans, len(text)), False)
        elif role == "assistant":
            content, reasoning_content = _assistant_content_and_reasoning(message)
            loss = bool(message.get("loss", True))
            render_reasoning = reasoning_content is not None and (not clear_thinking or index > last_user_index)
            loss = loss and (reasoning_content is None or render_reasoning)

            append("<|assistant|>", False)
            if render_reasoning:
                assert reasoning_content is not None
                append("<think>", False)
                append(reasoning_content, loss, literal=True)
                append("</think>", loss)
            else:
                append("<think>", False)
                append("</think>", loss)
            if content.strip():
                append(content.strip(), loss, literal=True)
            if message.get("tool_calls"):
                append(_render_tool_calls(message["tool_calls"], _literal_spans, len(text)), loss)
            previous_assistant_loss = loss
            next_role = messages[index + 1].get("role") if index + 1 < len(messages) else None
            if next_role not in _NEXT_ROLE_STOP_TOKENS:
                append(_END_OF_TEXT, loss)
        elif role == "tool":
            if index == 0 or messages[index - 1].get("role") != "tool":
                boundary_loss = (
                    index > 0 and messages[index - 1].get("role") == "assistant" and previous_assistant_loss
                )
                append(_NEXT_ROLE_STOP_TOKENS[role], boundary_loss)
            append(_render_tool_result(message.get("content", ""), tools, _literal_spans, len(text)), False)

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
    literal_spans: list[tuple[int, int]] = []
    text, loss_mask = render_glm53_chat(
        messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
        clear_thinking=clear_thinking,
        _literal_spans=literal_spans,
    )
    return _tokenize_with_loss_mask(tokenizer, text, loss_mask, literal_spans)


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
    full_spans: list[tuple[int, int]] = []
    full_text, _ = render_glm53_chat(
        messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
        clear_thinking=clear_thinking,
        _literal_spans=full_spans,
    )
    total_ids, _ = _encode_glm_text(tokenizer, full_text, full_spans)
    labels = [IGNORE_INDEX] * len(total_ids)

    curr_ptr = 0
    for index, message in enumerate(messages):
        if message.get("role") != "assistant" or not message.get("loss", True):
            continue

        prefix_spans: list[tuple[int, int]] = []
        prefix_text, _ = render_glm53_chat(
            messages[:index],
            tools=tools if index == 0 else None,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            clear_thinking=clear_thinking,
            _literal_spans=prefix_spans,
        )
        message_spans: list[tuple[int, int]] = []
        message_text, _ = render_glm53_chat(
            [m.copy() for m in messages[: index + 1]],
            tools=tools if index == 0 else None,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            clear_thinking=clear_thinking,
            _literal_spans=message_spans,
        )

        prefix_ids, _ = _encode_glm_text(tokenizer, prefix_text, prefix_spans)
        message_ids, _ = _encode_glm_text(tokenizer, message_text, message_spans)
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
