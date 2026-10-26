# Copyright (c) OpenMMLab. All rights reserved.
import copy
import json
from collections.abc import Mapping
from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict

from transformers import PreTrainedTokenizer
from xtuner.v1.utils import IGNORE_INDEX

from .qwen35_chat import get_offset_mapping


_MEDIA_TYPES = {"image", "image_url", "video", "video_url", "audio", "audio_url", "input_audio"}
_END_OF_TEXT = "<|endoftext|>"
_NEXT_ROLE_STOP_TOKENS = {"user": "<|user|>", "tool": "<|observation|>"}


def _append_literal(text: str, value: str, spans: list[tuple[int, int]] | None, offset: int = 0) -> str:
    if spans is not None and value:
        spans.append((offset + len(text), offset + len(text) + len(value)))
    return text + value


def _visible_text(content: Any, _literal_spans=None, _offset: int = 0, _media_placeholder=None) -> str:
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
            elif _media_placeholder is not None and item_type in _media_placeholder:
                text += _media_placeholder[item_type]
            elif item_type in _MEDIA_TYPES:
                media_type = item_type.replace("_url", "").replace("input_", "")
                text += (
                    f"<reminder>You are unable to process this {media_type} because you don't have multi-modal input "
                    "ability. Try different methods.</reminder>"
                )
        return text
    return _append_literal("", str(content), _literal_spans, _offset)


def _tool_to_json(tool: Mapping[str, Any]) -> str:
    parts = []
    for key, value in tool.items():
        if key in {"defer_loading", "strict"}:
            continue
        parts.append(f'"{key}": {json.dumps(value, ensure_ascii=False)}')
    return "{" + ", ".join(parts) + "}"


def _render_tools(tools: List[Dict[str, Any]], _literal_spans=None, _offset: int = 0) -> str:
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
        text = _append_literal(text, _tool_to_json(tool), _literal_spans, _offset) + "\n"
    text += (
        "</tools>\n\n"
        "For each function call, output the function name and arguments within the following XML format:\n"
        "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>"
        "<arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>"
    )
    return text


def _assistant_content_and_reasoning(
    message: Mapping[str, Any], _media_placeholder: Optional[Mapping[str, str]] = None
) -> tuple[str, str | None]:
    """Shared by GLM-5.2/5.3: ``_media_placeholder`` selects how media parts in
    assistant content render — as the given placeholder markers (GLM-5.3) when
    provided, or as ``<reminder>`` degradation text (GLM-5.2, the default)."""
    content = _visible_text(message.get("content", ""), _media_placeholder=_media_placeholder)
    reasoning_content = message.get("reasoning_content")
    # Reasoning is a structured field; think-tag spellings in content are always literal text.
    return content, reasoning_content if isinstance(reasoning_content, str) else None


def _tool_call_args(arguments: Mapping[str, Any], _literal_spans=None, _offset: int = 0) -> str:
    text = ""
    for key, value in arguments.items():
        text += "<arg_key>"
        text = _append_literal(text, str(key), _literal_spans, _offset)
        text += "</arg_key><arg_value>"
        value = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        text = _append_literal(text, value, _literal_spans, _offset)
        text += "</arg_value>"
    return text


def _render_tool_calls(tool_calls: List[Dict[str, Any]], _literal_spans=None, _offset: int = 0) -> str:
    text = ""
    for tool_call in tool_calls:
        tool_call = tool_call["function"] if "function" in tool_call else tool_call
        text += "<tool_call>"
        text = _append_literal(text, str(tool_call["name"]), _literal_spans, _offset)
        text += _tool_call_args(tool_call["arguments"], _literal_spans, _offset + len(text))
        text += "</tool_call>"
    return text


def _render_tool_result(
    content: Any, tools: Optional[List[Dict[str, Any]]], _literal_spans=None, _offset: int = 0
) -> str:
    text = "<tool_response>"
    if isinstance(content, list) and content and isinstance(content[0], Mapping):
        if content[0].get("type") == "tool_reference":
            text += "<tools>\n"
            for tool_ref in content:
                for tool in tools or []:
                    tool = tool["function"] if "function" in tool else tool
                    if tool.get("name") == tool_ref.get("name"):
                        text = _append_literal(text, _tool_to_json(tool), _literal_spans, _offset) + "\n"
            return text + "</tools></tool_response>"
        if "output" in content[0]:
            text = ""
            for tool_result in content:
                text += "<tool_response>"
                text = _append_literal(text, str(tool_result["output"]), _literal_spans, _offset)
                text += "</tool_response>"
            return text
    text += _visible_text(content, _literal_spans, _offset + len(text))
    return text + "</tool_response>"


def _assistant_generated_segments(
    message: Mapping[str, Any],
    message_index: int,
    last_user_index: int,
    clear_thinking: bool,
) -> list[str]:
    if not message.get("loss", True):
        return []

    content, reasoning_content = _assistant_content_and_reasoning(message)
    segments = []
    if reasoning_content is not None and (not clear_thinking or message_index > last_user_index):
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
    enable_thinking: bool = True,
    reasoning_effort: Literal["high", "max"] = "max",
    clear_thinking: bool = False,
    _literal_spans: list[tuple[int, int]] | None = None,
) -> tuple[str, list[bool]]:
    """渲染 GLM-5.2 对话；XTuner 默认保留多轮历史 reasoning 用于 SFT。"""
    text = ""
    loss_mask: list[bool] = []

    def append(value: str, loss: bool, *, literal: bool = False) -> None:
        nonlocal text
        text = _append_literal(text, value, _literal_spans) if literal else text + value
        loss_mask.extend([loss] * len(value))

    append("[gMASK]<sop>", False)
    if enable_thinking:
        effective_reasoning_effort = "high" if reasoning_effort == "high" else "max"
        append(f"<|system|>Reasoning Effort: {effective_reasoning_effort.capitalize()}", False)
    if tools:
        append(_render_tools(tools, _literal_spans, len(text)), False)

    last_user_index = -1
    for index, message in enumerate(messages):
        if message.get("role") == "user":
            last_user_index = index

    # user/observation 角色 token 同时是生成停止目标，其 loss 归属前一个 assistant。
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
                append("</think>", loss and enable_thinking)
            if content.strip():
                append(content.strip(), loss, literal=True)
            if message.get("tool_calls"):
                append(_render_tool_calls(message["tool_calls"], _literal_spans, len(text)), loss)
            previous_assistant_loss = loss
            next_role = messages[index + 1].get("role") if index + 1 < len(messages) else None
            if next_role is None and not add_generation_prompt:
                # 训练用 EOS 只追加在对话末尾，且仅当渲染的就是训练样本本身；官方模板从不在
                # 对话中间输出 EOS。add_generation_prompt=True 渲染的是推理前缀，末条即使是
                # assistant 也不能在 <|assistant|><think> 之前补 EOS，否则训练停止符落在
                # 推理前缀中间。
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
        append("<think>" if enable_thinking else "<think></think>", False)

    return text, loss_mask


@lru_cache(maxsize=8)
def _content_tokenizer(tokenizer):
    """Private encode-only backend; do not change the tokenizer used for decode
    or VL."""
    if not hasattr(tokenizer, "backend_tokenizer"):
        raise ValueError("GLM special-token literals require a fast tokenizer with a tokenizers backend.")
    backend = copy.deepcopy(tokenizer.backend_tokenizer)
    backend.add_special_tokens(list(tokenizer.added_tokens_decoder.values()))
    assert all(token.special for token in backend.get_added_tokens_decoder().values()), (
        "The content tokenizer must mark every added token as special before encoding literals."
    )
    backend.encode_special_tokens = True
    backend.no_padding()
    backend.no_truncation()
    return backend


def _encode_glm_text(tokenizer, text: str, literal_spans: list[tuple[int, int]] | None = None):
    try:
        encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        input_ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    except Exception:
        input_ids, offsets = get_offset_mapping(tokenizer, text)
    if not literal_spans:
        return input_ids, offsets

    added_ids = set(tokenizer.get_added_vocab().values())
    controls = []
    span_index = 0
    has_literal = False
    for token_id, (start, end) in zip(input_ids, offsets):
        if token_id not in added_ids or start == end:
            continue
        while span_index < len(literal_spans) and literal_spans[span_index][1] <= start:
            span_index += 1
        if span_index < len(literal_spans) and literal_spans[span_index][0] < end:
            has_literal = True
        else:
            controls.append((token_id, start, end))
    if not has_literal:
        return input_ids, offsets

    # Only genuine controls delimit BPE runs. Literal spellings stay with their context.
    backend = _content_tokenizer(tokenizer)
    input_ids, offsets = [], []
    cursor = 0
    for token_id, start, end in [*controls, (None, len(text), len(text))]:
        if cursor < start:
            content = backend.encode(text[cursor:start], add_special_tokens=False)
            input_ids.extend(content.ids)
            offsets.extend((cursor + left, cursor + right) for left, right in content.offsets)
        if token_id is not None:
            input_ids.append(token_id)
            offsets.append((start, end))
        cursor = end
    return input_ids, offsets


def _tokenize_with_loss_mask(
    tokenizer: PreTrainedTokenizer, text: str, loss_mask: list[bool], literal_spans=None
) -> tuple[list[int], list[int]]:
    input_ids, offset_mapping = _encode_glm_text(tokenizer, text, literal_spans)
    labels = []
    for token_id, (start, end) in zip(input_ids, offset_mapping):
        if start == end or not any(loss_mask[start:end]):
            labels.append(IGNORE_INDEX)
        else:
            labels.append(token_id)
    return input_ids, labels


def glm52_tokenize_fn_fastspeed(
    tokenizer: PreTrainedTokenizer,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    add_generation_prompt: bool = False,
    enable_thinking: bool = True,
    reasoning_effort: Literal["high", "max"] = "max",
    clear_thinking: bool = False,
    **kwargs,
) -> tuple[list[int], list[int]]:
    literal_spans: list[tuple[int, int]] = []
    text, loss_mask = render_glm52_chat(
        messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
        clear_thinking=clear_thinking,
        _literal_spans=literal_spans,
    )
    return _tokenize_with_loss_mask(tokenizer, text, loss_mask, literal_spans)


def glm52_tokenize_fn_slowspeed(
    tokenizer: PreTrainedTokenizer,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    add_generation_prompt: bool = False,
    enable_thinking: bool = True,
    reasoning_effort: Literal["high", "max"] = "max",
    clear_thinking: bool = False,
    **kwargs,
) -> tuple[list[int], list[int]]:
    """慢速 golden 参考实现：基于 token 级别前缀 diff 对齐 labels。

    渲染复用 GLM 多停止边界的 SFT 语义，标签仍通过独立的 token 前缀 diff 计算：
    1. 先渲染完整对话，得到唯一的 total_ids 参考序列。
    2. 对每条需要 loss 的 assistant 消息，渲染其历史前缀并加 generation prompt。
    3. 再渲染“历史 + 当前 assistant”，用 token 前缀差得到当前 assistant 应监督的 suffix。
    4. 从上次命中位置开始在 total_ids 里顺序查找 suffix，找到后复制到 labels。
    """
    # SFT 渲染保持官方轮间格式，并在没有后继角色边界时补标准 EOS。
    full_spans: list[tuple[int, int]] = []
    full_text, _ = render_glm52_chat(
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

    # 记录搜索起点，避免多轮对话里把后续 assistant 的 suffix 匹配到前面的同文案片段。
    curr_ptr = 0
    for index, message in enumerate(messages):
        if message.get("role") != "assistant" or not message.get("loss", True):
            continue

        # 历史前缀以 generation prompt 结束；prefix 之后的 token 才是当前 assistant 生成区间。
        prefix_spans: list[tuple[int, int]] = []
        prefix_text, _ = render_glm52_chat(
            messages[:index],
            tools=tools if index == 0 else None,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            clear_thinking=clear_thinking,
            _literal_spans=prefix_spans,
        )

        # 当前截断渲染可能和 full render 不同，例如历史 thinking 会被模板清掉；是否能在 full
        # render 中匹配上 suffix，正是 golden 语义的一部分。
        message_spans: list[tuple[int, int]] = []
        message_text, _ = render_glm52_chat(
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
            # 截断渲染以 endoftext 结尾；完整对话改用实际的下一角色停止 token。
            content_ids[-1] = tokenizer.convert_tokens_to_ids(_NEXT_ROLE_STOP_TOKENS[next_role])
        elif next_role is None and add_generation_prompt:
            # add_generation_prompt=True 时完整渲染以推理前缀结尾、不含训练 EOS；截断渲染补在
            # 末尾的 EOS 在 total_ids 里不存在，去掉再对齐。
            assert content_ids[-1] == tokenizer.convert_tokens_to_ids(_END_OF_TEXT)
            content_ids = content_ids[:-1]
            if not content_ids:
                continue

        # 在完整 token 序列中做 token 级绝对对齐，避免字符 offset 对特殊 token 的边界解释差异。
        for start in range(curr_ptr, len(total_ids) - len(content_ids) + 1):
            if total_ids[start : start + len(content_ids)] == content_ids:
                labels[start : start + len(content_ids)] = content_ids
                curr_ptr = start + len(content_ids)
                break
        else:
            if index == len(messages) - 1:
                raise ValueError("Could not align final assistant message in GLM-5.2 chat template rendering.")

    return total_ids, labels


class Glm52ChatMessages(BaseModel):
    model_config = ConfigDict(extra="forbid")
    messages: List[Dict[str, Any]]
    tools: Optional[List[Dict[str, Any]]] = None

    def tokenize(
        self,
        tokenizer: PreTrainedTokenizer,
        chat_template=None,
        add_generation_prompt: bool = False,
        enable_thinking: bool = True,
        reasoning_effort: Literal["high", "max"] = "max",
        clear_thinking: bool = False,
        **kwargs,
    ) -> Dict:
        # Keep default-system semantics aligned with the Qwen3.5 message path.
        if chat_template is not None and chat_template.default_system is not None:
            if self.messages[0]["role"] == "system":
                self.messages[0]["content"] = chat_template.default_system
            else:
                self.messages.insert(0, {"role": "system", "content": chat_template.default_system})

        input_ids, labels = glm52_tokenize_fn_fastspeed(
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
