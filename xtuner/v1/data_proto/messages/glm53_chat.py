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
   The placeholders are emitted for media in ANY role's content (user/system/assistant/tool),
   exactly like the official ``visible_text`` macro; whether the training pipeline can load a
   given role's media is the data layer's concern (see ``collect_image_video_paths_and_extra``,
   enabled for every role via ``Glm53VLTokenizeFunction.allow_all_roles``).
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
   to markers positionally (``glm53_vl_tokenize_fn._expand_marker`` pairs the i-th payload with
   the i-th marker), so the VLM tokenize fn collects payloads from :func:`glm53_media_messages`,
   which flattens messages with :func:`sort_tool_messages` internally, making the payload
   collection order follow the rendered marker order.
5. Structured ``{"output": ...}`` entries render their output through ``visible_text`` and a
   nested ``tool_reference`` list expands the matching tool definitions into
   ``<tool_response><tools>...</tools></tool_response>`` (GLM-5.2's official template has
   neither behavior, so these live here rather than in :mod:`glm52_chat`).
6. ``tool_calls`` whose ``arguments`` are an unparsed JSON string are rejected with a
   ``ValueError`` explaining the fix, instead of crashing with an ``AttributeError`` deep in
   glm52's renderer. ``ValueError`` belongs to the data pipeline's dirty-data error set, so
   such samples are discarded through the dirty-sample channel instead of killing the cache
   worker.
7. ``<|endoftext|>`` is appended only after the FINAL assistant message of a training render
   (``add_generation_prompt=False``), as the training stop suffix. The official template never
   emits EOS mid-conversation, so assistant→system and assistant→assistant boundaries get no
   extra token; only ``<|user|>``/``<|observation|>`` act as supervised stop targets. With
   ``add_generation_prompt=True`` the render is an inference prefix, so even a trailing
   assistant gets no EOS -- emitting one would leave the training stop token in the middle of
   the inference prefix.
8. ``clear_thinking=True`` masks the WHOLE historical assistant turn (both ``reasoning_content``
   and ``content``) before the last user message, not just turns that carry a reasoning block;
   a reasoning-less historical assistant is still historical output, and supervising it would
   contradict the flag's purpose.
9. Tool-message shapes are classified by probing only ``content[0]`` (the official
   ``is_list_of_outputs`` and ``tool_reference`` checks do the same), so an aggregated list
   whose LATER entry lacks an ``output`` key raises ``KeyError`` and the sample is discarded
   through the dirty-data channel. The official template silently renders an empty
   ``<tool_response></tool_response>`` for such entries; the discard is deliberate -- silently
   empty supervision is harder to detect than a dropped sample.
"""

from collections.abc import Mapping
from typing import Any, Dict, List, Literal, Optional, TypedDict, TypeGuard, Union, cast

from pydantic import BaseModel, ConfigDict

from transformers import PreTrainedTokenizer
from xtuner.v1.utils import IGNORE_INDEX

from .glm52_chat import (
    _NEXT_ROLE_STOP_TOKENS,
    _append_literal,
    _assistant_content_and_reasoning,
    _encode_glm_text,
    _render_tools,
    _tokenize_with_loss_mask,
    _tool_to_json,
)
from .glm52_chat import (
    _render_tool_calls as _render_tool_calls_glm52,
)
from .glm52_chat import (
    _visible_text as _visible_text_glm52,
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


# tool 消息 content 的形态谱系，对应官方 render_tool_response 宏的四个分支
# （chat_template.jinja:78-94，排序/拉平/渲染三处的分支注释共用这套编号；判定均按
# 首元素探测，与官方一致）：
#   A. 纯字符串
#   B. tool_reference 列表（整体），渲染为工具 schema、无媒体
#   C. 聚合 entry 列表（官方 is_list_of_outputs）：entry.output 再分
#      C1. tool_reference 列表（渲染为 schema、无媒体）/ C2. 其余 output（else 兜底，经 visible_text）
#   D. 其余 content（else 兜底，经 visible_text；part 列表里的媒体发 marker，非标量 str() 化）
# 运行时判定便宜的写成 TypeGuard；形状已由 guard 确定的位置直接 cast，不重复运行时校验。


class Glm53TextPart(TypedDict):
    """``{"type": "text", "text": ...}`` content part（分支 C2/D 的文本项）。"""

    type: Literal["text"]
    text: str


class Glm53MediaPart(TypedDict, total=False):
    """Image/video content part（分支 C2/D 的媒体项）；payload 键名与 ``type`` 一致。"""

    type: Literal["image", "image_url", "video", "video_url"]
    image: Dict[str, Any]
    image_url: Dict[str, Any]
    video: Dict[str, Any]
    video_url: Dict[str, Any]


class Glm53ToolReferencePart(TypedDict):
    """``{"type": "tool_reference", "name": ...}``（分支 B/C1）：展开同名工具的 schema。"""

    type: Literal["tool_reference"]
    name: str


# 分支 C2/D 经 visible_text 渲染的 part 列表项：text / 媒体 part，或混排的纯字符串。
Glm53VisiblePart = Union[Glm53TextPart, Glm53MediaPart, str]

# 分支 C 单条 entry 的 ``output``：C1 为 tool_reference 列表；C2 为字符串或 part 列表。
Glm53ToolOutput = Union[str, List[Glm53ToolReferencePart], List[Glm53VisiblePart]]


class Glm53ToolOutputEntry(TypedDict, total=False):
    """聚合结果条目 ``{"id": ..., "output": ...}``（分支 C）。

    ``output`` 为 ``str`` 或 part 列表（C2），或 ``List[Glm53ToolReferencePart]``（C1）；
    id 可在 ``id`` 或 ``tool_call_id`` 键上（官方 id_of 宏）。
    """

    id: str
    tool_call_id: str
    output: Glm53ToolOutput


def _is_media_part(part: Any) -> TypeGuard[Glm53MediaPart]:
    return isinstance(part, Mapping) and part.get("type") in ("image", "image_url", "video", "video_url")


def _is_tool_reference_list(value: Any) -> TypeGuard[List[Glm53ToolReferencePart]]:
    """分支 B/C1：首项为 tool_reference 的非空列表（官方同样只探首项）。"""
    return (
        isinstance(value, list)
        and len(value) > 0
        and isinstance(value[0], Mapping)
        and value[0].get("type") == "tool_reference"
    )


def _is_output_entry_list(content: Any) -> TypeGuard[List[Glm53ToolOutputEntry]]:
    """分支 C：首元素带 ``output`` 键的非空列表（官方 is_list_of_outputs 宏）。"""
    return (
        isinstance(content, list) and len(content) > 0 and isinstance(content[0], Mapping) and "output" in content[0]
    )


def _visible_text(content: Any, _literal_spans=None, _offset: int = 0) -> str:
    return _visible_text_glm52(content, _literal_spans, _offset, _media_placeholder=_MEDIA_PLACEHOLDER)


def _id_of(obj: Mapping[str, Any]) -> Any:
    return obj.get("tool_call_id") or obj.get("id")


def _render_tool_calls(tool_calls: List[Dict[str, Any]], _literal_spans=None, _offset: int = 0) -> str:
    """Validate ``tool_calls`` then delegate to glm52's renderer.

    String ``arguments`` (unparsed JSON) are rejected with a repair hint instead of crashing
    with an ``AttributeError`` inside glm52's ``_tool_call_args``; the ``ValueError`` falls in
    the VL pipeline's dirty-data error set, so the sample is discarded rather than killing the
    cache worker.
    """
    for tool_call in tool_calls:
        function = tool_call["function"] if "function" in tool_call else tool_call
        arguments = function["arguments"]
        if not isinstance(arguments, Mapping):
            raise ValueError(
                f"GLM-5.3-Flash tool {function['name']!r} arguments must be an object; "
                f"got {type(arguments).__name__}. Parse JSON strings before tokenization."
            )
    return _render_tool_calls_glm52(tool_calls, _literal_spans, _offset)


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

    数据级别示例（排序发生在第 4 级 entry，而非第 2 级 message；形态编号与文件
    顶部类型谱系一致，覆盖官方 render_tool_response 宏的全部分支）::

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
            # 形态 D: content 是 part 列表;分散形态,这条 message 整体是一个排序单位(id 在 message 上)
            {"role": "tool", "tool_call_id": "c", "content": [
                {"type": "text", "text": "RESULT_C"},           # ← 级别 5: content part
            ]},
            # 形态 A: content 是纯字符串,同样整条 message 是一个排序单位
            # {"role": "tool", "tool_call_id": "x", "content": "RESULT_X"},
            # 形态 B: content 整体是 tool_reference 列表(渲染为工具 schema、无媒体),
            # 仍按 message 上的 id 参与排序
            # {"role": "tool", "tool_call_id": "y",
            #  "content": [{"type": "tool_reference", "name": "a"}]},
            # ↓ 形态 C: 聚合形态,content 是 entry 数组,内含 2 个 entry,顺序 b, a(与调用序相反)
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
                # 形态 C1: entry.output 是 tool_reference 列表(渲染为 schema、无媒体),
                # 仍按 entry 的 id 参与排序;上面的 b/a 两个 entry 即形态 C2(字符串或 part 列表)
                # {"id": "z", "output": [{"type": "tool_reference", "name": "b"}]},
            ]},
            {"role": "assistant", "content": "ASSISTANT_ANSWER"},
        ]

        # E. can_sort 成立时的拉平结果: [entry a 的 message, entry b 的 message, entry c 的 message]
        #    (顺序 == tool_calls)
        # F. can_sort 任一条件不满足(无前置 assistant tool_calls / 结果 id 缺失或块内重复 /
        #    结果 id 不在 tool_calls / tool_call id 缺失或重复)时,块保持消息原序,不拉平。
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
                    # 形态 C: 聚合消息拆成单 entry 浅拷贝(其他键保留)。
                    entries = cast(List[Glm53ToolOutputEntry], message["content"])
                    units.extend({**message, "content": [entry]} for entry in entries)
                else:
                    # 形态 A/B/D: 整条 message 是一个排序单位。
                    units.append(message)
            out.extend(sorted(units, key=lambda unit: position[_result_ids_of(unit)[0]]))
        index = end + 1
    return out


def glm53_media_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Lift every media part into flat one-source messages in rendered marker
    order, for media payload collection.

    The renderer emits a two-stage placeholder marker for media parts anywhere in the conversation --
    including nested inside structured tool outputs (``{"id": ..., "output": [...]}`` entries) -- while
    ``collect_image_video_paths_and_extra`` only sees the direct content parts of each message. Feeding it
    this flattened view keeps payloads positionally paired with the rendered markers: tool blocks follow
    the same can_sort ordering the template renders with (``sort_tool_messages``, idempotent on
    pre-sorted input), each aggregated entry's ``output`` is lifted in entry order, and tool_reference
    outputs (which render as tool schemas, not media) are skipped.

    Each synthesized message carries the REAL role of its source message (``user`` for user content,
    ``tool`` for tool results whether plain or aggregated): the role on this view is provenance, not a
    filter. ``GLM53VLTokenizeFunction`` collects with ``allow_all_roles=True``, so role gating stays
    entirely in this flatten step, which lifts exactly the media the renderer emits markers for.

    数据级别示例（形态编号同 sort_tool_messages 的示例）::

        messages = [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "U1"}}]},
            {"role": "assistant", "content": "", "tool_calls": [调用a, 调用b]},  # 顺序: a, b
            {"role": "tool", "content": [                    # 形态 C: 聚合 entry 列表
                {"id": "b", "output": [                      # C2: part 列表,含媒体
                    {"type": "text", "text": "RESULT_B"},
                    {"type": "image_url", "image_url": {"url": "IMG_B"}},
                ]},
                {"id": "a", "output": [                      # C1: tool_reference 列表
                    {"type": "tool_reference", "name": "a"},
                ]},
            ]},
            # 形态 B(content 整体是 tool_reference 列表)与形态 A(纯字符串)同样无媒体,
            # 不产生合成消息。
        ]

        # 拉平结果(entry 按 tool_calls 序 a→b 排列;role 保留来源消息的真实角色):
        # [
        #     {"role": "user", "content": [U1 的 image part]},
        #     {"role": "tool", "content": [IMG_B 的 image part]},  # 来源是 entry b 所在的 tool 消息
        #     # entry a 是 C1(tool_reference),渲染为工具 schema、无媒体,被跳过
        # ]
    """
    media_messages: List[Dict[str, Any]] = []
    for message in sort_tool_messages(messages):
        content = message.get("content", "")
        if message.get("role") == "tool" and _is_output_entry_list(content):
            # 形态 C: 聚合 entry 列表,逐 entry 取 output(后续 entry 缺 output 键时抛
            # KeyError 走脏数据丢弃,见模块 docstring 第 8 条)。
            outputs: List[Any] = [entry["output"] for entry in content]
        else:
            # 形态 A/B/D: 整条消息的 content 即一个结果的 output。
            outputs = [content]
        for output in outputs:
            if not isinstance(output, list):
                # 形态 A / C2 的字符串 output: 无媒体。
                continue
            if _is_tool_reference_list(output):
                # 形态 B / C1: 渲染为工具 schema,不产生媒体 marker,跳过。
                continue
            # 形态 C2/D 的 part 列表: 按序收集媒体 part,与渲染 marker 位置式配对。
            media = [part for part in output if _is_media_part(part)]
            if media:
                media_messages.append({"role": message["role"], "content": media})
    return media_messages


def _render_tool_reference_list(
    refs: List[Glm53ToolReferencePart], tools: Optional[List[Dict[str, Any]]], _literal_spans=None, _offset: int = 0
) -> str:
    """分支 B/C1 共用：展开 ``tool_reference`` 列表匹配的工具定义。

    对应官方 tool_references_to_response 宏（jinja:18-31）。
    """
    text = "<tool_response><tools>\n"
    for tool_ref in refs:
        for tool in tools or []:
            tool = tool["function"] if "function" in tool else tool
            if tool.get("name") == tool_ref.get("name"):
                text = _append_literal(text, _tool_to_json(tool), _literal_spans, _offset) + "\n"
    return text + "</tools></tool_response>"


def _render_tool_output(
    output: Any, tools: Optional[List[Dict[str, Any]]], _literal_spans=None, _offset: int = 0
) -> str:
    """Render one aggregated ``{output: ...}`` entry the way the official template's
    render_tool_response does: a tool_reference list expands to the referenced tool
    schemas, anything else renders through ``_visible_text`` (media parts become the
    two-stage placeholder markers)."""
    if _is_tool_reference_list(output):
        # 分支 C1: output 是 tool_reference 列表,展开匹配的工具定义。
        return _render_tool_reference_list(output, tools, _literal_spans, _offset)
    # 分支 C2: 字符串或 part 列表,经 visible_text 渲染(媒体发两段式 marker)。
    prefix = "<tool_response>"
    return prefix + _visible_text(output, _literal_spans, _offset + len(prefix)) + "</tool_response>"


def _render_tool_result(
    content: Any, tools: Optional[List[Dict[str, Any]]], _literal_spans=None, _offset: int = 0
) -> str:
    """GLM-5.3 version of the tool-result renderer.

    GLM-5.2 has no multimodal input, so its renderer emits media as a ``<reminder>`` text; the official GLM-5.3
    template renders media parts in tool content as the two-stage placeholder markers instead, so this version goes
    through GLM-5.3's ``_visible_text``.
    """
    if isinstance(content, str):
        # 分支 A: 纯字符串结果。
        return _append_literal("<tool_response>", content, _literal_spans, _offset) + "</tool_response>"
    if _is_tool_reference_list(content):
        # 分支 B: content 整体是 tool_reference 列表,展开匹配的工具定义。
        return _render_tool_reference_list(content, tools, _literal_spans, _offset)
    if _is_output_entry_list(content):
        # 分支 C: 聚合 entry 列表,逐 entry 渲染(官方只探 content[0];后续 entry 缺
        # output 键时抛 KeyError 走脏数据丢弃,见模块 docstring 第 8 条)。
        text = ""
        for entry in content:
            text += _render_tool_output(entry["output"], tools, _literal_spans, _offset + len(text))
        return text
    # 分支 D: part 列表或其他标量,经 visible_text 渲染。
    prefix = "<tool_response>"
    return prefix + _visible_text(content, _literal_spans, _offset + len(prefix)) + "</tool_response>"


def _render_tool_block(
    messages: List[Dict[str, Any]],
    start: int,
    tools: Optional[List[Dict[str, Any]]],
    _literal_spans=None,
    _offset: int = 0,
) -> str:
    """Render the contiguous tool-message block starting at ``start``.

    Results are emitted in the preceding assistant's ``tool_calls`` order when the official
    template's can_sort rule holds (see ``_sortable_call_ids``); any violation keeps the
    message order. Media blocks sort as well, matching the official template and the
    inference stack: media payloads are collected from :func:`glm53_media_messages`, which
    applies the same ``sort_tool_messages`` ordering, so payloads collected in message order
    follow the sorted marker order.
    """
    end = start
    while end + 1 < len(messages) and messages[end + 1].get("role") == "tool":
        end += 1
    block = messages[start : end + 1]

    call_ids = _sortable_call_ids(messages, start, end)
    if call_ids is None:
        text = ""
        for message in block:
            text += _render_tool_result(message.get("content", ""), tools, _literal_spans, _offset + len(text))
        return text

    text = ""
    for call_id in call_ids:
        for message in block:
            if _is_list_of_outputs(message):
                # 形态 C: 聚合消息,抽出 id 匹配的 entry 渲染。
                entries = cast(List[Glm53ToolOutputEntry], message["content"])
                for entry in entries:
                    if _id_of(entry) == call_id:
                        text += _render_tool_output(entry["output"], tools, _literal_spans, _offset + len(text))
            elif _id_of(message) == call_id:
                text += _render_tool_result(message.get("content", ""), tools, _literal_spans, _offset + len(text))
    return text


def _assistant_plan(
    message: Mapping[str, Any],
    index: int,
    last_user_index: int,
    clear_thinking: bool,
) -> tuple[str, str | None, bool, bool]:
    """单条 assistant 的判定式（监督与否、是否发射 reasoning），render 与 slowspeed 的唯一事实来源。

    返回 ``(content, reasoning_content, supervised, render_reasoning)``。
    """
    content, reasoning_content = _assistant_content_and_reasoning(message, _media_placeholder=_MEDIA_PLACEHOLDER)
    # slowspeed 的生成段仍逐段手写复刻 render 的发射逻辑——交叉校验的价值正在于生成段独立；
    # 但判定式若两处各写一份，render 判错时 slowspeed 会以同样方式判错，fast == slow 对照失效
    # 且无测试报警，所以判定式只允许存在这一份。clear_thinking 屏蔽"最后一个 user 之前"的整轮
    # 历史 assistant（reasoning 与 content 一起），而不是只屏蔽带 reasoning 的轮次——无 reasoning
    # 的历史 assistant 同样是历史输出，留下监督会让 clear_thinking 语义自相矛盾。
    after_last_user = index > last_user_index
    supervised = bool(message.get("loss", True)) and (not clear_thinking or after_last_user)
    render_reasoning = reasoning_content is not None and (not clear_thinking or after_last_user)
    return content, reasoning_content, supervised, render_reasoning


def render_glm53_chat(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    add_generation_prompt: bool = False,
    enable_thinking: bool = True,
    reasoning_effort: Literal["low", "high", "max"] = "max",
    clear_thinking: bool = False,
    _literal_spans: list[tuple[int, int]] | None = None,
) -> tuple[str, list[bool]]:
    """渲染 GLM-5.3-Flash 对话；结构与 GLM-5.2 相同，媒体渲染为两段式 placeholder 标记。

    ``clear_thinking=True`` 时，最后一个 user 之前的整轮历史 assistant（reasoning 与 content
    一起）不进 loss；``add_generation_prompt=True`` 时不为末尾 assistant 补训练 EOS。
    """
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
            content, reasoning_content, loss, render_reasoning = _assistant_plan(
                message, index, last_user_index, clear_thinking
            )

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
                append(_render_tool_block(messages, index, tools, _literal_spans, len(text)), False)
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
    """慢速 golden 参考实现：与 ``glm53_tokenize_fn_fastspeed`` 相同的渲染，但逐条 assistant
    独立重算标签，用精确前缀对齐校验而不是滑动窗口搜索。

    每条受监督的 assistant：把 ``messages[:index]`` 渲染成推理前缀（以 ``<|assistant|><think>``
    结尾），再逐段复刻渲染器为该 assistant 发射的文本（reasoning/``</think>``/content/
    tool_calls）及其停止目标（user/tool 边界的交接 token，或对话末尾的训练 EOS）。前缀与生成段
    拼接编码后必须与全文编码结果逐 token 一致，任何错位直接抛 ``ValueError``——旧实现的滑动
    窗口会在错误位置静默标注或丢掉监督（assistant→system、连续 assistant 边界因此与 fast
    不一致）。
    """
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

    # clear_thinking 的屏蔽口径与 render 层一致：最后一个 user 之前的 assistant 整轮跳过。
    last_user_index = max(
        (index for index, message in enumerate(messages) if message.get("role") == "user"), default=-1
    )
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        # 判定式（监督与否、是否发射 reasoning）与 render 共用同一份实现，见 _assistant_plan。
        content, reasoning_content, supervised, render_reasoning = _assistant_plan(
            message, index, last_user_index, clear_thinking
        )
        if not supervised:
            continue

        prefix_spans: list[tuple[int, int]] = []
        prefix_text, _ = render_glm53_chat(
            messages[:index],
            tools=tools,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
            clear_thinking=clear_thinking,
            _literal_spans=prefix_spans,
        )

        generated = ""
        generated_spans: list[tuple[int, int]] = []

        def gen_append(value: str, *, literal: bool = False) -> None:
            nonlocal generated
            if literal:
                generated_spans.append((len(generated), len(generated) + len(value)))
            generated += value

        # ---- 生成段保持手写复刻（交叉校验的价值在这里，不与 render 共用）----
        if render_reasoning:
            assert reasoning_content is not None
            gen_append(reasoning_content, literal=True)
            gen_append("</think>")
        else:
            gen_append("</think>")
        if content.strip():
            gen_append(content.strip(), literal=True)
        if message.get("tool_calls"):
            gen_append(_render_tool_calls(message["tool_calls"], generated_spans, len(generated)))
        next_role = messages[index + 1].get("role") if index + 1 < len(messages) else None
        if next_role in _NEXT_ROLE_STOP_TOKENS:
            gen_append(_NEXT_ROLE_STOP_TOKENS[next_role])
        elif next_role is None and not add_generation_prompt:
            gen_append(_END_OF_TEXT)

        prefix_ids, _ = _encode_glm_text(tokenizer, prefix_text, prefix_spans)
        combined_spans = [
            *prefix_spans,
            *[(start + len(prefix_text), end + len(prefix_text)) for start, end in generated_spans],
        ]
        combined_ids, _ = _encode_glm_text(tokenizer, prefix_text + generated, combined_spans)
        start = len(prefix_ids)
        content_ids = combined_ids[start:]
        if not content_ids:
            continue
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
        enable_thinking: bool = True,
        reasoning_effort: Literal["low", "high", "max"] = "max",
        clear_thinking: bool = False,
        **kwargs,
    ) -> Dict:
        # 训练侧暂不感知 clear_thinking=True：历史轮数据应由用户侧拆成多条样本（每条只有结尾
        # 轮算 loss）。实现（render/fast/slow）与测试保留，开放时只需删掉这句断言；VL 侧在
        # Glm53VLTokenizeFunction 构造期有同样的守卫。
        assert clear_thinking is False, (
            "Glm53ChatMessages only supports clear_thinking=False; prepare clear-thinking data "
            "by splitting turns into separate samples instead."
        )
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
