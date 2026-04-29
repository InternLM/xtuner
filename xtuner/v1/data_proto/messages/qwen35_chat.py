# Copyright (c) OpenMMLab. All rights reserved.
import copy
import json
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from transformers import PreTrainedTokenizer
from xtuner.v1.data_proto.templates import HybridChatTemplate


def get_offset_mapping(tokenizer, text: str):
    encoding = tokenizer(text, add_special_tokens=False)
    input_ids = encoding["input_ids"]
    offset_mapping = []
    pos = 0
    pending_ids = []  # type: ignore
    max_pending = 8

    def _flush_pending(start, end):
        nonlocal pending_ids, pos
        offset_mapping.extend([(start, end)] * len(pending_ids))
        pos = end
        pending_ids = []

    def _flush_pending_as_empty():
        nonlocal pending_ids
        offset_mapping.extend([(pos, pos)] * len(pending_ids))
        pending_ids = []

    for token_id in input_ids:
        pending_ids.append(token_id)
        decoded = tokenizer.decode(pending_ids, skip_special_tokens=False)
        if not decoded:
            continue
        idx = text.find(decoded, pos)
        if idx != -1:
            end = idx + len(decoded)
            _flush_pending(idx, end)
        elif "\ufffd" not in decoded or len(pending_ids) >= max_pending:
            _flush_pending_as_empty()

    if pending_ids:
        _flush_pending_as_empty()
    return input_ids, offset_mapping


def render_content(content, do_vision_count, image_count, video_count, add_vision_id=False):
    if isinstance(content, str):
        return content, image_count, video_count
    result = ""
    for item in content:
        if "image" in item or "image_url" in item or item.get("type") == "image":
            if do_vision_count:
                image_count += 1
            if add_vision_id:
                result += f"Picture {image_count}: "
            result += "<|vision_start|><|image_pad|><|vision_end|>"
        elif "video" in item or item.get("type") == "video":
            if do_vision_count:
                video_count += 1
            if add_vision_id:
                result += f"Video {video_count}: "

            video_content = item.get("video", {})
            assert isinstance(video_content, dict), f"video_content must be a dict, but got {type(video_content)}"
            timestamps = video_content.get("timestamps", [])
            if len(timestamps) > 0:
                video_placeholder = ""
                for timestamp in timestamps:
                    video_placeholder += f"<{timestamp:.1f} seconds><|vision_start|><|video_pad|><|vision_end|>"
                result += video_placeholder
            else:
                # 每个视频可能有 n 帧，每一帧里面可能占据 m 个 token
                assert "num_frames" in video_content, "num_frames must be in video_content"
                num_frames = video_content["num_frames"]
                for _ in range(len(num_frames)):
                    result += "<|vision_start|><|video_pad|><|vision_end|>"
            conversation_timestamp = video_content.get("conversation_timestamps", [])
            if len(conversation_timestamp) > 0:
                start_time = conversation_timestamp[0]
                end_time = conversation_timestamp[1]
                timestamps = f"<{start_time:.1f}-{end_time:.1f} seconds>"
                result += timestamps

        elif "text" in item:
            result += item["text"]
    return result, image_count, video_count


# Qwen3.5 工具系统提示（与 Qwen3 不同的 XML 格式）
_QWEN35_TOOL_SYSTEM = "# Tools\n\nYou have access to the following functions:\n\n<tools>"
_QWEN35_TOOL_INSTRUCTIONS = (
    "\n</tools>\n\n"
    "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n"
    "<function=example_function_name>\n"
    "<parameter=example_parameter_1>\n"
    "value_1\n"
    "</parameter>\n"
    "<parameter=example_parameter_2>\n"
    "This is the value for the second parameter\n"
    "that can span\n"
    "multiple lines\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>\n\n"
    "<IMPORTANT>\n"
    "Reminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> "
    "block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language BEFORE "
    "the function call, but NOT after\n"
    "- If there is no function call available, answer the question like normal with your "
    "current knowledge and do not tell the user about function calls\n"
    "</IMPORTANT>"
)


def _render_tool_call_args(arguments: dict) -> str:
    """将 tool_call arguments dict 渲染为 Qwen3.5 XML 参数格式。"""
    parts = ""
    for k, v in arguments.items():
        parts += f"<parameter={k}>\n"
        if isinstance(v, (dict, list)):
            parts += json.dumps(v, ensure_ascii=False)
        else:
            parts += str(v)
        parts += "\n</parameter>\n"
    return parts


def qwen35_tokenize_fn_fastspeed(
    messages,
    tokenizer=None,
    tools=None,
    add_generation_prompt=False,
    add_vision_id=False,
    return_labels=True,
    enable_thinking=None,
):
    if enable_thinking is None:
        enable_thinking = any("reasoning_content" in msg for msg in messages)
    else:
        enable_thinking = enable_thinking

    image_count = 0
    video_count = 0
    result = ""
    loss_mask: list[bool] = []

    def _render(content, do_vision_count: bool) -> str:
        nonlocal image_count, video_count
        out, image_count, video_count = render_content(
            content, do_vision_count, image_count, video_count, add_vision_id
        )
        return out

    def _append(text: str, is_loss: bool) -> None:
        nonlocal result
        result += text
        loss_mask.extend([is_loss] * len(text))

    # ── system / tools 块 ─────────────────────────────────────────────────
    if tools:
        _append("<|im_start|>system\n", False)
        _append(_QWEN35_TOOL_SYSTEM, False)
        for tool in tools:
            _append("\n" + json.dumps(tool, ensure_ascii=False), False)
        _append(_QWEN35_TOOL_INSTRUCTIONS, False)
        if messages[0]["role"] == "system":
            sys_content = _render(messages[0]["content"], False).strip()
            if sys_content:
                _append("\n\n" + sys_content, False)
        _append("<|im_end|>\n", False)
    else:
        if messages[0]["role"] == "system":
            sys_content = _render(messages[0]["content"], False).strip()
            _append(f"<|im_start|>system\n{sys_content}<|im_end|>\n", False)

    # ── 计算 last_query_index ─────────────────────────────────────────────
    multi_step_tool = True
    last_query_index = len(messages) - 1
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if multi_step_tool and msg["role"] == "user":
            content_str = _render(msg["content"], False).strip()
            if not (content_str.startswith("<tool_response>") and content_str.endswith("</tool_response>")):
                multi_step_tool = False
                last_query_index = i

    # ── 主循环 ────────────────────────────────────────────────────────────
    for idx, message in enumerate(messages):
        is_first = idx == 0
        is_last = idx == len(messages) - 1
        content = _render(message["content"], True).strip()
        role = message["role"]

        if role == "user" or (role == "system" and not is_first):
            _append(f"<|im_start|>{role}\n{content}<|im_end|>\n", False)

        elif role == "assistant":
            reasoning_content = ""
            if isinstance(message.get("reasoning_content"), str):
                reasoning_content = message["reasoning_content"]
            else:
                if "</think>" in content:
                    reasoning_content = content.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                    content = content.split("</think>")[-1].lstrip("\n")
            # Qwen3.5 模板对 reasoning_content 做 |trim
            reasoning_content = reasoning_content.strip()

            is_loss = message.get("loss", True)

            _append(f"<|im_start|>{role}\n", False)

            if idx > last_query_index:
                # 最后查询之后的轮次：渲染 <think> 块，并计算 loss
                _append("<think>\n", False)
                if reasoning_content:
                    # 有 reasoning：gen prompt 以 <think>\n 结尾，content_tokens 从 reasoning 开始
                    _append(reasoning_content + "\n", is_loss)
                    _append("</think>\n\n", is_loss)
                elif enable_thinking:
                    # enable_thinking=True 但无 reasoning：gen prompt 以 <think>\n 结尾
                    # content_tokens 从 </think> 开始，所以 </think>\n\n 算 loss
                    _append("\n", False)  # 空内容的 \n（与 <think>\n 合并为 \n\n token，不算 loss）
                    _append("</think>\n\n", is_loss)
                else:
                    # enable_thinking=False：gen prompt 以完整 <think>\n\n</think>\n\n 结尾
                    # content_tokens 只包含实际回复，</think>\n\n 不算 loss
                    _append("\n", False)
                    _append("</think>\n\n", False)
                body_is_loss = is_loss
            else:
                # 历史轮次：
                # - enable_thinking=False：gen prompt 含完整 <think>\n\n</think>\n\n，
                #   content_tokens 只有回复内容，在 total_ids 中可以找到 → 用 is_loss
                # - enable_thinking=True：content_tokens 以 </think> 开头，
                #   total_ids 里历史轮无 <think> 块 → NOT FOUND → 不算 loss
                body_is_loss = is_loss if not enable_thinking else False
                _append(content, body_is_loss)

            if idx > last_query_index:
                _append(content, body_is_loss)

            # tool_calls（Qwen3.5 XML 格式）
            if message.get("tool_calls"):
                for tc_idx, tool_call in enumerate(message["tool_calls"]):
                    tc = tool_call.get("function", tool_call)
                    tc_name = tc["name"]
                    tc_args = tc.get("arguments", {})

                    if tc_idx == 0:
                        if content.strip():
                            _append("\n\n", body_is_loss)
                        _append(f"<tool_call>\n<function={tc_name}>\n", body_is_loss)
                    else:
                        _append(f"\n<tool_call>\n<function={tc_name}>\n", body_is_loss)

                    if isinstance(tc_args, dict):
                        _append(_render_tool_call_args(tc_args), body_is_loss)
                    _append("</function>\n</tool_call>", body_is_loss)

            _append("<|im_end|>\n", body_is_loss)

        elif role == "tool":
            prev_role = messages[idx - 1]["role"] if idx > 0 else None
            if is_first or prev_role != "tool":
                _append("<|im_start|>user", False)
            _append("\n<tool_response>\n", False)
            _append(content, False)
            _append("\n</tool_response>", False)
            next_role = messages[idx + 1]["role"] if not is_last else None
            if is_last or next_role != "tool":
                _append("<|im_end|>\n", False)

    if add_generation_prompt:
        _append("<|im_start|>assistant\n", False)
        if not enable_thinking:
            _append("<think>\n\n</think>\n\n", False)
        else:
            _append("<think>\n", False)

    # ── 不需要 labels ─────────────────────────────────────────────────────
    if not return_labels:
        return result, loss_mask

    # ── 需要 labels ───────────────────────────────────────────────────────
    assert tokenizer is not None, "return_labels=True 时必须传入 tokenizer"

    try:
        encoded = tokenizer(
            result,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"]
        offset_mapping = encoded["offset_mapping"]
    except Exception:
        input_ids, offset_mapping = get_offset_mapping(tokenizer, result)

    labels = []
    for token_id, (start, end) in zip(input_ids, offset_mapping):
        if start == end:
            labels.append(-100)
        elif any(loss_mask[i] for i in range(start, end)):
            labels.append(token_id)
        else:
            labels.append(-100)

    return input_ids, labels


def qwen35_tokenize_fn_slowspeed(tokenizer, messages: List[Dict[str, str]], tools=None, add_vision_id=False, **kwargs):
    """
    终极稳定版 Tokenize：基于 Token 级别的绝对对齐 (椒盐算法升级版)。
    逻辑：
    1. 生成全量 total_ids 作为唯一真实的参考系。
    2. 对于每个 assistant 消息，通过历史截断渲染，提取出它“应该长什么样”的 token 序列。
    3. 在 total_ids 中顺藤摸瓜，精确匹配这些 token 序列。
    4. 完美解决字符偏移错位、模板历史修改、以及特殊 Token 对齐问题。
    """

    enable_thinking = any("reasoning_content" in msg for msg in messages)

    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, tools=tools, add_vision_id=add_vision_id, enable_thinking=enable_thinking, **kwargs
    )
    total_ids = tokenizer.encode(full_text, add_special_tokens=False)
    labels = [-100] * len(total_ids)
    # 记录在 total_ids 中搜索的起始位置，确保不会搜到前面的轮次
    curr_ptr = 0
    for i, msg in enumerate(messages):
        if msg["role"] == "assistant" and msg.get("loss", True):
            # 1. 获取包含当前消息之前所有内容的“前缀”文本 (带 generation prompt)
            prompt_text = tokenizer.apply_chat_template(
                messages[:i],
                tokenize=False,
                add_generation_prompt=True,
                add_vision_id=add_vision_id,
                enable_thinking=enable_thinking,
                tools=tools if i == 0 else None,
                **kwargs,
            )
            # 2. 获取包含当前消息的完整“截断”文本
            # 我们通过修改当前消息的内容，强制在末尾加上一个罕见标记，来准确捕获这部分的内容
            # 为什么要加标记？因为我们想知道当前消息的结束符（如 <|im_end|>）被 tokenizer 编成了什么
            temp_msgs = [m.copy() for m in messages[: i + 1]]
            # 提取真实内容
            m_text = tokenizer.apply_chat_template(
                temp_msgs,
                tokenize=False,
                add_vision_id=add_vision_id,
                enable_thinking=enable_thinking,
                tools=tools if i == 0 else None,
                **kwargs,
            )
            # 转换为 Token 序列
            p_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            m_ids = tokenizer.encode(m_text, add_special_tokens=False)
            # 3. 提取当前消息的纯内容 Tokens (包含 reasoning, content, tool_calls, 以及结尾的 im_end)
            # 注意：由于 tokenizer 的特性，m_ids 的前缀可能并不完美等于 p_ids
            # 所以我们要寻找 p_ids 的特征来切分
            # 为了最稳健，我们直接在 m_ids 的末尾倒推。
            # 我们知道 m_ids 是由 p_ids + current_content_ids 组成的
            # 我们直接取差集：
            content_tokens = m_ids[len(p_ids) :]
            if not content_tokens:
                continue
            # 4. 在全量 total_ids 中搜索这段 content_tokens
            found = False
            # 从 curr_ptr 开始往后搜
            for s_ptr in range(curr_ptr, len(total_ids) - len(content_tokens) + 1):
                if total_ids[s_ptr : s_ptr + len(content_tokens)] == content_tokens:
                    # 匹配成功！
                    labels[s_ptr : s_ptr + len(content_tokens)] = content_tokens
                    curr_ptr = s_ptr + len(content_tokens)
                    found = True
                    break
            if not found:
                # 如果没找到，说明模板在全量渲染时，修改了这条历史消息的内容（例如删了 thinking）
                # 这是允许的，只要它不是当前轮次（我们不强求历史轮次一定要匹配上，因为我们通常只对最后的 Turn 算 loss）
                # 但如果是最后一条消息还没匹配上，那就一定是出大问题了
                if i == len(messages) - 1:
                    raise ValueError("严重错误：最后一条 Assistant 消息无法在全量 Token 中对齐。")
    return total_ids, labels


# 我们采用全新逻辑，因此不需要继承 BaseChatMessages，后续之前的 ChatMessages 逻辑全部删除
class Qwen35ChatMessages(BaseModel):
    model_config = ConfigDict(extra="forbid")
    messages: List[dict]  # 暂时不做校验
    tools: Optional[List[Dict]] = None

    def tokenize(
        self,
        tokenizer: PreTrainedTokenizer,
        chat_template: HybridChatTemplate,
        add_vision_id=False,
        add_generation_prompt=False,
        enable_thinking=None,
        **kwargs,
    ) -> Dict:
        is_pretrain = False
        if len(self.messages) == 1 and self.messages[0]["role"] == "pretrain":
            is_pretrain = True

        if is_pretrain:
            text, _, _ = render_content(
                self.messages[0]["content"],
                do_vision_count=True,
                image_count=0,
                video_count=0,
                add_vision_id=add_vision_id,
            )
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            label_ids = copy.deepcopy(token_ids)
        else:
            # replace system message
            if chat_template.default_system is not None:
                if self.messages[0]["role"] == "system":
                    self.messages[0]["content"] = chat_template.default_system
                else:
                    self.messages.insert(0, {"role": "system", "content": chat_template.default_system})

            token_ids, label_ids = qwen35_tokenize_fn_fastspeed(
                self.messages,
                tokenizer,
                self.tools,
                add_vision_id=add_vision_id,
                return_labels=True,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking,
            )
        return {"input_ids": token_ids, "labels": label_ids}
