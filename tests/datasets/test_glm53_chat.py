"""GLM-5.3-Flash 的 chat template 与 loss mask，见设计文档 F1。

TestGlm53Rendering
    test_reasoning_effort_domain_matches_jinja  reasoning_effort 取值域与 jinja 一致
    test_enable_thinking_false_rejected         enable_thinking=False 被显式拒绝（jinja 忽略该参数）
    test_all_generation_eos_are_supervised_at_assistant_boundaries  停止 token 计入 loss
    test_media_placeholder_matches_jinja        媒体占位符与 jinja 一致
    test_add_generation_prompt_after_trailing_assistant_emits_no_eos  推理前缀不为末尾 assistant 补训练 EOS
    test_clear_thinking_masks_whole_historical_assistant_turns  clear_thinking 屏蔽整轮历史 assistant（含无 reasoning 的轮次）
TestGlm53OfficialTemplateContract
    test_render_matches_official_jinja          逐字符对齐官方 jinja
    test_can_sort_fallback_keeps_message_order  排序前提被破坏时回退为消息顺序
    test_mid_conversation_boundaries_use_their_actual_occurrence  对话中间无 EOS、边界 token loss 正确（含 fast/slow 一致）
    test_tool_results_are_never_supervised      tool 结果不计入 loss
TestGlm53MediaMessages
    媒体拉平视图 glm53_media_messages：真实 role、tool_reference 形态（B/C1）跳过
"""

import os

import pytest

from transformers import AutoTokenizer
from xtuner.v1.data_proto.messages import Glm53ChatMessages
from xtuner.v1.data_proto.messages.glm53_chat import (
    glm53_media_messages,
    glm53_tokenize_fn_fastspeed,
    glm53_tokenize_fn_slowspeed,
    render_glm53_chat,
    sort_tool_messages,
)


GLM_5_3_FLASH_PATH = os.environ["GLM_5_3_FLASH_PATH"]
GLM53_TEMPLATE_DEFAULTS = {"enable_thinking": True, "reasoning_effort": "max", "clear_thinking": False}

_TOOLS = [
    {"type": "function", "function": {"name": "a", "parameters": {"type": "object"}}},
    {"type": "function", "function": {"name": "b", "parameters": {"type": "object"}}},
]
_TOOL_CALLS = [
    {"function": {"arguments": {}, "name": "a"}, "id": "a", "type": "function"},
    {"function": {"arguments": {}, "name": "b"}, "id": "b", "type": "function"},
]
_IMAGE_PART = {"type": "image", "image": {"url": "unused.jpg", "image_wh": [112, 112]}}

_CONTRACT_CASES = {
    # content 列表混排纯字符串与 text 字典（jinja 均按文本渲染）。
    "text_content_parts": {
        "messages": [
            {"role": "user", "content": ["prefix", {"type": "text", "text": "正文"}, "suffix"]},
            {"role": "assistant", "content": "ASSISTANT_ANSWER"},
        ]
    },
    # assistant→system：官方模板不在中间插 EOS。
    "assistant_then_system": {
        "messages": [
            {"role": "user", "content": "USER_INPUT"},
            {"role": "assistant", "content": "first"},
            {"role": "system", "content": "New policy"},
            {"role": "assistant", "content": "FINAL_ANSWER"},
        ]
    },
    # 连续 assistant：同样不插 EOS。
    "consecutive_assistants": {
        "messages": [
            {"role": "user", "content": "USER_INPUT"},
            {"role": "assistant", "content": "first"},
            {"role": "assistant", "content": "second"},
        ]
    },
    # 分散 tool 结果按 tool_calls 顺序重排。
    "tool_results_reversed": {
        "messages": [
            {"role": "user", "content": "USER_INPUT"},
            {"role": "assistant", "content": "", "tool_calls": _TOOL_CALLS},
            {"role": "tool", "content": "B", "tool_call_id": "b"},
            {"role": "tool", "content": "A", "tool_call_id": "a"},
            {"role": "assistant", "content": "ASSISTANT_ANSWER"},
        ],
        "tools": _TOOLS,
    },
    # 单条 tool 消息聚合多个 output 时同样按调用序重排。
    "tool_aggregate_reversed": {
        "messages": [
            {"role": "assistant", "content": "", "tool_calls": _TOOL_CALLS},
            {"role": "tool", "content": [{"id": "b", "output": "B"}, {"id": "a", "output": "A"}]},
            {"role": "assistant", "content": "ASSISTANT_ANSWER"},
        ],
        "tools": _TOOLS,
    },
    # 结构化 output 渲染可见文本，而不是 Python 字符串化。
    "tool_structured_text": {
        "messages": [
            {"role": "tool", "content": [{"output": [{"type": "text", "text": "done"}]}]},
            {"role": "assistant", "content": "ASSISTANT_ANSWER"},
        ]
    },
    # output 内嵌套的 tool_reference 展开匹配的工具定义。
    "tool_nested_reference": {
        "messages": [
            {"role": "tool", "content": [{"output": [{"type": "tool_reference", "name": "a"}]}]},
            {"role": "assistant", "content": "ASSISTANT_ANSWER"},
        ],
        "tools": _TOOLS,
    },
    # content 整体是 tool_reference 列表：官方 render_tool_response 的独立分支
    # （jinja:81-82，优先于 is_list_of_outputs 判定），展开匹配的工具定义。
    "tool_reference_content": {
        "messages": [
            {"role": "tool", "content": [{"type": "tool_reference", "name": "a"}], "tool_call_id": "a"},
            {"role": "assistant", "content": "ASSISTANT_ANSWER"},
        ],
        "tools": _TOOLS,
    },
    # tool 角色的顶层/嵌套媒体渲染为三段式标记（渲染层；数据层能否加载另说）。
    "tool_image_probe": {
        "messages": [
            {"role": "user", "content": "USER_INPUT"},
            {"role": "assistant", "content": "", "tool_calls": [_TOOL_CALLS[0]]},
            {"role": "tool", "content": [_IMAGE_PART], "tool_call_id": "a"},
            {"role": "assistant", "content": "ASSISTANT_ANSWER"},
        ],
        "tools": _TOOLS,
    },
    "tool_nested_image_probe": {
        "messages": [
            {"role": "tool", "content": [{"output": [_IMAGE_PART]}]},
            {"role": "assistant", "content": "ASSISTANT_ANSWER"},
        ]
    },
    "assistant_image_probe": {
        "messages": [
            {"role": "user", "content": "USER_INPUT"},
            {"role": "assistant", "content": [_IMAGE_PART]},
        ]
    },
    "system_image_probe": {
        "messages": [
            {"role": "system", "content": [_IMAGE_PART]},
            {"role": "user", "content": "USER_INPUT"},
            {"role": "assistant", "content": "ASSISTANT_ANSWER"},
        ]
    },
    # 音频占位符在模板层有定义（运行时支持范围不在此断言）。
    "audio_placeholder_probe": {
        "messages": [
            {"role": "user", "content": [{"type": "audio"}]},
            {"role": "assistant", "content": "ASSISTANT_ANSWER"},
        ]
    },
}

# can_sort 前提被破坏时，官方模板回退为消息顺序；这些样例与上面的排序样例互为对照。
_FALLBACK_CASES = {
    "dup_result_id": {
        "messages": [
            {"role": "assistant", "content": "", "tool_calls": _TOOL_CALLS},
            {"role": "tool", "content": "B", "tool_call_id": "a"},
            {"role": "tool", "content": "A", "tool_call_id": "a"},
            {"role": "assistant", "content": "done"},
        ],
        "tools": _TOOLS,
    },
    "unmatched_result_id": {
        "messages": [
            {"role": "assistant", "content": "", "tool_calls": _TOOL_CALLS},
            {"role": "tool", "content": "B", "tool_call_id": "b"},
            {"role": "tool", "content": "X", "tool_call_id": "zzz"},
            {"role": "assistant", "content": "done"},
        ],
        "tools": _TOOLS,
    },
    "no_preceding_tool_calls": {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "tool", "content": "B", "tool_call_id": "b"},
            {"role": "tool", "content": "A", "tool_call_id": "a"},
            {"role": "assistant", "content": "done"},
        ],
        "tools": _TOOLS,
    },
    # 结果消息缺 id（jinja:190 的 not eid 分支）。
    "missing_result_id": {
        "messages": [
            {"role": "assistant", "content": "", "tool_calls": _TOOL_CALLS},
            {"role": "tool", "content": "B"},
            {"role": "tool", "content": "A", "tool_call_id": "a"},
            {"role": "assistant", "content": "done"},
        ],
        "tools": _TOOLS,
    },
    # tool_call 缺 id（jinja:211 的 not tc_id 分支）。
    "missing_tool_call_id": {
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"arguments": {}, "name": "a"}, "type": "function"}],
            },
            {"role": "tool", "content": "A", "tool_call_id": "a"},
            {"role": "assistant", "content": "done"},
        ],
        "tools": _TOOLS,
    },
    # tool_call id 重复（jinja:214-218 的重复检测分支）。
    "dup_tool_call_id": {
        "messages": [
            {"role": "assistant", "content": "", "tool_calls": [_TOOL_CALLS[0], _TOOL_CALLS[0]]},
            {"role": "tool", "content": "A", "tool_call_id": "a"},
            {"role": "assistant", "content": "done"},
        ],
        "tools": _TOOLS,
    },
}

# 携带媒体的 tool 块同样按 tool_calls 顺序重排，与官方 jinja 渲染对齐（jinja 的发射单位是
# entry：can_sort 时把匹配 entry 从所在消息中抽出按序输出）。HF 处理器的媒体收集按原始消息
# 顺序、位置式配对，与 jinja 排序并不自洽，训练侧由 sort_tool_messages 在收集前按同一规则
# 拉平消息，保证 payload 收集顺序与渲染 marker 顺序一致。
# id 均满足 can_sort，纯文本对照见 _CONTRACT_CASES 的 tool_results_reversed / tool_aggregate_reversed。
_MEDIA_BLOCK_CASES = {
    "tool_results_media_reversed": {
        "messages": [
            {"role": "assistant", "content": "", "tool_calls": _TOOL_CALLS},
            {
                "role": "tool",
                "tool_call_id": "b",
                "content": [
                    {"type": "text", "text": "RESULT_B"},
                    {"type": "image_url", "image_url": {"url": "IMG_B"}},
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "a",
                "content": [
                    {"type": "text", "text": "RESULT_A"},
                    {"type": "image_url", "image_url": {"url": "IMG_A"}},
                ],
            },
            {"role": "assistant", "content": "ASSISTANT_ANSWER"},
        ],
        "tools": _TOOLS,
    },
    "tool_aggregate_media_reversed": {
        "messages": [
            {"role": "assistant", "content": "", "tool_calls": _TOOL_CALLS},
            {
                "role": "tool",
                "content": [
                    {
                        "id": "b",
                        "output": [
                            {"type": "text", "text": "RESULT_B"},
                            {"type": "image_url", "image_url": {"url": "IMG_B"}},
                        ],
                    },
                    {
                        "id": "a",
                        "output": [
                            {"type": "text", "text": "RESULT_A"},
                            {"type": "image_url", "image_url": {"url": "IMG_A"}},
                        ],
                    },
                ],
            },
            {"role": "assistant", "content": "ASSISTANT_ANSWER"},
        ],
        "tools": _TOOLS,
    },
}


@pytest.fixture(scope="module")
def tokenizer():
    if not os.path.isdir(GLM_5_3_FLASH_PATH):
        pytest.fail(f"GLM_5_3_FLASH_PATH not found: {GLM_5_3_FLASH_PATH}")
    return AutoTokenizer.from_pretrained(GLM_5_3_FLASH_PATH, trust_remote_code=True)


def _render_from_hf(tokenizer, messages, **kwargs):
    return tokenizer.apply_chat_template(messages, tokenize=False, **{**GLM53_TEMPLATE_DEFAULTS, **kwargs})


def _token_index_after_context(tokenizer, rendered, context, token):
    context_end = rendered.index(context) + len(context)
    token_start = rendered.index(token, context_end)
    return len(tokenizer.encode(rendered[:token_start], add_special_tokens=False))


class TestGlm53Rendering:
    @pytest.mark.parametrize("reasoning_effort", ["low", "high", "max", "bogus"])
    def test_reasoning_effort_domain_matches_jinja(self, tokenizer, reasoning_effort):
        # jinja: reasoning_effort in ['low', 'high'] else 'max' -- "bogus" exercises the fallback.
        messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        tokenized = Glm53ChatMessages(messages=messages).tokenize(tokenizer, reasoning_effort=reasoning_effort)
        hf_rendered = _render_from_hf(tokenizer, messages, reasoning_effort=reasoning_effort)
        rendered = tokenizer.decode(tokenized["input_ids"], skip_special_tokens=False)
        assert rendered == hf_rendered + tokenizer.eos_token
        slow_ids, slow_labels = glm53_tokenize_fn_slowspeed(tokenizer, messages, reasoning_effort=reasoning_effort)
        assert tokenized["input_ids"] == slow_ids
        assert tokenized["labels"] == slow_labels

    def test_enable_thinking_false_rejected(self, tokenizer):
        # 官方 jinja 完全忽略 enable_thinking（Reasoning Effort 头与未闭合 <think> 永远输出），
        # False 会被静默忽略，因此 xtuner 侧显式拒绝而不是假装支持。
        messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        with pytest.raises(AssertionError, match="enable_thinking"):
            Glm53ChatMessages(messages=messages).tokenize(tokenizer, enable_thinking=False)
        with pytest.raises(AssertionError, match="enable_thinking"):
            glm53_tokenize_fn_slowspeed(tokenizer, messages, enable_thinking=False)

    def test_string_tool_call_arguments_rejected(self):
        # arguments 必须是对象；未解析的 JSON 字符串抛带修复提示的 ValueError（落入数据管线
        # 脏样本错误集合，走丢弃通道），而不是 glm52 渲染器深处的 AttributeError 裸崩。
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "a", "arguments": '{"city": "beijing"}'}, "id": "a", "type": "function"}
                ],
            },
        ]
        with pytest.raises(ValueError, match="Parse JSON strings before tokenization"):
            render_glm53_chat(messages, **GLM53_TEMPLATE_DEFAULTS)

    def test_all_generation_eos_are_supervised_at_assistant_boundaries(self, tokenizer):
        # 助手轮边界上的停止 token 必须计入 loss，否则模型学不会停。
        messages = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Final question"},
            {"role": "assistant", "content": "Final answer"},
        ]
        tokenized = Glm53ChatMessages(messages=messages).tokenize(tokenizer)
        slow_ids, slow_labels = glm53_tokenize_fn_slowspeed(tokenizer, messages)
        rendered = tokenizer.decode(tokenized["input_ids"], skip_special_tokens=False)
        hf_rendered = _render_from_hf(tokenizer, messages)

        assert rendered == hf_rendered + tokenizer.eos_token
        assert tokenized["input_ids"] == slow_ids
        assert tokenized["labels"] == slow_labels
        user_id = tokenizer.convert_tokens_to_ids("<|user|>")
        first_user_label = tokenized["labels"][tokenized["input_ids"].index(user_id)]
        assert first_user_label == -100  # first <|user|> has no preceding assistant turn
        assert tokenized["labels"][-1] == tokenizer.convert_tokens_to_ids("<|endoftext|>")

    def test_media_placeholder_matches_jinja(self, tokenizer):
        # 媒体占位符必须与官方 jinja 的两段式标记一致。
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "video"},
                    {"type": "audio"},
                    {"type": "text", "text": "describe these"},
                ],
            },
            {"role": "assistant", "content": "ok"},
        ]
        tokenized = Glm53ChatMessages(messages=messages).tokenize(tokenizer)
        rendered = tokenizer.decode(tokenized["input_ids"], skip_special_tokens=False)
        hf_rendered = _render_from_hf(tokenizer, messages)

        assert rendered == hf_rendered + tokenizer.eos_token
        assert "<|begin_of_image|><|image|><|end_of_image|>" in rendered
        assert "<|begin_of_video|><|video|><|end_of_video|>" in rendered
        assert "<|begin_of_audio|><|end_of_audio|>" in rendered

    def test_add_generation_prompt_after_trailing_assistant_emits_no_eos(self, tokenizer):
        # add_generation_prompt=True 渲染的是推理前缀：末条为 assistant 时，旧实现在其内容后
        # 补训练 EOS，再把 <|assistant|><think> 追加到 EOS 之后，训练停止符落在推理前缀中间。
        messages = [
            {"role": "user", "content": "USER_INPUT"},
            {"role": "assistant", "content": "ANS"},
        ]
        text, _ = render_glm53_chat(messages, add_generation_prompt=True)
        assert "<|endoftext|>" not in text
        assert text.endswith("<|assistant|><think>")

        fast_ids, fast_labels = glm53_tokenize_fn_fastspeed(tokenizer, messages, add_generation_prompt=True)
        slow_ids, slow_labels = glm53_tokenize_fn_slowspeed(tokenizer, messages, add_generation_prompt=True)
        assert fast_ids == slow_ids
        assert fast_labels == slow_labels

    def test_clear_thinking_masks_whole_historical_assistant_turns(self, tokenizer):
        # clear_thinking=True 屏蔽最后一个 user 之前的整轮历史 assistant——无论该轮有没有
        # reasoning（旧实现只屏蔽带 reasoning 的轮次，无 reasoning 的历史轮照样监督，语义
        # 自相矛盾）；最后一个 user 之后的轮次照常监督。训练入口 Glm53ChatMessages.tokenize
        # 暂不开放该参数（数据应由用户侧拆分），这里直接测 tokenize fn 层的实现。
        messages = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "HISTORICAL_ANSWER"},
            {"role": "user", "content": "Second question"},
            {"role": "assistant", "content": "final answer", "reasoning_content": "actual reasoning"},
        ]
        fast_ids, fast_labels = glm53_tokenize_fn_fastspeed(tokenizer, messages, clear_thinking=True)
        slow_ids, slow_labels = glm53_tokenize_fn_slowspeed(tokenizer, messages, clear_thinking=True)
        assert fast_ids == slow_ids
        assert fast_labels == slow_labels

        user_id = tokenizer.convert_tokens_to_ids("<|user|>")
        last_user = max(index for index, token in enumerate(fast_ids) if token == user_id)
        assert all(label == -100 for index, label in enumerate(fast_labels) if index <= last_user)
        assert any(label != -100 for label in fast_labels[last_user:])

        kept_ids, kept_labels = glm53_tokenize_fn_fastspeed(tokenizer, messages, clear_thinking=False)
        assert (fast_ids, fast_labels) != (kept_ids, kept_labels)
        assert kept_ids == slow_ids  # clear_thinking 只动 loss，不动输入序列
        assert any(label != -100 for label in kept_labels[:last_user])

        # 训练入口守卫：clear_thinking=True 直接报 AssertionError，而不是静默生效。
        with pytest.raises(AssertionError, match="clear_thinking"):
            Glm53ChatMessages(messages=messages).tokenize(tokenizer, clear_thinking=True)


class TestGlm53OfficialTemplateContract:
    @pytest.mark.parametrize("case_name", list(_CONTRACT_CASES))
    def test_render_matches_official_jinja(self, tokenizer, case_name):
        case = _CONTRACT_CASES[case_name]
        messages, tools = case["messages"], case.get("tools")

        text, _ = render_glm53_chat(messages, tools=tools, **GLM53_TEMPLATE_DEFAULTS)
        hf_rendered = _render_from_hf(tokenizer, messages, tools=tools)
        assert text == hf_rendered + tokenizer.eos_token

        tokenized = Glm53ChatMessages(messages=messages, tools=tools).tokenize(tokenizer)
        assert tokenizer.decode(tokenized["input_ids"], skip_special_tokens=False) == hf_rendered + tokenizer.eos_token

    @pytest.mark.parametrize(
        ("messages", "boundary_token", "boundary_is_supervised"),
        [
            (
                [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "BOUNDARY_SENTINEL"},
                    {"role": "system", "content": "New policy"},
                    {"role": "assistant", "content": "Final answer"},
                ],
                "<|system|>",
                False,
            ),
            (
                [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "BOUNDARY_SENTINEL"},
                    {"role": "assistant", "content": "Second assistant turn"},
                ],
                "<|assistant|>",
                False,
            ),
            (
                [
                    {"role": "user", "content": "First question"},
                    {"role": "assistant", "content": "BOUNDARY_SENTINEL"},
                    {"role": "user", "content": "Second question"},
                    {"role": "assistant", "content": "Final answer"},
                ],
                "<|user|>",
                True,
            ),
            (
                [
                    {"role": "user", "content": "Question"},
                    {
                        "role": "assistant",
                        "content": "BOUNDARY_SENTINEL",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": {}},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_1", "content": "result"},
                    {"role": "assistant", "content": "Final answer"},
                ],
                "<|observation|>",
                True,
            ),
        ],
    )
    def test_mid_conversation_boundaries_use_their_actual_occurrence(
        self, tokenizer, messages, boundary_token, boundary_is_supervised
    ):
        official_text = _render_from_hf(tokenizer, messages)
        rendered, _ = render_glm53_chat(messages, **GLM53_TEMPLATE_DEFAULTS)
        assert rendered == official_text + tokenizer.eos_token
        assert tokenizer.eos_token not in official_text

        tokenized = Glm53ChatMessages(messages=messages).tokenize(tokenizer)
        eos_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        assert tokenized["input_ids"].count(eos_id) == 1
        assert tokenized["input_ids"][-1] == eos_id
        assert tokenized["labels"][-1] == eos_id

        # slowspeed 与 fast 在同一停止目标口径下逐 token 一致（assistant→system、连续
        # assistant 边界曾因滑动窗口错位静默丢监督）。
        slow_ids, slow_labels = glm53_tokenize_fn_slowspeed(tokenizer, messages)
        assert tokenized["input_ids"] == slow_ids
        assert tokenized["labels"] == slow_labels

        boundary_index = _token_index_after_context(tokenizer, official_text, "BOUNDARY_SENTINEL", boundary_token)
        boundary_id = tokenizer.convert_tokens_to_ids(boundary_token)
        assert tokenized["input_ids"][boundary_index] == boundary_id
        if boundary_is_supervised:
            assert tokenized["labels"][boundary_index] == boundary_id
        else:
            assert tokenized["labels"][boundary_index] == -100

    @pytest.mark.parametrize("case_name", list(_FALLBACK_CASES))
    def test_can_sort_fallback_keeps_message_order(self, tokenizer, case_name):
        # 结果 id 缺失/重复/不匹配或没有前序 tool_calls 时，官方模板不排序，保持消息顺序。
        case = _FALLBACK_CASES[case_name]
        messages, tools = case["messages"], case.get("tools")

        text, _ = render_glm53_chat(messages, tools=tools, **GLM53_TEMPLATE_DEFAULTS)
        hf_rendered = _render_from_hf(tokenizer, messages, tools=tools)
        assert text == hf_rendered + tokenizer.eos_token

        tokenized = Glm53ChatMessages(messages=messages, tools=tools).tokenize(tokenizer)
        slow_ids, slow_labels = glm53_tokenize_fn_slowspeed(tokenizer, messages, tools=tools)
        assert tokenized["input_ids"] == slow_ids
        assert tokenized["labels"] == slow_labels

    @pytest.mark.parametrize("case_name", list(_MEDIA_BLOCK_CASES))
    def test_media_block_sorts_like_official_jinja(self, tokenizer, case_name):
        # 携带媒体的 tool 块同样按 tool_calls 顺序重排，渲染与官方 jinja 逐字符一致；payload
        # 收集顺序由 sort_tool_messages 按同一规则拉平消息保证（见
        # test_sort_tool_messages_reorders_media_block /
        # test_sort_tool_messages_flattens_aggregated_entries 与 tokenize fn 测试）。
        case = _MEDIA_BLOCK_CASES[case_name]
        messages, tools = case["messages"], case.get("tools")

        text, _ = render_glm53_chat(messages, tools=tools, **GLM53_TEMPLATE_DEFAULTS)
        hf_rendered = _render_from_hf(tokenizer, messages, tools=tools)
        assert text == hf_rendered + tokenizer.eos_token
        assert text.index("RESULT_A") < text.index("RESULT_B")

    def test_sort_tool_messages_reorders_media_block(self):
        # 预排把乱序媒体块排成 tool_calls 顺序：媒体收集按消息顺序进行，随之与渲染后的
        # marker 顺序一致，payload 不错配；渲染器对预排序结果再排是 no-op（输出相同）。
        case = _MEDIA_BLOCK_CASES["tool_results_media_reversed"]
        sorted_messages = sort_tool_messages(case["messages"])
        assert [m["tool_call_id"] for m in sorted_messages if m["role"] == "tool"] == ["a", "b"]

        text, _ = render_glm53_chat(sorted_messages, tools=case["tools"], **GLM53_TEMPLATE_DEFAULTS)
        original, _ = render_glm53_chat(case["messages"], tools=case["tools"], **GLM53_TEMPLATE_DEFAULTS)
        assert text == original

    def test_sort_tool_messages_flattens_aggregated_entries(self):
        # 官方 jinja 的发射单位是 entry 而非消息：聚合消息内的 entry 会被抽出按 tool_calls
        # 顺序输出。预排按同一单位拉平——聚合消息拆成单 entry 浅拷贝（其余键保留），否则
        # 收集顺序（消息内原序）会与渲染 marker 顺序（按 call_id 重排）错配。
        case = _MEDIA_BLOCK_CASES["tool_aggregate_media_reversed"]
        sorted_messages = sort_tool_messages(case["messages"])
        tool_entries = [m["content"][0] for m in sorted_messages if m["role"] == "tool"]
        assert [entry["id"] for entry in tool_entries] == ["a", "b"]
        assert all(len(m["content"]) == 1 for m in sorted_messages if m["role"] == "tool")

        text, _ = render_glm53_chat(sorted_messages, tools=case["tools"], **GLM53_TEMPLATE_DEFAULTS)
        original, _ = render_glm53_chat(case["messages"], tools=case["tools"], **GLM53_TEMPLATE_DEFAULTS)
        assert text == original
        # 幂等：对拉平结果再排是 no-op。
        assert sort_tool_messages(sorted_messages) == sorted_messages

    @pytest.mark.parametrize("case_name", list(_FALLBACK_CASES))
    def test_sort_tool_messages_keeps_unsortable_blocks(self, case_name):
        # can_sort 前提被破坏时不重排，消息保持原顺序。
        case = _FALLBACK_CASES[case_name]
        assert sort_tool_messages(case["messages"]) == case["messages"]

    def test_tool_results_are_never_supervised(self, tokenizer):
        for case_name in ("tool_results_reversed", "tool_aggregate_reversed"):
            case = _CONTRACT_CASES[case_name]
            tokenized = Glm53ChatMessages(messages=case["messages"], tools=case["tools"]).tokenize(tokenizer)
            response_id = tokenizer.convert_tokens_to_ids("<|observation|>")
            start = tokenized["input_ids"].index(response_id)
            end = tokenized["input_ids"].index(tokenizer.convert_tokens_to_ids("<|assistant|>"), start)
            # start 处的 <|observation|> 是前一条 assistant 的停止目标（计入其 loss）；
            # 其后的 tool 结果内容完全不监督。
            assert all(label == -100 for label in tokenized["labels"][start + 1 : end])


class TestGlm53MediaMessages:
    def test_real_roles_and_entry_order(self):
        # 合成消息保留来源消息的真实 role；聚合 entry 按 tool_calls 序拉平，C1 被跳过。
        user_image = {"type": "image_url", "image_url": {"url": "u1.jpg"}}
        tool_image = {"type": "image_url", "image_url": {"url": "b.jpg"}}
        messages = [
            {"role": "user", "content": [user_image]},
            {"role": "assistant", "content": "", "tool_calls": _TOOL_CALLS},
            {
                "role": "tool",
                "content": [
                    {"id": "b", "output": [{"type": "text", "text": "B"}, tool_image]},
                    {"id": "a", "output": [{"type": "tool_reference", "name": "a"}]},
                ],
            },
        ]
        assert glm53_media_messages(messages) == [
            {"role": "user", "content": [user_image]},
            {"role": "tool", "content": [tool_image]},
        ]

    def test_tool_reference_content_skipped(self):
        # 形态 B：content 整体是 tool_reference 列表，渲染为工具 schema、无媒体，不产生合成消息。
        messages = [
            {"role": "assistant", "content": "", "tool_calls": [_TOOL_CALLS[0]]},
            {"role": "tool", "tool_call_id": "a", "content": [{"type": "tool_reference", "name": "a"}]},
        ]
        assert glm53_media_messages(messages) == []

    def test_plain_tool_message_keeps_tool_role(self):
        # 形态 D：分散形态 tool 消息的媒体同样保留 "tool" role。
        image = {"type": "image_url", "image_url": {"url": "t.jpg"}}
        messages = [
            {"role": "assistant", "content": "", "tool_calls": [_TOOL_CALLS[0]]},
            {"role": "tool", "tool_call_id": "a", "content": [{"type": "text", "text": "R"}, image]},
        ]
        assert glm53_media_messages(messages) == [{"role": "tool", "content": [image]}]
