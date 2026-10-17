"""GLM-5.3-Flash 的 chat template 与 loss mask，见设计文档 F1。

TestGlm53Rendering
    test_reasoning_effort_domain_matches_jinja  reasoning_effort 取值域与 jinja 一致
    test_enable_thinking_false_rejected         enable_thinking=False 被显式拒绝（jinja 忽略该参数）
    test_all_generation_eos_are_supervised_at_assistant_boundaries  停止 token 计入 loss
    test_media_placeholder_matches_jinja        媒体占位符与 jinja 一致
TestGlm53OfficialTemplateContract
    test_render_matches_official_jinja          飞书失败报告样例逐字符对齐官方 jinja + fast/slow 一致
    test_no_eos_and_no_loss_at_mid_conversation_boundaries  对话中间无 EOS、边界 token 无 loss
"""

import os

import pytest

from transformers import AutoTokenizer
from xtuner.v1.data_proto.messages import Glm53ChatMessages
from xtuner.v1.data_proto.messages.glm53_chat import glm53_tokenize_fn_slowspeed, render_glm53_chat


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

# 飞书《glm5.3 flash训推chat template一致性》失败报告（2026-10-15）的渲染层样例。
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
    # tool/assistant/system 角色的媒体渲染为三段式标记（渲染层；数据层能否加载另说）。
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
}

GLM_5_3_FLASH_PATH = os.environ["GLM_5_3_FLASH_PATH"]
GLM53_TEMPLATE_DEFAULTS = {"enable_thinking": True, "reasoning_effort": "max", "clear_thinking": False}


@pytest.fixture(scope="module")
def tokenizer():
    if not os.path.isdir(GLM_5_3_FLASH_PATH):
        pytest.fail(f"GLM_5_3_FLASH_PATH not found: {GLM_5_3_FLASH_PATH}")
    return AutoTokenizer.from_pretrained(GLM_5_3_FLASH_PATH, trust_remote_code=True)


def _render_from_hf(tokenizer, messages, **kwargs):
    return tokenizer.apply_chat_template(messages, tokenize=False, **{**GLM53_TEMPLATE_DEFAULTS, **kwargs})


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


class TestGlm53OfficialTemplateContract:
    @pytest.mark.parametrize("case_name", list(_CONTRACT_CASES))
    def test_render_matches_official_jinja(self, tokenizer, case_name):
        case = _CONTRACT_CASES[case_name]
        messages, tools = case["messages"], case.get("tools")

        text, _ = render_glm53_chat(messages, tools=tools, **GLM53_TEMPLATE_DEFAULTS)
        hf_rendered = _render_from_hf(tokenizer, messages, tools=tools)
        assert text == hf_rendered + tokenizer.eos_token

        tokenized = Glm53ChatMessages(messages=messages, tools=tools).tokenize(tokenizer)
        slow_ids, slow_labels = glm53_tokenize_fn_slowspeed(tokenizer, messages, tools=tools)
        assert tokenized["input_ids"] == slow_ids
        assert tokenized["labels"] == slow_labels

    def test_no_eos_and_no_loss_at_mid_conversation_boundaries(self, tokenizer):
        # 官方模板从不在对话中间输出 EOS；assistant→system/assistant 的边界 token
        # （<|system|>/<|assistant|>）不是生成停止目标，不计入 loss。
        for case_name in ("assistant_then_system", "consecutive_assistants"):
            messages = _CONTRACT_CASES[case_name]["messages"]
            tokenized = Glm53ChatMessages(messages=messages).tokenize(tokenizer)
            eos_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
            assert tokenized["input_ids"].count(eos_id) == 1
            assert tokenized["input_ids"][-1] == eos_id
            assert tokenized["labels"][-1] == eos_id

            boundary_token = "<|system|>" if case_name == "assistant_then_system" else "<|assistant|>"
            boundary_id = tokenizer.convert_tokens_to_ids(boundary_token)
            boundary_index = tokenized["input_ids"].index(boundary_id)
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
