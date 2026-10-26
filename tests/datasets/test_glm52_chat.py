"""GLM-5.2 的 chat template 与官方 jinja 的逐字符契约，对齐 #2198 的 GLM-5.3-Flash 修复。

TestGlm52OfficialTemplateContract
    test_render_matches_official_jinja          渲染逐字符对齐官方 jinja + 仅末尾 EOS
    test_mid_conversation_boundaries_use_their_actual_occurrence  对话中间无 EOS、边界 token loss 正确
    test_add_generation_prompt_after_trailing_assistant_emits_no_eos  推理前缀末尾 assistant 不补训练 EOS
"""

import os

import pytest

from transformers import AutoTokenizer
from xtuner.v1.data_proto.messages import Glm52ChatMessages
from xtuner.v1.data_proto.messages.glm52_chat import (
    glm52_tokenize_fn_fastspeed,
    glm52_tokenize_fn_slowspeed,
    render_glm52_chat,
)


GLM5_2_TINY_MOE_PATH = os.environ["GLM5_2_TINY_MOE_PATH"]
GLM52_TEMPLATE_DEFAULTS = {"enable_thinking": True, "reasoning_effort": "max", "clear_thinking": False}

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
}


@pytest.fixture(scope="module")
def tokenizer():
    if not os.path.isdir(GLM5_2_TINY_MOE_PATH):
        pytest.fail(f"GLM5_2_TINY_MOE_PATH not found: {GLM5_2_TINY_MOE_PATH}")
    return AutoTokenizer.from_pretrained(GLM5_2_TINY_MOE_PATH, trust_remote_code=True)


def _render_from_hf(tokenizer, messages, **kwargs):
    return tokenizer.apply_chat_template(messages, tokenize=False, **{**GLM52_TEMPLATE_DEFAULTS, **kwargs})


def _token_index_after_context(tokenizer, rendered, context, token):
    context_end = rendered.index(context) + len(context)
    token_start = rendered.index(token, context_end)
    return len(tokenizer.encode(rendered[:token_start], add_special_tokens=False))


class TestGlm52OfficialTemplateContract:
    @pytest.mark.parametrize("case_name", list(_CONTRACT_CASES))
    def test_render_matches_official_jinja(self, tokenizer, case_name):
        case = _CONTRACT_CASES[case_name]
        messages, tools = case["messages"], case.get("tools")

        text, _ = render_glm52_chat(messages, tools=tools, **GLM52_TEMPLATE_DEFAULTS)
        hf_rendered = _render_from_hf(tokenizer, messages, tools=tools)
        assert text == hf_rendered + tokenizer.eos_token

        tokenized = Glm52ChatMessages(messages=messages, tools=tools).tokenize(tokenizer)
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
        rendered, _ = render_glm52_chat(messages, **GLM52_TEMPLATE_DEFAULTS)
        assert rendered == official_text + tokenizer.eos_token
        assert tokenizer.eos_token not in official_text

        tokenized = Glm52ChatMessages(messages=messages).tokenize(tokenizer)
        eos_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        assert tokenized["input_ids"].count(eos_id) == 1
        assert tokenized["input_ids"][-1] == eos_id
        assert tokenized["labels"][-1] == eos_id

        boundary_index = _token_index_after_context(tokenizer, official_text, "BOUNDARY_SENTINEL", boundary_token)
        boundary_id = tokenizer.convert_tokens_to_ids(boundary_token)
        assert tokenized["input_ids"][boundary_index] == boundary_id
        if boundary_is_supervised:
            assert tokenized["labels"][boundary_index] == boundary_id
        else:
            assert tokenized["labels"][boundary_index] == -100

    def test_add_generation_prompt_after_trailing_assistant_emits_no_eos(self, tokenizer):
        # add_generation_prompt=True 渲染的是推理前缀：末条为 assistant 时，旧实现在其内容后
        # 补训练 EOS，再把 <|assistant|><think> 追加到 EOS 之后，训练停止符落在推理前缀中间。
        messages = [
            {"role": "user", "content": "USER_INPUT"},
            {"role": "assistant", "content": "ANS"},
        ]
        text, _ = render_glm52_chat(messages, add_generation_prompt=True, **GLM52_TEMPLATE_DEFAULTS)
        assert "<|endoftext|>" not in text
        assert text.endswith("<|assistant|><think>")

        fast_ids, fast_labels = glm52_tokenize_fn_fastspeed(tokenizer, messages, add_generation_prompt=True)
        slow_ids, slow_labels = glm52_tokenize_fn_slowspeed(tokenizer, messages, add_generation_prompt=True)
        assert fast_ids == slow_ids
        assert fast_labels == slow_labels
