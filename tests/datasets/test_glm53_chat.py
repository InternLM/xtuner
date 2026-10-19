"""GLM-5.3-Flash 的 chat template 与 loss mask，见设计文档 F1。

TestGlm53Rendering
    test_reasoning_effort_domain_matches_jinja  reasoning_effort 取值域与 jinja 一致
    test_enable_thinking_false_rejected         enable_thinking=False 被显式拒绝（jinja 忽略该参数）
    test_all_generation_eos_are_supervised_at_assistant_boundaries  停止 token 计入 loss
    test_media_placeholder_matches_jinja        媒体占位符与 jinja 一致
TestGlm53OfficialTemplateContract
    test_render_matches_official_jinja          逐字符对齐官方 jinja
    test_mid_conversation_boundaries_use_their_actual_occurrence  对话中间无 EOS、边界 token loss 正确
"""

import os

import pytest

from transformers import AutoTokenizer
from xtuner.v1.data_proto.messages import Glm53ChatMessages
from xtuner.v1.data_proto.messages.glm53_chat import glm53_tokenize_fn_slowspeed, render_glm53_chat


GLM_5_3_FLASH_PATH = os.environ["GLM_5_3_FLASH_PATH"]
GLM53_TEMPLATE_DEFAULTS = {"enable_thinking": True, "reasoning_effort": "max", "clear_thinking": False}

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

        boundary_index = _token_index_after_context(tokenizer, official_text, "BOUNDARY_SENTINEL", boundary_token)
        boundary_id = tokenizer.convert_tokens_to_ids(boundary_token)
        assert tokenized["input_ids"][boundary_index] == boundary_id
        if boundary_is_supervised:
            assert tokenized["labels"][boundary_index] == boundary_id
        else:
            assert tokenized["labels"][boundary_index] == -100
