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
}


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
