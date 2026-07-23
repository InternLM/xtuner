"""GLM-5.2 OpenAI 对话分词行为测试。

TestGlm52Rendering
    test_plain_text_matches_hf_and_golden_labels: 普通对话与 HF 模板及慢速 golden 对齐。
    test_multiturn_reasoning_masks_history_and_supervises_latest_answer: 多轮推理只监督有效生成内容。
    test_tools_and_loss_switch_follow_template_masking: 工具对话与 loss 开关生成正确标签。
TestGlm52MessageOptions
    test_generation_prompt_matches_hf_and_is_masked: generation prompt 与 HF 一致且不计 loss。
    test_default_system_is_inserted_or_replaced: 默认 system 消息可插入或替换并正确掩码。
"""

import os

import pytest

from transformers import AutoTokenizer
from xtuner.v1.data_proto.messages import Glm52ChatMessages
from xtuner.v1.data_proto.messages.glm52_chat import glm52_tokenize_fn_slowspeed
from xtuner.v1.data_proto.templates import HybridChatTemplate
from xtuner.v1.datasets import OpenaiTokenizeFunctionConfig


GLM52_TOKENIZER = os.environ["GLM5_2_MOE_PATH"]


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(GLM52_TOKENIZER, trust_remote_code=True)


@pytest.fixture(scope="module")
def tokenize_fn(tokenizer):
    return OpenaiTokenizeFunctionConfig(chat_template="glm5.2").build(tokenizer)


def _ids_from_hf_render(tokenizer, messages, **kwargs):
    text = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
    return tokenizer.encode(text, add_special_tokens=False)


def _label_flags_for_span(tokenizer, text, labels, substring):
    start = text.index(substring)
    end = start + len(substring)
    offsets = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
    return [label != -100 for label, (left, right) in zip(labels, offsets) if left < end and right > start]


class TestGlm52Rendering:
    def test_plain_text_matches_hf_and_golden_labels(self, tokenizer, tokenize_fn):
        # 验证普通对话的 token 与标签同时对齐 HF 模板和独立慢速实现。
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there."},
        ]

        tokenized = tokenize_fn({"messages": messages})
        slow_input_ids, slow_labels = glm52_tokenize_fn_slowspeed(tokenizer, messages)

        assert tokenized["input_ids"] == _ids_from_hf_render(tokenizer, messages, add_generation_prompt=False)
        assert tokenized["input_ids"] == slow_input_ids
        assert tokenized["labels"] == slow_labels
        assert (
            tokenizer.decode(
                [label for label in tokenized["labels"] if label != -100],
                skip_special_tokens=False,
            )
            == "</think>Hi there."
        )

    def test_multiturn_reasoning_masks_history_and_supervises_latest_answer(self, tokenizer, tokenize_fn):
        # 验证多轮对话清除历史 thinking，并只监督当前推理、答案和应训练的历史答案。
        messages = [
            {"role": "user", "content": "Question one"},
            {"role": "assistant", "reasoning_content": "old trace", "content": "Old answer."},
            {"role": "user", "content": "Question two"},
            {"role": "assistant", "reasoning_content": "new trace", "content": "Final answer."},
        ]

        tokenized = tokenize_fn({"messages": messages})
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        slow_input_ids, slow_labels = glm52_tokenize_fn_slowspeed(tokenizer, messages)

        assert "old trace" not in rendered
        assert tokenized["input_ids"] == tokenizer.encode(rendered, add_special_tokens=False)
        assert tokenized["input_ids"] == slow_input_ids
        assert tokenized["labels"] == slow_labels
        assert not any(_label_flags_for_span(tokenizer, rendered, tokenized["labels"], "<think></think>"))
        assert not any(_label_flags_for_span(tokenizer, rendered, tokenized["labels"], "Old answer."))
        assert all(_label_flags_for_span(tokenizer, rendered, tokenized["labels"], "new trace</think>"))
        assert all(_label_flags_for_span(tokenizer, rendered, tokenized["labels"], "Final answer."))

    def test_tools_and_loss_switch_follow_template_masking(self, tokenizer, tokenize_fn):
        # 验证工具定义与结果被掩码、assistant 工具调用被监督，且 loss=False 可关闭监督。
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Gets the weather.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                    "strict": True,
                },
            }
        ]
        messages = [
            {"role": "user", "content": "Weather in Paris?"},
            {
                "role": "assistant",
                "content": "Let me check",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": {"city": "Paris", "unit": "C"}},
                    }
                ],
            },
            {"role": "tool", "content": "22C"},
            {"role": "assistant", "content": "It is 22C.", "loss": False},
        ]

        tokenized = tokenize_fn({"messages": messages, "tools": tools})
        rendered = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=False)
        slow_input_ids, slow_labels = glm52_tokenize_fn_slowspeed(tokenizer, messages, tools=tools)

        assert tokenized["input_ids"] == tokenizer.encode(rendered, add_special_tokens=False)
        assert tokenized["input_ids"] == slow_input_ids
        assert tokenized["labels"] == slow_labels
        assert not any(
            _label_flags_for_span(tokenizer, rendered, tokenized["labels"], '"description": "Gets the weather."')
        )
        assert all(_label_flags_for_span(tokenizer, rendered, tokenized["labels"], "Let me check"))
        assert all(
            _label_flags_for_span(
                tokenizer,
                rendered,
                tokenized["labels"],
                "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris",
            )
        )
        assert not any(
            _label_flags_for_span(tokenizer, rendered, tokenized["labels"], "<tool_response>22C</tool_response>")
        )
        assert not any(_label_flags_for_span(tokenizer, rendered, tokenized["labels"], "It is 22C."))


class TestGlm52MessageOptions:
    def test_generation_prompt_matches_hf_and_is_masked(self, tokenizer):
        # 验证 generation prompt 的 token 与 HF 模板一致且全部不参与 loss。
        messages = [{"role": "user", "content": "Write a short answer."}]

        tokenized = Glm52ChatMessages(messages=messages).tokenize(tokenizer, add_generation_prompt=True)
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        assert tokenized["input_ids"] == tokenizer.encode(rendered, add_special_tokens=False)
        assert all(label == -100 for label in tokenized["labels"])

    def test_default_system_is_inserted_or_replaced(self, tokenizer):
        # 验证默认 system 指令既能补到无 system 对话，也能替换已有 system 指令。
        chat_template = HybridChatTemplate(default_system="Default system instruction.")
        inserted_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        replaced_messages = [
            {"role": "system", "content": "Original system instruction."},
            *inserted_messages,
        ]

        inserted = Glm52ChatMessages(messages=inserted_messages).tokenize(tokenizer, chat_template)
        replaced = Glm52ChatMessages(messages=replaced_messages).tokenize(tokenizer, chat_template)
        expected_messages = [
            {"role": "system", "content": "Default system instruction."},
            *inserted_messages,
        ]
        rendered = tokenizer.apply_chat_template(expected_messages, tokenize=False, add_generation_prompt=False)

        expected_ids = tokenizer.encode(rendered, add_special_tokens=False)
        assert inserted["input_ids"] == expected_ids
        assert replaced["input_ids"] == expected_ids
        assert not any(
            _label_flags_for_span(
                tokenizer,
                rendered,
                replaced["labels"],
                "Default system instruction.",
            )
        )
        assert all(_label_flags_for_span(tokenizer, rendered, replaced["labels"], "Hi"))
