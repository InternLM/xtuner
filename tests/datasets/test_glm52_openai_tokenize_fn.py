import os

from transformers import AutoTokenizer

from xtuner.v1.data_proto.messages import Glm52ChatMessages
from xtuner.v1.data_proto.messages.glm52_chat import glm52_tokenize_fn_slowspeed
from xtuner.v1.data_proto.templates import HybridChatTemplate
from xtuner.v1.datasets import OpenaiTokenizeFunctionConfig


GLM52_TOKENIZER = os.environ["GLM5_2_MOE_PATH"]


def _ids_from_hf_render(tokenizer, messages, **kwargs):
    text = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
    return tokenizer.encode(text, add_special_tokens=False)


def _label_flags_for_span(tokenizer, text, labels, substring):
    start = text.index(substring)
    end = start + len(substring)
    offsets = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
    return [label != -100 for label, (left, right) in zip(labels, offsets) if left < end and right > start]


def test_glm52_plain_text_matches_hf_rendering_and_slow_labels():
    tokenizer = AutoTokenizer.from_pretrained(GLM52_TOKENIZER, trust_remote_code=True)
    tokenize_fn = OpenaiTokenizeFunctionConfig(chat_template="glm5.2").build(tokenizer)
    messages = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there."},
    ]

    tokenized = tokenize_fn({"messages": messages})
    expected_input_ids = _ids_from_hf_render(tokenizer, messages, add_generation_prompt=False)
    slow_input_ids, slow_labels = glm52_tokenize_fn_slowspeed(tokenizer, messages)

    assert tokenized["input_ids"] == expected_input_ids
    assert tokenized["input_ids"] == slow_input_ids
    assert tokenized["labels"] == slow_labels
    assert any(label != -100 for label in tokenized["labels"])


def test_glm52_reasoning_traces_are_supervised_but_template_scaffolding_is_masked():
    tokenizer = AutoTokenizer.from_pretrained(GLM52_TOKENIZER, trust_remote_code=True)
    tokenize_fn = OpenaiTokenizeFunctionConfig(chat_template="glm5.2").build(tokenizer)
    messages = [
        {"role": "user", "content": "Question one"},
        {"role": "assistant", "reasoning_content": "old trace", "content": "Old answer."},
        {"role": "user", "content": "Question two"},
        {"role": "assistant", "reasoning_content": "new trace", "content": "Final answer."},
    ]

    tokenized = tokenize_fn({"messages": messages})
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    slow_input_ids, slow_labels = glm52_tokenize_fn_slowspeed(tokenizer, messages)

    assert tokenized["input_ids"] == tokenizer.encode(rendered, add_special_tokens=False)
    assert tokenized["input_ids"] == slow_input_ids
    assert tokenized["labels"] == slow_labels
    assert all(not flag for flag in _label_flags_for_span(tokenizer, rendered, tokenized["labels"], "<think></think>"))
    assert all(flag for flag in _label_flags_for_span(tokenizer, rendered, tokenized["labels"], "new trace</think>"))
    assert all(flag for flag in _label_flags_for_span(tokenizer, rendered, tokenized["labels"], "Final answer."))


def test_glm52_tools_tool_calls_tool_results_and_loss_disabling():
    tokenizer = AutoTokenizer.from_pretrained(GLM52_TOKENIZER, trust_remote_code=True)
    tokenize_fn = OpenaiTokenizeFunctionConfig(chat_template="glm5.2").build(tokenizer)
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
    assert all(not flag for flag in _label_flags_for_span(tokenizer, rendered, tokenized["labels"], '"description": "Gets the weather."'))
    assert all(flag for flag in _label_flags_for_span(tokenizer, rendered, tokenized["labels"], "Let me check"))
    assert all(flag for flag in _label_flags_for_span(tokenizer, rendered, tokenized["labels"], "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris"))
    assert all(not flag for flag in _label_flags_for_span(tokenizer, rendered, tokenized["labels"], "<tool_response>22C</tool_response>"))
    assert all(not flag for flag in _label_flags_for_span(tokenizer, rendered, tokenized["labels"], "It is 22C."))


def test_glm52_generation_prompt_matches_hf_and_is_masked():
    tokenizer = AutoTokenizer.from_pretrained(GLM52_TOKENIZER, trust_remote_code=True)
    messages = [{"role": "user", "content": "Write a short answer."}]

    tokenized = Glm52ChatMessages(messages=messages).tokenize(tokenizer, add_generation_prompt=True)
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    assert tokenized["input_ids"] == tokenizer.encode(rendered, add_special_tokens=False)
    assert all(label == -100 for label in tokenized["labels"])


def test_glm52_default_system_inserts_or_replaces_system_message():
    tokenizer = AutoTokenizer.from_pretrained(GLM52_TOKENIZER, trust_remote_code=True)
    chat_template = HybridChatTemplate(default_system="Default system instruction.")

    inserted_messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    inserted = Glm52ChatMessages(messages=inserted_messages).tokenize(tokenizer, chat_template)
    expected_inserted = [{"role": "system", "content": "Default system instruction."}] + inserted_messages
    inserted_rendered = tokenizer.apply_chat_template(expected_inserted, tokenize=False, add_generation_prompt=False)

    replaced_messages = [
        {"role": "system", "content": "Original system instruction."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    replaced = Glm52ChatMessages(messages=replaced_messages).tokenize(tokenizer, chat_template)
    expected_replaced = [
        {"role": "system", "content": "Default system instruction."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    replaced_rendered = tokenizer.apply_chat_template(expected_replaced, tokenize=False, add_generation_prompt=False)

    assert inserted["input_ids"] == tokenizer.encode(inserted_rendered, add_special_tokens=False)
    assert replaced["input_ids"] == tokenizer.encode(replaced_rendered, add_special_tokens=False)
    assert all(
        not flag
        for flag in _label_flags_for_span(
            tokenizer, replaced_rendered, replaced["labels"], "Default system instruction."
        )
    )
    assert all(flag for flag in _label_flags_for_span(tokenizer, replaced_rendered, replaced["labels"], "Hi"))


def test_glm52_truncation_keeps_input_ids_and_labels_aligned():
    tokenizer = AutoTokenizer.from_pretrained(GLM52_TOKENIZER, trust_remote_code=True)
    full_tokenize_fn = OpenaiTokenizeFunctionConfig(chat_template="glm5.2").build(tokenizer)
    truncated_tokenize_fn = OpenaiTokenizeFunctionConfig(chat_template="glm5.2", max_length=12).build(tokenizer)
    messages = [
        {"role": "user", "content": "Explain why the sky is blue in one sentence."},
        {"role": "assistant", "content": "Because air scatters shorter blue wavelengths more than red wavelengths."},
    ]

    full = full_tokenize_fn({"messages": messages})
    truncated = truncated_tokenize_fn({"messages": messages})

    assert truncated["input_ids"] == full["input_ids"][:12]
    assert truncated["labels"] == full["labels"][:12]
    assert truncated["num_tokens"] == 12
    assert len(truncated["input_ids"]) == len(truncated["labels"])
