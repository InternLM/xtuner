from xtuner.v1.data_proto.messages.qwen35_chat import qwen35_tokenize_fn_fastspeed


class _CharacterTokenizer:
    def __init__(self):
        self.text = ""

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=False):
        assert not add_special_tokens
        self.text = text
        output = {"input_ids": [ord(char) for char in text]}
        if return_offsets_mapping:
            output["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return output


def _labels_for(text, labels, value):
    start = text.index(value)
    return labels[start : start + len(value)]


def _is_trained(text, input_ids, labels, value):
    start = text.index(value)
    end = start + len(value)
    return labels[start:end] == input_ids[start:end]


def _is_masked(text, labels, value):
    return all(label == -100 for label in _labels_for(text, labels, value))


def test_qwen35_trains_every_assistant_turn():
    messages = [
        {"role": "system", "content": "SYSTEM_TEXT"},
        {"role": "user", "content": "FIRST_USER_TEXT"},
        {
            "role": "assistant",
            "reasoning_content": "FIRST_THINK_TEXT",
            "content": "FIRST_ANSWER_TEXT",
        },
        {"role": "user", "content": "SECOND_USER_TEXT"},
        {
            "role": "assistant",
            "reasoning_content": "SECOND_THINK_TEXT",
            "content": "SECOND_ANSWER_TEXT",
        },
        {
            "role": "assistant",
            "reasoning_content": "DISABLED_THINK_TEXT",
            "content": "DISABLED_ANSWER_TEXT",
            "loss": False,
        },
    ]

    tokenizer = _CharacterTokenizer()
    input_ids, labels = qwen35_tokenize_fn_fastspeed(messages, tokenizer=tokenizer)
    text = tokenizer.text

    assert _is_trained(text, input_ids, labels, "FIRST_THINK_TEXT")
    assert _is_trained(text, input_ids, labels, "FIRST_ANSWER_TEXT")
    assert _is_trained(text, input_ids, labels, "SECOND_THINK_TEXT")
    assert _is_trained(text, input_ids, labels, "SECOND_ANSWER_TEXT")

    assert _is_masked(text, labels, "SYSTEM_TEXT")
    assert _is_masked(text, labels, "FIRST_USER_TEXT")
    assert _is_masked(text, labels, "SECOND_USER_TEXT")
    assert _is_masked(text, labels, "DISABLED_THINK_TEXT")
    assert _is_masked(text, labels, "DISABLED_ANSWER_TEXT")


def test_qwen35_trains_tool_calls_but_not_tool_responses():
    messages = [
        {"role": "system", "content": "SYSTEM_TEXT"},
        {"role": "user", "content": "USER_TEXT"},
        {
            "role": "assistant",
            "reasoning_content": "TOOL_THINK_TEXT",
            "content": "TOOL_ANSWER_TEXT",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "UNIQUE_TOOL_NAME",
                        "arguments": {"argument": "UNIQUE_ARGUMENT_VALUE"},
                    },
                }
            ],
        },
        {"role": "tool", "content": "UNIQUE_TOOL_RESPONSE"},
        {"role": "assistant", "content": "FINAL_ANSWER_TEXT"},
    ]

    tokenizer = _CharacterTokenizer()
    input_ids, labels = qwen35_tokenize_fn_fastspeed(messages, tokenizer=tokenizer)
    text = tokenizer.text

    assert _is_trained(text, input_ids, labels, "TOOL_THINK_TEXT")
    assert _is_trained(text, input_ids, labels, "TOOL_ANSWER_TEXT")
    assert _is_trained(text, input_ids, labels, "UNIQUE_TOOL_NAME")
    assert _is_trained(text, input_ids, labels, "UNIQUE_ARGUMENT_VALUE")
    assert _is_trained(text, input_ids, labels, "FINAL_ANSWER_TEXT")
    assert _is_masked(text, labels, "UNIQUE_TOOL_RESPONSE")
