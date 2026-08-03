from xtuner.v1.data_proto.messages.qwen35_chat import qwen35_tokenize_fn_fastspeed
from xtuner.v1.data_proto.messages.qwen36_chat import qwen36_tokenize_fn_fastspeed


def _render(render_fn, arguments):
    messages = [
        {"role": "user", "content": "call the tool"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "demo", "arguments": arguments}}],
        },
    ]
    text, _ = render_fn(messages, return_labels=False)
    return text


def test_qwen36_only_changes_tool_argument_serialization():
    arguments = {
        "string": "raw text",
        "truth": True,
        "falsehood": False,
        "nothing": None,
        "number": 3,
        "mapping": {"enabled": True},
        "sequence": [False, None],
    }

    qwen35_text = _render(qwen35_tokenize_fn_fastspeed, arguments)
    qwen36_text = _render(qwen36_tokenize_fn_fastspeed, arguments)

    assert qwen35_text.replace("True", "true").replace("False", "false").replace("None", "null") == qwen36_text
    assert "<parameter=string>\nraw text\n</parameter>" in qwen36_text
    assert "<parameter=truth>\ntrue\n</parameter>" in qwen36_text
    assert "<parameter=falsehood>\nfalse\n</parameter>" in qwen36_text
    assert "<parameter=nothing>\nnull\n</parameter>" in qwen36_text
    assert '<parameter=mapping>\n{"enabled": true}\n</parameter>' in qwen36_text
    assert "<parameter=sequence>\n[false, null]\n</parameter>" in qwen36_text


def test_qwen36_preserves_qwen35_non_tool_rendering():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "reasoning_content": "first thought", "content": "first answer"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "reasoning_content": "second thought", "content": "second answer"},
    ]

    qwen35_text, qwen35_loss_mask = qwen35_tokenize_fn_fastspeed(messages, return_labels=False)
    qwen36_text, qwen36_loss_mask = qwen36_tokenize_fn_fastspeed(messages, return_labels=False)

    assert qwen36_text == qwen35_text
    assert qwen36_loss_mask == qwen35_loss_mask
