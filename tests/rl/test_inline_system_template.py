import copy
import json

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from transformers import PreTrainedTokenizerFast
from xtuner.v1.rl.rollout.chat_template import canonicalize_messages_for_chat_template


def tokenizer(inline):
    template = """{% for message in messages %}
{% if message.role == 'system' and not loop.first and not inline %}
{{ raise_exception('System message must be at the beginning.') }}
{% endif %}{{ message.role }}:{{ message.content }};{% endfor %}"""
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"<unk>": 0}, unk_token="<unk>")),
        unk_token="<unk>",
        chat_template=template.replace("not inline", "false" if inline else "true"),
    )


@pytest.mark.parametrize("inline", [False, True])
def test_inline_system_projection_matches_template_capability_without_mutation(inline):
    tok = tokenizer(inline)
    messages = [
        {"role": "system", "content": "first"},
        {"role": "user", "content": "task"},
        {"role": "system", "content": "late"},
    ]
    original = copy.deepcopy(messages)
    if not inline:
        with pytest.raises(Exception, match="System message must be at the beginning"):
            tok.apply_chat_template(messages, tokenize=False)
    projected = canonicalize_messages_for_chat_template(messages, tokenizer=tok)
    prompt = tok.apply_chat_template(projected, tokenize=False)
    assert "first" in prompt and "late" in prompt and "task" in prompt
    assert [m["role"] for m in projected] == (["system", "user", "system"] if inline else ["system", "user"])
    if not inline:
        assert projected[0]["content"] == "firstlate"
    assert messages == original


def test_projection_preserves_system_content_blocks_and_tool_arguments():
    messages = [
        {"role": "system", "content": "first"},
        {"role": "user", "content": "task"},
        {"role": "system", "content": [{"type": "text", "text": "late"}]},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "Bash", "arguments": '{"command":"pwd"}'}}],
        },
    ]
    original = copy.deepcopy(messages)
    projected = canonicalize_messages_for_chat_template(messages, tokenizer=tokenizer(False))
    assert projected[0]["content"] == [{"type": "text", "text": "first"}, {"type": "text", "text": "late"}]
    assert projected[-1]["tool_calls"][0]["function"]["arguments"] == {"command": "pwd"}
    assert messages == original


@pytest.mark.parametrize("encoded", [False, True])
def test_sdk_argument_reordering_preserves_the_rendered_trace_key(encoded):
    tok = tokenizer(True)
    tok.chat_template = """{% for m in messages %}{% for call in m.tool_calls %}
{% for key, value in call.function.arguments.items() %}{{ key }}={{ value | tojson }};
{% endfor %}{% endfor %}{% endfor %}"""
    first = {"description": "write result", "command": "printf neutral", "nested": [{"z": 1, "a": 2}]}
    reordered = {"command": "printf neutral", "nested": [{"a": 2, "z": 1}], "description": "write result"}

    def render(arguments):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "Bash", "arguments": json.dumps(arguments) if encoded else arguments}}
                ],
            }
        ]
        original = copy.deepcopy(messages)
        result = tok.apply_chat_template(
            canonicalize_messages_for_chat_template(messages, tokenizer=tok), tokenize=False
        )
        assert messages == original
        return result

    assert render(first) == render(reordered)
    assert list(first) == ["description", "command", "nested"]
