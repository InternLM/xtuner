# Copyright (c) OpenMMLab. All rights reserved.
"""GLM text-SFT literals, template controls, and loss masks.

The byte-level fixture runs without a checkpoint. Set GLM_LITERAL_TOKENIZER_PATH for the same cases against an official
GLM tokenizer (no model weights needed).
"""

import copy
import json
import os

import pytest
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers

from transformers import AutoTokenizer, PreTrainedTokenizerFast
from xtuner.v1.data_proto.messages import glm52_chat, glm53_chat
from xtuner.v1.datasets.sft_tokenize_fn import openai
from xtuner.v1.datasets.sft_tokenize_fn.openai import OpenaiTokenizeFunctionConfig


GLM_TOKENS = [
    "<|endoftext|>",
    "[MASK]",
    "[gMASK]",
    "[sMASK]",
    "<sop>",
    "<eop>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|observation|>",
    "<|begin_of_image|>",
    "<|end_of_image|>",
    "<|begin_of_video|>",
    "<|end_of_video|>",
    "<|begin_of_audio|>",
    "<|end_of_audio|>",
    "<|begin_of_transcription|>",
    "<|end_of_transcription|>",
    "<|code_prefix|>",
    "<|code_middle|>",
    "<|code_suffix|>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<arg_key>",
    "</arg_key>",
    "<arg_value>",
    "</arg_value>",
    "/nothink",
    "<|begin_of_box|>",
    "<|end_of_box|>",
    "<|image|>",
    "<|video|>",
]


@pytest.fixture(scope="module", params=["byte_level", "official"])
def tokenizer(request):
    if request.param == "official":
        path = os.environ.get("GLM_LITERAL_TOKENIZER_PATH")
        if not path:
            pytest.skip("Set GLM_LITERAL_TOKENIZER_PATH for official-tokenizer integration tests")
        return AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    vocab = {token: i for i, token in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    vocab['"<'] = len(vocab)
    vocab['>"'] = len(vocab)
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[('"', "<"), (">", '"')]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.add_tokens([AddedToken(token, special=i < 18, normalized=False) for i, token in enumerate(GLM_TOKENS)])
    return PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<|endoftext|>", pad_token="<|endoftext|>")


@pytest.fixture(params=[glm52_chat, glm53_chat], ids=["glm52", "glm53"])
def chat(request):
    module = request.param
    name = "glm52" if module is glm52_chat else "glm53"
    return (
        getattr(module, f"render_{name}_chat"),
        getattr(module, f"{name}_tokenize_fn_fastspeed"),
        getattr(module, f"{name}_tokenize_fn_slowspeed"),
    )


def _fixture(slot, literal):
    messages = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer"},
    ]
    tools = None
    call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "lookup", "arguments": {"query": "value"}}}],
    }
    if slot == "system":
        messages.insert(0, {"role": "system", "content": literal})
    elif slot == "user":
        messages[0]["content"] = literal
    elif slot == "reasoning":
        messages[1]["reasoning_content"] = literal
    elif slot == "assistant":
        messages[1]["content"] = literal
    elif slot in ("arg_key", "arg_value", "tool_name"):
        messages[1] = call
        function = call["tool_calls"][0]["function"]
        if slot == "arg_key":
            function["arguments"] = {literal: "value"}
        elif slot == "arg_value":
            function["arguments"] = {"query": literal}
        else:
            function["name"] = literal
    elif slot == "tool_result":
        messages.insert(1, call)
        messages.insert(2, {"role": "tool", "content": literal})
    elif slot == "schema":
        tools = [{"type": "function", "function": {"name": "lookup", "description": literal}}]
    return messages, tools


@pytest.mark.parametrize("literal", GLM_TOKENS)
def test_all_added_literals_keep_only_template_controls(tokenizer, chat, literal):
    render, fast, _ = chat
    added_ids = set(tokenizer.get_added_vocab().values())
    for context in (
        literal,
        f"中文{literal}继续",
        f'x = "{literal}"',
        json.dumps({"s": literal}),
        f"before\n{literal}\nafter",
    ):
        for slot in (
            "system",
            "user",
            "reasoning",
            "assistant",
            "arg_key",
            "arg_value",
            "tool_name",
            "tool_result",
            "schema",
        ):
            messages, tools = _fixture(slot, context)
            clean_messages, clean_tools = _fixture(slot, "ordinary text")
            ids, labels = fast(tokenizer, messages, tools=tools)
            clean_ids, clean_labels = fast(tokenizer, clean_messages, tools=clean_tools)
            assert [(i, label) for i, label in zip(ids, labels) if i in added_ids] == [
                (i, label) for i, label in zip(clean_ids, clean_labels) if i in added_ids
            ]
            text, _ = render(messages, tools=tools)
            assert tokenizer.decode(ids, skip_special_tokens=False) == text
            assert all(label in (-100, token_id) for token_id, label in zip(ids, labels))


@pytest.mark.parametrize(
    "literal",
    ["<|endoftext|>", "<|endoftext|><|endoftext|>", '使用 "<tool_call>" 来标识', "</think>", "🧪中文<|user|>继续"],
)
def test_body_matches_independent_base_bpe_and_keeps_eos(tokenizer, chat, literal):
    _, fast, slow = chat
    messages, _ = _fixture("assistant", literal)
    serialized = json.loads(tokenizer.backend_tokenizer.to_str())
    serialized["added_tokens"] = []
    base = Tokenizer.from_str(json.dumps(serialized))
    base.no_padding()
    base.no_truncation()
    expected = base.encode(literal, add_special_tokens=False).ids
    ids, labels = fast(tokenizer, messages)
    start = ids.index(tokenizer.convert_tokens_to_ids("</think>")) + 1
    assert ids[start:-1] == labels[start:-1] == expected
    assert ids[-1] == labels[-1] == tokenizer.eos_token_id
    assert ids.count(tokenizer.eos_token_id) == 1
    assert (ids, labels) == slow(tokenizer, messages)


@pytest.mark.parametrize("reasoning", [{}, {"reasoning_content": None}, {"reasoning_content": ""}])
@pytest.mark.parametrize(
    "literal", ["讨论 </think> 的用法", "<think>这也是正文</think>答案", "前</think>中</think>后"]
)
def test_content_never_infers_reasoning_from_think_literals(tokenizer, chat, reasoning, literal):
    render, fast, slow = chat
    message = {"role": "assistant", "content": literal, **reasoning}
    text, _ = render([message])
    assert text.endswith("<|assistant|><think></think>" + literal + "<|endoftext|>")
    serialized = json.loads(tokenizer.backend_tokenizer.to_str())
    serialized["added_tokens"] = []
    base = Tokenizer.from_str(json.dumps(serialized))
    base.no_padding()
    base.no_truncation()
    for clear in (False, True):
        ids, labels = fast(tokenizer, [message], clear_thinking=clear)
        start = ids.index(tokenizer.convert_tokens_to_ids("</think>")) + 1
        assert ids[start:-1] == labels[start:-1] == base.encode(literal, add_special_tokens=False).ids
        assert ids.count(tokenizer.convert_tokens_to_ids("<think>")) == 1
        assert ids.count(tokenizer.convert_tokens_to_ids("</think>")) == 1
        assert (ids, labels) == slow(tokenizer, [message], clear_thinking=clear)


def test_think_literals_in_history_when_clearing_reasoning(tokenizer, chat):
    render, fast, slow = chat
    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "正文 </think> 保留"},
        {"role": "user", "content": "continue"},
        {"role": "assistant", "content": "answer", "reasoning_content": "actual reasoning"},
    ]
    # 两种 clear_thinking 取值下 fast 与 slow 的标签都必须一致。
    assert fast(tokenizer, messages, clear_thinking=True) == slow(tokenizer, messages, clear_thinking=True)
    assert fast(tokenizer, messages, clear_thinking=False) == slow(tokenizer, messages, clear_thinking=False)

    ids_cleared, labels_cleared = fast(tokenizer, messages, clear_thinking=True)
    ids_kept, labels_kept = fast(tokenizer, messages, clear_thinking=False)
    if render is glm52_chat.render_glm52_chat:
        # glm52 语义：clear_thinking 只屏蔽带 reasoning 的历史轮，无 reasoning 的历史 content 照样监督。
        assert (ids_cleared, labels_cleared) == (ids_kept, labels_kept)
        return
    # glm53 语义：clear_thinking 屏蔽最后一个 user 之前的整轮历史 assistant（含无 reasoning 的
    # 轮次，其 content 里的字面 </think> 拼写一并不再监督）；最后一轮照常监督。
    assert (ids_cleared, labels_cleared) != (ids_kept, labels_kept)
    assistant_id = tokenizer.convert_tokens_to_ids("<|assistant|>")
    user_id = tokenizer.convert_tokens_to_ids("<|user|>")
    first_assistant = ids_cleared.index(assistant_id)
    boundary_user = ids_cleared.index(user_id, first_assistant)
    assert all(label == -100 for label in labels_cleared[first_assistant:boundary_user])
    assert any(label != -100 for label in labels_cleared[boundary_user:])


@pytest.mark.parametrize("suffix", ["<|", "<", "</", "</think", "<|assistant", "<|observation"])
@pytest.mark.parametrize("next_role", ["assistant", "tool"])
def test_partial_token_suffix_preserves_following_control(tokenizer, chat, suffix, next_role):
    _, fast, slow = chat
    messages = [{"role": "user", "content": "question"}]
    if next_role == "tool":
        messages.append({"role": "assistant", "content": ""})
    messages[-1]["content"] = "literal </think> followed by " + suffix
    content_index = len(messages) - 1
    messages.append({"role": next_role, "content": "next message"})
    clean = copy.deepcopy(messages)
    clean[content_index]["content"] = "ordinary text"
    ids, labels = fast(tokenizer, messages)
    clean_ids, clean_labels = fast(tokenizer, clean)
    added_ids = set(tokenizer.get_added_vocab().values())
    assert [(i, label) for i, label in zip(ids, labels) if i in added_ids] == [
        (i, label) for i, label in zip(clean_ids, clean_labels) if i in added_ids
    ]
    boundary = "<|assistant|>" if next_role == "assistant" else "<|observation|>"
    assert ids.count(tokenizer.convert_tokens_to_ids(boundary)) == 1
    assert (ids, labels) == slow(tokenizer, messages)


@pytest.mark.parametrize("loss", [False, True])
@pytest.mark.parametrize("next_role", [None, "user", "tool"])
def test_loss_flag_and_all_stop_boundaries(tokenizer, chat, loss, next_role):
    _, fast, slow = chat
    messages, _ = _fixture("arg_value", "<|endoftext|></think><tool_call>")
    messages[1]["loss"] = loss
    if next_role is not None:
        messages.append({"role": next_role, "content": "context"})
    ids, labels = fast(tokenizer, messages)
    assert (ids, labels) == slow(tokenizer, messages)
    assert any(label != -100 for label in labels) == loss
    stop = {None: "<|endoftext|>", "user": "<|user|>", "tool": "<|observation|>"}[next_role]
    stop_id = tokenizer.convert_tokens_to_ids(stop)
    assert [label for i, label in zip(ids, labels) if i == stop_id][-1] == (stop_id if loss else -100)


def test_ordinary_ids_labels_and_render_api_are_unchanged(tokenizer, chat):
    render, fast, slow = chat
    for slot in ("assistant", "reasoning", "arg_value", "tool_result", "schema"):
        messages, tools = _fixture(slot, "ordinary 中文 text")
        text, char_mask = render(messages, tools=tools)
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        expected_labels = [
            i if start != end and any(char_mask[start:end]) else -100
            for i, (start, end) in zip(encoded["input_ids"], encoded["offset_mapping"])
        ]
        spans = []
        assert render(messages, tools=tools, _literal_spans=spans) == (text, char_mask)
        assert (encoded["input_ids"], expected_labels) == fast(tokenizer, messages, tools=tools)
        assert fast(tokenizer, messages, tools=tools) == slow(tokenizer, messages, tools=tools)


def test_original_tokenizer_and_cached_backend_are_preserved(tokenizer, chat):
    _, fast, _ = chat
    before = tokenizer.backend_tokenizer.to_str()
    messages, _ = _fixture("assistant", "<tool_call><|endoftext|>")
    fast(tokenizer, messages)
    backend = glm52_chat._content_tokenizer(tokenizer)
    fast(tokenizer, messages)
    assert glm52_chat._content_tokenizer(tokenizer) is backend
    assert backend.get_vocab() == tokenizer.get_vocab()
    assert all(token.special for token in backend.get_added_tokens_decoder().values())
    assert tokenizer.backend_tokenizer.to_str() == before


def test_content_backend_rejects_missing_special_promotion(tokenizer, monkeypatch):
    class BackendWithoutSpecialPromotion:
        def add_special_tokens(self, tokens):
            pass

        def get_added_tokens_decoder(self):
            return {0: AddedToken("</think>", special=False)}

    monkeypatch.setattr(glm52_chat.copy, "deepcopy", lambda _: BackendWithoutSpecialPromotion())
    with pytest.raises(AssertionError, match="special"):
        glm52_chat._content_tokenizer.__wrapped__(tokenizer)


def test_multiturn_reasoning_clearing_and_generation_prompt(tokenizer, chat):
    _, fast, slow = chat
    messages = [
        {"role": "user", "content": "<|endoftext|>"},
        {"role": "assistant", "content": "old", "reasoning_content": "<tool_call>"},
        {"role": "user", "content": "continue"},
        {"role": "assistant", "content": "<|endoftext|>", "reasoning_content": "</think>"},
    ]
    for clear in (False, True):
        assert fast(tokenizer, messages, clear_thinking=clear) == slow(tokenizer, messages, clear_thinking=clear)
    ids, labels = fast(tokenizer, messages[:1], add_generation_prompt=True)
    assert all(label == -100 for label in labels)
    assert ids[-1] == tokenizer.convert_tokens_to_ids("<think>")


def test_vl_default_call_keeps_media_and_literal_encoding(tokenizer):
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "<|image|>"}, {"type": "image"}]},
        {"role": "assistant", "content": "ok"},
    ]
    text, mask = glm53_chat.render_glm53_chat(messages)
    ids, _ = glm52_chat._tokenize_with_loss_mask(tokenizer, text, mask)
    assert ids == tokenizer.encode(text, add_special_tokens=False)
    image_id = tokenizer.convert_tokens_to_ids("<|image|>")
    assert ids.count(image_id) == 2  # VL callers deliberately have no literal-spans argument.
    text_ids, _ = glm53_chat.glm53_tokenize_fn_fastspeed(tokenizer, messages)
    assert text_ids.count(image_id) == 1  # The real media item still retains template origin.


def test_tool_result_variants_and_nested_json(tokenizer, chat):
    _, fast, slow = chat
    messages, tools = _fixture("schema", "<tool_call><|endoftext|>")
    messages[1]["tool_calls"] = [{"function": {"name": "lookup", "arguments": {"data": {"s": "<|endoftext|>"}}}}]
    for content in (
        [{"output": "<tool_call>"}, {"output": "<|endoftext|>"}],
        [{"type": "tool_reference", "name": "lookup"}],
        [{"type": "text", "text": "<tool_call>"}],
    ):
        dialog = [*messages, {"role": "tool", "content": content}, {"role": "assistant", "content": "done"}]
        assert fast(tokenizer, dialog, tools=tools) == slow(tokenizer, dialog, tools=tools)


def test_openai_config_cache_version_and_explicit_override(tokenizer):
    for template in ("glm5.2", "glm5.3"):
        function = OpenaiTokenizeFunctionConfig(chat_template=template).build(tokenizer)
        assert function.hash().endswith("_literal_added_tokens_v2")
        messages, _ = _fixture("assistant", "<|endoftext|>")
        function.set_state("cache")
        cached = function({"messages": copy.deepcopy(messages)})
        function.set_state("runtime")
        runtime = function({"messages": copy.deepcopy(messages)})
        assert cached["input_ids"] == runtime["input_ids"]
        assert cached["num_tokens"] == len(runtime["input_ids"])
        assert runtime["input_ids"].count(tokenizer.eos_token_id) == 1
        assert (
            OpenaiTokenizeFunctionConfig(chat_template=template, hash="user-version").build(tokenizer).hash()
            == "user-version"
        )
    assert (
        not OpenaiTokenizeFunctionConfig(chat_template="qwen3")
        .build(tokenizer)
        .hash()
        .endswith("_literal_added_tokens_v2")
    )


def test_explicit_glm_hash_warns_without_overriding_it(tokenizer, monkeypatch):
    warnings = []
    monkeypatch.setattr(openai.logger, "warning", warnings.append)
    for template in ("glm5.2", "glm5.3"):
        warnings.clear()
        function = OpenaiTokenizeFunctionConfig(chat_template=template, hash="user-version").build(tokenizer)
        assert function.hash() == function.hash() == "user-version"
        assert len(warnings) == 1
        assert "explicit hash" in warnings[0] and "rebuild" in warnings[0]
    warnings.clear()
    OpenaiTokenizeFunctionConfig(chat_template="glm5.3").build(tokenizer).hash()
    OpenaiTokenizeFunctionConfig(chat_template="qwen3", hash="user-version").build(tokenizer).hash()
    assert warnings == []


@pytest.mark.parametrize("aggregated", [False, True])
def test_sorted_glm53_tool_results_preserve_literal_spans(tokenizer, aggregated):
    calls = [
        {"id": "a", "function": {"name": "lookup", "arguments": {}}},
        {"id": "b", "function": {"name": "lookup", "arguments": {}}},
    ]
    if aggregated:
        results = [
            {"role": "tool", "content": [{"id": "b", "output": "B<tool_call>"}, {"id": "a", "output": "A</think>"}]}
        ]
    else:
        results = [
            {"role": "tool", "tool_call_id": "b", "content": "B<tool_call>"},
            {"role": "tool", "tool_call_id": "a", "content": "A</think>"},
        ]
    messages = [
        {"role": "assistant", "content": "", "tool_calls": calls},
        *results,
        {"role": "assistant", "content": "done"},
    ]
    text, _ = glm53_chat.render_glm53_chat(messages)
    assert text.index("A</think>") < text.index("B<tool_call>")
    ids, labels = glm53_chat.glm53_tokenize_fn_fastspeed(tokenizer, messages)
    assert ids.count(tokenizer.convert_tokens_to_ids("</think>")) == 2
    assert ids.count(tokenizer.convert_tokens_to_ids("<tool_call>")) == 2
    assert tokenizer.decode(ids, skip_special_tokens=False) == text
    assert (ids, labels) == glm53_chat.glm53_tokenize_fn_slowspeed(tokenizer, messages)
