from xtuner.v1.rl.agent_loop.multi_turn_tool_agent_loop import (
    PROFILES,
    _parse_forced_call,
    parse_qwen_tool_call,
    parse_tool_call,
)


def test_parse_glm_tool_call() -> None:
    text = "<tool_call>get_page_text<arg_key>page</arg_key><arg_value>3</arg_value></tool_call>"
    assert parse_tool_call(text) == ("get_page_text", {"page": "3"})


def test_parse_glm_tool_call_returns_none_without_call() -> None:
    assert parse_tool_call("plain answer") is None


def test_parse_qwen_tool_call() -> None:
    text = '<tool_call>{"name": "get_doc_meta", "arguments": {}}</tool_call>'
    assert parse_qwen_tool_call(text) == ("get_doc_meta", {})


def test_parse_qwen_tool_call_returns_none_for_invalid_json() -> None:
    assert parse_qwen_tool_call("<tool_call>{not-json}</tool_call>") is None


def test_parse_forced_call() -> None:
    assert _parse_forced_call("get_page_image(page=1)") == ("get_page_image", {"page": "1"})
    assert _parse_forced_call("get_doc_meta()") == ("get_doc_meta", {})


def test_profiles_define_tool_wrappers() -> None:
    assert set(PROFILES) == {"glm53", "qwen35_vl"}
    for profile in PROFILES.values():
        assert "{}" in profile.tool_wrap
        assert profile.image_markup
