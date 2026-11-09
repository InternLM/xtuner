import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from transformers import PreTrainedTokenizerFast
from xtuner.v1.rl.rollout.session_server import FMT_OPENAI, SessionServer


class TestSessionTemplateBoundary:
    @pytest.mark.parametrize("suffix", ["", "<eos>"])
    @pytest.mark.parametrize("reasoning", [False, True])
    def test_template_boundary_preserves_raw_trace(self, suffix: str, reasoning: bool) -> None:
        server = self._server(suffix)
        request, response = self._turn(reasoning)
        asyncio.run(server.on_response(response, FMT_OPENAI, request))

        writes = server.store.insert.remote.await_args_list
        assert len(writes) == 2
        old_prompt = writes[0].kwargs["key"]
        new_prompt = writes[1].kwargs["key"]
        segment = writes[1].kwargs["value"]
        assert new_prompt.startswith(old_prompt)
        assert new_prompt.endswith("<eos>") == bool(suffix)
        assert segment.text == ("thinking</think>" if reasoning else "</think>pong") + suffix
        # The template is only a cache key: do not synthesize EOS IDs or logprobs.
        assert segment.token_ids == [7, 8, 9]
        assert segment.labels == [7, 8, 9]
        assert segment.logprobs == [-0.1, -0.2, -0.3]
        assert segment.length == 3
        assert segment.expert_key is None

    def test_prefix_mismatch_rejected_before_trace_write(self) -> None:
        server = self._server("")
        server.tokenizer.chat_template += "{% if add_generation_prompt %}wrong-prefix{% endif %}"
        request, response = self._turn(False)
        with pytest.raises(RuntimeError, match="generation prompt prefix"):
            asyncio.run(server.on_response(response, FMT_OPENAI, request))
        server.store.insert.remote.assert_not_awaited()

    def test_invalid_logprobs_rejected_before_trace_write(self) -> None:
        server = self._server("")
        request, response = self._turn(False)
        response["choices"][0]["output_token_logprobs"] = [[-0.1, 99]]
        with pytest.raises(RuntimeError):
            asyncio.run(server.on_response(response, FMT_OPENAI, request))
        server.store.insert.remote.assert_not_awaited()

    @staticmethod
    def _server(suffix: str) -> SessionServer:
        # A real HF renderer, with GLM-style reasoning and optional turn EOS.
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
            eos_token="<eos>",
            chat_template=(
                "{% for message in messages %}{{ message['role'] }}:"
                "{% if message['role'] == 'assistant' %}<think>"
                "{{ message.get('reasoning_content') or '' }}</think>{% endif %}"
                "{{ message.get('content') or '' }}"
                + suffix
                + "{% endfor %}{% if add_generation_prompt %}assistant:<think>{% endif %}"
            ),
        )
        server = SessionServer.__new__(SessionServer)
        server.tokenizer = tokenizer
        server.stop_word = tokenizer.eos_token
        server.enable_return_routed_experts = False
        server.store = SimpleNamespace(
            search=SimpleNamespace(remote=AsyncMock(return_value=("", []))),
            insert=SimpleNamespace(remote=AsyncMock()),
        )
        return server

    @staticmethod
    def _turn(reasoning: bool) -> tuple[dict, dict]:
        request = {"session_id": "boundary", "messages": [{"role": "user", "content": "Reply with pong"}]}
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "" if reasoning else "pong",
                        "reasoning_content": "thinking" if reasoning else "",
                    },
                    "output_ids": [7, 8, 9],
                    "output_token_logprobs": [[-0.1, 7], [-0.2, 8], [-0.3, 9]],
                }
            ]
        }
        return request, response
