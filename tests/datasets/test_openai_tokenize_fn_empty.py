from unittest.mock import Mock

from transformers import PreTrainedTokenizer
from xtuner.v1.datasets import OpenaiTokenizeFunctionConfig


class TestOpenaiTokenizeFunctionEmptyMessages:
    def test_empty_messages_are_marked_as_damaged_in_cache(self):
        tokenize_fn = OpenaiTokenizeFunctionConfig(chat_template="qwen3.5-vl").build(Mock(spec=PreTrainedTokenizer))
        tokenize_fn.set_state("cache")

        assert tokenize_fn([]) == {"num_tokens": 0, "proxy_attn_flops": 0.0}
