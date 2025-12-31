import os
import parametrize
from unittest import TestCase
from transformers import AutoTokenizer

from xtuner.v1.datasets import OpenaiTokenizeFunctionConfig

QWEN3_PATH = os.environ["QWEN3_VL_DENSE_PATH"]  # We need instruct model


class TestOpenaiTokenizeFunction(TestCase):

    @parametrize.parametrize(
        "template_type, tokenizer_path",
        [
            ("qwen3", QWEN3_PATH),
        ],
    )
    def test_qwen3_tool_template(self, template_type, tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        tokenizer_fn_cfg = OpenaiTokenizeFunctionConfig(chat_template=template_type)
        tokenizer_fn = tokenizer_fn_cfg.build(tokenizer)

        messages = {
            "messages": [
                {"role": "system", "content": "这是系统消息"},
                {"role": "user", "content": "Hey, what's the temperature in Paris right now?"},
                {"role": "assistant", "content": "我需要先调用一些工具才能知道",
                 "tool_calls": [
                     {
                         "id": "call_123",
                         "type": "function",
                         "function": {
                             "name": "get_weather",
                             "arguments": "{\"location\": \"Boston\"}"
                         }
                     },
                     {
                         "id": "call_456",
                         "type": "function",
                         "function": {
                             "name": "get_weather",
                             "arguments": "{\"location\": \"beijing \"}"
                         }
                     }
                 ],
                 "reasoning_content": ""},
                {"role": "tool", "content": "22"},
                {"role": "assistant", "content": "你问的特别好"}
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_current_temperature",
                    "description": "Gets the temperature at a given location.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The location to get the temperature for"
                            }
                        },
                        "required": [
                            "location"
                        ]
                    }
                }
            },
                {"type": "function", "function": {"name": "get_current_wind_speed",
                                                  "description": "Get the current wind speed in km/h at a given "
                                                                 "location.",
                                                  "parameters": {"type": "object", "properties": {
                                                      "location": {"type": "string",
                                                                   "description": "The location to get the wind speed for, in the format \"City, Country\""}},
                                                                 "required": ["location"]}}}
            ],
        }
        input_ids_ref = tokenizer.apply_chat_template(
            messages["messages"],
            tools=messages["tools"],
            tokenize=True,
            add_generation_prompt=False,
        )
        input_ids = tokenizer_fn(messages)['input_ids']
        self.assertEqual(input_ids, input_ids_ref)
