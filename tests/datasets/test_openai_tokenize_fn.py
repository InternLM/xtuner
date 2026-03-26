import os
import parametrize
from unittest import TestCase
from transformers import AutoTokenizer
import unittest
from packaging.version import Version
from transformers import __version__ as transformers_version
import json
from xtuner.v1.datasets import OpenaiTokenizeFunctionConfig
import copy

QWEN3_PATH = os.environ["QWEN3_VL_DENSE_PATH"]  # We need instruct model


class TestOpenaiTokenizeFunction(TestCase):
    
    @unittest.skipIf(
        Version(transformers_version) < Version("5.2.0"),
        f"transformers >= 5.2.0 is required, but got {transformers_version}"
    )
    def test_qwen3p5_openai_tokenize_fn(self):
        QWEN3P5_PATH = os.environ["QWEN3_5_MOE_PATH"]
        demo_data_path = 'tests/resource/qwen35_tokenize_data.jsonl'

        tokenizer = AutoTokenizer.from_pretrained(QWEN3P5_PATH, trust_remote_code=True)
        tokenizer_fn_cfg = OpenaiTokenizeFunctionConfig(chat_template="qwen3.5-vl")
        tokenizer_fn = tokenizer_fn_cfg.build(tokenizer)
        
        all_data = []
        with open(demo_data_path, 'r') as f:
            for line in f:
                all_data.append(json.loads(line))
        
        for data in all_data:
            id = data["id"]
            if id == 7:
                data_with_system = copy.deepcopy(data)
                input_ids_ref = tokenizer.apply_chat_template(data_with_system['messages'], tools=data.get('tools'), tokenize=True, add_generation_prompt=False)['input_ids']
                # 临时方案，为了和 hf 对齐
                data_with_system['messages'][2]["content"] = "<think>\n\n</think>\n\n我需要先调用一些工具才能知道\n"
                data_with_system['messages'][-1]["content"] = "<think>\n\n</think>\n\n基于我的观察，今天北京的天气是35度。"

                input_ids = tokenizer_fn(data_with_system)['input_ids']
                prompt_ref = tokenizer.decode(input_ids_ref,skip_special_tokens=False)
                prompt = tokenizer.decode(input_ids,skip_special_tokens=False)
                self.assertEqual(prompt_ref, prompt)
                self.assertEqual(input_ids_ref, input_ids)

                data_wo_system = copy.deepcopy(data)
                del data_wo_system['messages'][0]
                input_ids_ref = tokenizer.apply_chat_template(data_wo_system['messages'], tools=data.get('tools'), tokenize=True, add_generation_prompt=False)['input_ids']
                # 临时方案，为了和 hf 对齐
                data_wo_system['messages'][1]["content"] = "<think>\n\n</think>\n\n我需要先调用一些工具才能知道\n"
                data_wo_system['messages'][-1]["content"] = "<think>\n\n</think>\n\n基于我的观察，今天北京的天气是35度。"

                input_ids = tokenizer_fn(data_wo_system)['input_ids']
                prompt_ref = tokenizer.decode(input_ids_ref,skip_special_tokens=False)
                prompt = tokenizer.decode(input_ids,skip_special_tokens=False)
                self.assertEqual(prompt_ref, prompt)
                self.assertEqual(input_ids_ref, input_ids)

                
    # TODO: Remove this test later
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
