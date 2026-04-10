from datetime import datetime
import os
import json
import parametrize
from unittest import TestCase
from transformers import AutoTokenizer
import torch
from packaging.version import Version
from transformers import __version__ as transformers_version
import unittest

from xtuner.v1.data_proto.templates import CHAT_TEMPLATE_MAP 
from xtuner.v1.data_proto.messages import ChatMessages
from xtuner.v1.data_proto.messages.qwen35_chat import Qwen35ChatMessages, qwen35_tokenize_fn_slowspeed


QWEN3_PATH = os.environ["QWEN3_PATH"]
QWEN3_VL_DENSE_PATH = os.environ["QWEN3_VL_DENSE_PATH"]
GPT_OSS_MINI_PATH = os.environ["GPT_OSS_MINI_PATH"]
DEEPSEEK_V3_PATH = os.environ["DEEPSEEK_V3_PATH"]

class TestChatTemplate(TestCase):

    def setUp(self):
        

        self.chatml_messages = {
            "messages": [
                {"role": "system", "content": "Youer are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi! How can I help you today?"},
                {"role": "user", "content": "Can you tell me a joke?"},
                {"role": "assistant", "content": "Sure!"},
                {"role": "user", "content": "Please!"},
            ]
        }


    @parametrize.parametrize(
        "template_type, tokenizer",
        [   
            ("qwen3", QWEN3_PATH),
        ],
    )
    def test_qwen3_template(self, template_type, tokenizer):

        chat_template = CHAT_TEMPLATE_MAP[template_type]
        tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        
        messages = self.chatml_messages 
        
        prompt_ref = tokenizer.apply_chat_template(
            messages["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )  

        _messages = ChatMessages(**messages)
        prompt = _messages.get_prompt(chat_template)
    
        prompt_ref = prompt_ref.replace("<think>\n\n</think>\n\n", "")
     
        self.assertEqual(prompt, prompt_ref)

        input_ids_ref = tokenizer.encode(prompt_ref, add_special_tokens=False)

        input_ids = _messages.tokenize(tokenizer, chat_template)['input_ids']

        self.assertTrue((input_ids == input_ids_ref))

    @parametrize.parametrize(
        "template_type, tokenizer",
        [
            ("qwen3-vl", QWEN3_VL_DENSE_PATH),
        ],
    )
    def test_qwen3vl_tool_template(self, template_type, tokenizer):
        chat_template = CHAT_TEMPLATE_MAP[template_type]
        tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)

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
        prompt_ref = tokenizer.apply_chat_template(
            messages["messages"],
            tools=messages["tools"],
            tokenize=False,
            add_generation_prompt=False,
        )

        _messages = ChatMessages(**messages)
        prompt = _messages.get_prompt(chat_template)
        self.assertEqual(prompt, prompt_ref)

        _messages = ChatMessages(**messages)
        input_ids_ref = tokenizer.encode(prompt_ref, add_special_tokens=False)
        input_ids = _messages.tokenize(tokenizer, chat_template)['input_ids']
        self.assertTrue((input_ids == input_ids_ref))

    @parametrize.parametrize(
        "template_type, tokenizer",
        [   
            ("gpt-oss", GPT_OSS_MINI_PATH),
        ],
    )
    def test_gpt_oss_template(self, template_type, tokenizer):

        chat_template = CHAT_TEMPLATE_MAP[template_type]
        tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)

        messages_simple = {
            "messages": [
                {"role": "developer", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi! How can I help you today?"},
                {"role": "user", "content": "Can you tell me a joke?"},
                {"role": "assistant", "content": "Sure!"},
            ]
        }
        messages_thinking = {
            "messages": [
                {"role": "developer", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi! How can I help you today?"},
                {"role": "user", "content": "Can you tell me a joke?"},
                {"role": "assistant", "thinking": "Thinking real hard...", "content": "Okay!"},
                {"role": "user", "content": "Please!"},
                {"role": "assistant", "thinking": "Thinking real hard...Thinking real hard...", "content": "Sure!"},
            ]
        }

        for messages in [messages_simple, messages_thinking]:
            prompt_ref = tokenizer.apply_chat_template(
                messages["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )

            _messages = ChatMessages(**messages)
            prompt = _messages.get_prompt(chat_template)

            self.assertEqual(prompt, prompt_ref)

            input_ids_ref = tokenizer.encode(prompt_ref, add_special_tokens=False)

            input_ids = _messages.tokenize(tokenizer, chat_template)['input_ids']
            self.assertTrue((input_ids == input_ids_ref))

    @parametrize.parametrize(
        "template_type, thinking, tokenizer",
        [   
            ("deepseek-v3", False, DEEPSEEK_V3_PATH),
        ],
    )
    def test_deepseek_v3_template(self, template_type,thinking, tokenizer):

        chat_template = CHAT_TEMPLATE_MAP[template_type]
        tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        
        messages = self.chatml_messages 
        
        prompt_ref = tokenizer.apply_chat_template(
            messages["messages"],
            tokenize=False,
            thinking=thinking,
            add_generation_prompt=True,
        )  
    
        _messages = ChatMessages(**messages)
        prompt = _messages.get_prompt(chat_template)
     
        self.assertEqual(prompt, prompt_ref)

        input_ids_ref = tokenizer.encode(prompt_ref, add_special_tokens=False)

        input_ids = _messages.tokenize(tokenizer, chat_template)['input_ids']
        
        self.assertTrue((input_ids == input_ids_ref))
    

    @unittest.skipIf(
        Version(transformers_version) < Version("5.2.0"),
        f"transformers >= 5.2.0 is required, but got {transformers_version}"
    )
    def test_qwen35vl_template(self):
        QWEN35_VL_PATH = os.environ["QWEN3_5_MOE_PATH"]
        chat_template = CHAT_TEMPLATE_MAP["qwen3.5-vl"]
        tokenizer = AutoTokenizer.from_pretrained(QWEN35_VL_PATH, trust_remote_code=True)
        
        jsonl_path = 'tests/resource/qwen35_tokenize_data.jsonl'
        all_data= []
        with open(jsonl_path, 'r') as f:
            for line in f:
                all_data.append(json.loads(line))
        
        for j, data in enumerate(all_data):
            if j in [13,14]: # video 肯定和 hf 对不上
                continue
            gt_token_ids, gt_labels = qwen35_tokenize_fn_slowspeed(tokenizer, data['messages'], tools=data.get('tools'), add_vision_id=True)
            _messages = Qwen35ChatMessages(messages=data["messages"], tools=data.get("tools"))
            tokenized = _messages.tokenize(tokenizer, chat_template, add_vision_id=True)
            decode_str = tokenizer.decode(tokenized['input_ids'], skip_special_tokens=False)

            if j!=15 and j!=16:
                self.assertEqual(tokenized['input_ids'], gt_token_ids)
                self.assertEqual(tokenized['labels'], gt_labels)

                enable_thinking = any("reasoning_content" in msg for msg in data['messages'])
                hf_text = tokenizer.apply_chat_template(data['messages'],   
                                                tools=data.get('tools'),       
                                                add_vision_id=True,   
                                                tokenize=False,
                                                enable_thinking=enable_thinking,
                                                add_generation_prompt=False)
                self.assertEqual(decode_str, hf_text)
            else:
                if j==15:
                    self.assertTrue('Video 1: <|vision_start|><|video_pad|><|vision_end|><|vision_start|><|video_pad|><|vision_end|><|vision_start|><|video_pad|><|vision_end|><0.0-10.0 seconds>Describe the video in detail. [NO_REASONING]<|im_end|>' in decode_str)
                else:
                    self.assertTrue('Video 1: <0.0 seconds><|vision_start|><|video_pad|><|vision_end|><1.0 seconds><|vision_start|><|video_pad|><|vision_end|><2.0 seconds><|vision_start|><|video_pad|><|vision_end|><0.0-10.0 seconds>Describe the video in detail. [NO_REASONING]<|im_end|>' in decode_str)
