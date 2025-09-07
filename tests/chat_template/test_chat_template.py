from unittest import TestCase
import torch
import torch.nn as nn
from xtuner.v1.loss.ce_loss import CELossConfig, CELossContextInputItem
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.utils.test_utils import assert_verbose_allclose
from xtuner.v1.loss.utils import cal_global_grad_tokens, cal_global_sum_loss_weight, len2weight
from torch.testing._internal.common_distributed import DistributedTestBase
import os
import torch.distributed as dist
from xtuner.v1.data_proto.utils import pad_to_multiple_of, split_for_sequence_parallel
from xtuner.v1.utils.test_utils import init_data_mesh
import parametrize
from functools import wraps
from xtuner.v1.data_proto.templates import CHAT_TEMPLATE_MAP 
from xtuner.v1.data_proto.messages import ChatMessages
from transformers import AutoTokenizer
from datetime import datetime


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
            ("qwen3", 'Qwen/Qwen3-4B'),
            ("qwen3", 'Qwen/Qwen3-4B-Instruct-2507'),
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
            ("gpt-oss", 'openai/gpt-oss-20b'),
        ],
    )
    def test_gpt_oss_template(self, template_type, tokenizer):

        chat_template = CHAT_TEMPLATE_MAP[template_type]
        tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        
        current_date = datetime.now().strftime('%Y-%m-%d')

        messages = {
            "messages": [
                {"role": "system", "content": f"You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: {current_date}\n\nReasoning: medium\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message."},
                {"role": "developer", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi! How can I help you today?"},
                {"role": "user", "content": "Can you tell me a joke?"},
                {"role": "assistant", "content": "Sure!"},
                {"role": "user", "content": "Please!"},
            ]
        }
        
       
        prompt_ref = tokenizer.apply_chat_template(
            messages["messages"][1:],
            tokenize=False,
            add_generation_prompt=True,
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
            ("deepseek-v3", False, 'deepseek-ai/DeepSeek-V3.1'),
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


        