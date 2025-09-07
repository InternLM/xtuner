from datetime import datetime
import os
import parametrize
from unittest import TestCase
from transformers import AutoTokenizer

from xtuner.v1.data_proto.templates import CHAT_TEMPLATE_MAP 
from xtuner.v1.data_proto.messages import ChatMessages


QWEN3_PATH = os.environ["QWEN3_PATH"]
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
            ("gpt-oss", GPT_OSS_MINI_PATH),
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


        