import os
from unittest import TestCase

from xtuner.v1.datasets import InternS1VLTokenizeFnConfig
from transformers import AutoTokenizer, AutoProcessor
import json
from xtuner.v1.model import InternVL3P5Dense8BConfig

INTERN_VL_1B_PATH = os.environ["INTERN_VL_1B_PATH"]


class TestMLLMTokenizeFn(TestCase):
    def setUp(self):
        model_cfg = InternVL3P5Dense8BConfig()
        tokenizer = AutoTokenizer.from_pretrained(INTERN_VL_1B_PATH, trust_remote_code=True)
        self.tokenize_fn = InternS1VLTokenizeFnConfig(model_cfg=model_cfg).build(tokenizer)
        self.processor = AutoProcessor.from_pretrained(INTERN_VL_1B_PATH, trust_remote_code=True)

    def test_intern_vl_single_image(self):
        data_path = 'tests/resource/mllm_sft_single_image_example_data.jsonl'
        total_step = 5
        with open(data_path) as f:
            for i, line in enumerate(f):
                if i >= total_step:
                    break
                raw_data = json.loads(line)

                ret = self.tokenize_fn(raw_data, media_root='tests/')
                input_ids_xtuner = ret['input_ids']

                # to hf openai format
                messages = raw_data['messages']
                messages[0]['content'][0]['type'] = 'image'
                messages[0]['content'][0]['path'] = 'tests/' + messages[0]['content'][0]['image_url']
                del messages[0]['content'][0]['image_url']
                messages[0]['content'][1]['text'] = messages[0]['content'][1]['text'].replace('<IMG_CONTEXT>\n', '')
                for msg in messages:
                    if not isinstance(msg['content'], list):
                        msg['content'] = [{"type": "text", "text": msg['content']}]

                ret = self.processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=True,
                                                         return_dict=True)
                input_ids_hf = ret['input_ids'][0]
                assert input_ids_xtuner == input_ids_hf

    def test_intern_vl_multi_image(self):
        data_path = 'tests/resource/mllm_sft_multi_image_example_data.jsonl'
        total_step = 5
        # input_ids 天然就对不齐,因为 hf 里面实现的是错误的。当多图时候，max_num 应该要缩小，但是 hf 里面是独立处理，导致 <IMG_CONTEXT> 数目必然对不上
        with open(data_path) as f:
            for i, line in enumerate(f):
                if i >= total_step:
                    break
                raw_data = json.loads(line)

                ret = self.tokenize_fn(raw_data, media_root='tests/')
                input_ids_xtuner = ret['input_ids']
                input_str = self.tokenize_fn.tokenizer.decode(input_ids_xtuner, skip_special_tokens=False)
                input_str = input_str.replace('<IMG_CONTEXT>', '')
                input_xtuner_str = input_str.replace('<img></img>', '<IMG_CONTEXT>')

                # to hf openai format
                messages = raw_data['messages']
                messages[0]['content'][0]['type'] = 'image'
                messages[0]['content'][0]['path'] = 'tests/' + messages[0]['content'][0]['image_url']
                messages[0]['content'][1]['type'] = 'image'
                messages[0]['content'][1]['path'] = 'tests/' + messages[0]['content'][1]['image_url']
                del messages[0]['content'][0]['image_url']
                del messages[0]['content'][1]['image_url']
                messages[0]['content'][2]['text'] = messages[0]['content'][2]['text'].replace('<IMG_CONTEXT>\n', '')
                for msg in messages:
                    if not isinstance(msg['content'], list):
                        msg['content'] = [{"type": "text", "text": msg['content']}]

                input_hf_str = self.processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False, return_dict=True)
                self.assertEqual(input_xtuner_str, input_hf_str)

    def test_intern_vl_pure_text(self):
        data_path = 'tests/resource/mllm_sft_text_example_data.jsonl'
        total_step = 5
        with open(data_path) as f:
            for i, line in enumerate(f):
                if i >= total_step:
                    break
                raw_data = json.loads(line)

                ret = self.tokenize_fn(raw_data)
                input_ids_xtuner = ret['input_ids']

                # to hf openai format
                messages = raw_data['messages']
                for msg in messages:
                    if not isinstance(msg['content'], list):
                        msg['content'] = [{"type": "text", "text": msg['content']}]

                ret = self.processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=True,
                                                         return_dict=True)
                input_ids_hf = ret['input_ids'][0]
                assert input_ids_xtuner == input_ids_hf
