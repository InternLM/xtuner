import os
from unittest import TestCase
import torch
from xtuner.v1.datasets import InternS1VLTokenizeFnConfig, PretrainTokenizeFunction
from transformers import AutoTokenizer, AutoProcessor
import json
from xtuner.v1.model import InternVL3P5Dense8BConfig

INTERN_VL_1B_PATH = os.environ["INTERN_VL_1B_PATH"]
VIDEO_ROOT = os.environ["VIDEO_ROOT"]


class TestMLLMTokenizeFn(TestCase):
    def setUp(self):
        model_cfg = InternVL3P5Dense8BConfig()
        tokenizer = AutoTokenizer.from_pretrained(INTERN_VL_1B_PATH, trust_remote_code=True)
        self.tokenizer = tokenizer
        self.tokenize_fn = InternS1VLTokenizeFnConfig(model_cfg=model_cfg,
                                                      template_name='internvl-3.5').build(tokenizer)
        self.tokenize_fn.chat_template.default_system = None
        self.processor = AutoProcessor.from_pretrained(INTERN_VL_1B_PATH, trust_remote_code=True)

    def test_intern_vl_sft_single_image(self):
        data_path = 'tests/resource/mllm_sft_single_image_example_data.jsonl'
        total_step = 5
        with open(data_path, encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= total_step:
                    break
                raw_data = json.loads(line)

                ret = self.tokenize_fn(raw_data, media_root='tests/')
                input_ids_xtuner = ret['input_ids']
                pixel_values_xtuner: torch.Tensor = ret['pixel_values']

                # to hf openai format
                messages = raw_data['messages']
                messages[0]['content'][0]['type'] = 'image'
                messages[0]['content'][0]['path'] = 'tests/' + messages[0]['content'][0]['image_url']['url']
                del messages[0]['content'][0]['image_url']
                # 需要把 \n 去掉，因为 internvl chat_template 里面会加上 \n
                messages[0]['content'][1]['text'] = messages[0]['content'][1]['text'].replace('<IMG_CONTEXT>\n', '')
                for msg in messages:
                    if not isinstance(msg['content'], list):
                        msg['content'] = [{"type": "text", "text": msg['content']}]

                ret = self.processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=True,
                                                         return_dict=True)
                input_ids_hf = ret['input_ids'][0]
                pixel_values_hf = torch.stack(ret['pixel_values'])
                self.assertTrue(input_ids_xtuner, input_ids_hf)
                self.assertTrue(pixel_values_xtuner.shape, pixel_values_hf.shape)

    def test_intern_vl_sft_multi_image(self):
        for data_path in ['tests/resource/mllm_sft_multi_image_example_data.jsonl',
                          'tests/resource/mllm_sft_multi_image_example_data2.jsonl']:
            total_step = 5
            # input_ids 天然就对不齐,因为 hf 里面实现的是错误的。当多图时候，max_num 应该要缩小，但是 hf 里面是独立处理，导致 <IMG_CONTEXT> 数目必然对不上
            with open(data_path, encoding='utf-8') as f:
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

                    # 处理所有消息中的图片内容
                    for msg in messages:
                        if isinstance(msg['content'], list):
                            # 处理包含图片的消息
                            for content_item in msg['content']:
                                if isinstance(content_item, dict) and 'image_url' in content_item:
                                    content_item['type'] = 'image'
                                    content_item['path'] = 'tests/' + content_item['image_url']['url']
                                    del content_item['image_url']
                                elif isinstance(content_item, dict) and 'text' in content_item:
                                    # 处理文本内容中的 <IMG_CONTEXT>
                                    content_item['text'] = content_item['text'].replace('<IMG_CONTEXT>\n', '')
                        else:
                            # 处理纯文本消息
                            msg['content'] = [{"type": "text", "text": msg['content']}]

                    input_hf_str = self.processor.apply_chat_template(messages, add_generation_prompt=False,
                                                                      tokenize=False, return_dict=True)
                    self.assertEqual(input_xtuner_str, input_hf_str)

    def test_intern_vl_sft_video(self):
        data_path = 'tests/resource/mllm_sft_video_example_data.jsonl'
        with open(data_path, encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 1:
                    break
                raw_data = json.loads(line)

                ret = self.tokenize_fn(raw_data, media_root=VIDEO_ROOT)
                input_ids_xtuner = ret['input_ids']

                input_str = self.tokenize_fn.tokenizer.decode(input_ids_xtuner, skip_special_tokens=False)
                input_str = input_str.replace('<IMG_CONTEXT>', '')
                input_str = input_str.replace('<img></img>', '<IMG_CONTEXT>')
                expected_str = "<|im_start|>user\nFrame-1: <IMG_CONTEXT>\nFrame-2: <IMG_CONTEXT>\nFrame-3: " \
                               "<IMG_CONTEXT>\nFrame-4: <IMG_CONTEXT>\nFrame-5: <IMG_CONTEXT>\nFrame-6: " \
                               "<IMG_CONTEXT>\nFrame-7: <IMG_CONTEXT>\nFrame-8: <IMG_CONTEXT>\nFrame-9: " \
                               "<IMG_CONTEXT>\nFrame-10: <IMG_CONTEXT>\nFrame-11: <IMG_CONTEXT>\nFrame-12: " \
                               "<IMG_CONTEXT>\nFrame-13: " \
                               "<IMG_CONTEXT>\n请描述下视频内容？<|im_end|>\n<|im_start|>assistant\n一男一女在打网球<|im_end|>\n" \
                               "<|im_start|>user\n请简要解释下网球<|im_end|>\n<|im_start|>assistant\n" \
                               "网球是一项运动，运动员使用球拍将球击打过网进入对方场地。目标是通过让球落入对方场地且对方无法回击来得分。网球可以单人对战（单打）或双人组队对战（双打）。<|im_end|>"
                assert input_str.strip() == expected_str.strip()

    def test_intern_vl_sft_pure_text(self):
        data_path = 'tests/resource/mllm_sft_text_example_data.jsonl'
        total_step = 5
        with open(data_path, encoding='utf-8') as f:
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

    def test_intern_vl_pretrain_pure_text(self):
        data_path = 'tests/resource/pretrain_example_data.jsonl'
        tokenize_fn = PretrainTokenizeFunction(self.tokenizer)
        total_step = 5
        with open(data_path, encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= total_step:
                    break
                raw_data = json.loads(line)

                ret = tokenize_fn(raw_data)
                input_ids_xtuner = ret['input_ids'][:-1]  # remove eos_token_id

                content = raw_data['messages'][0]['content']
                input_ids_hf = self.tokenizer(content)['input_ids']
                self.assertEqual(input_ids_xtuner, input_ids_hf)

    def test_intern_vl_pretrain_image(self):
        data_path = 'tests/resource/mllm_pretrain_image_example_data.jsonl'
        total_step = 6
        with open(data_path, encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= total_step:
                    break
                raw_data = json.loads(line)

                ret = self.tokenize_fn(raw_data, media_root='tests/')
                input_ids_xtuner = ret['input_ids']
                labels_xtuner = torch.tensor(ret['labels'])
                input_str = self.tokenize_fn.tokenizer.decode(input_ids_xtuner, skip_special_tokens=False)
                input_str = input_str.replace('<IMG_CONTEXT>', '')
                input_xtuner_str = input_str.replace('<img></img>', '<IMG_CONTEXT>')
                ground_truth_content = raw_data['messages'][0]
                for item in ground_truth_content['content']:
                    if item['type'] == 'text':
                        ground_truth_str = item['text'] + "<|im_end|>"
                self.assertEqual(input_xtuner_str.strip(), ground_truth_str.strip())
                self.assertTrue((labels_xtuner == self.tokenize_fn.img_context_token_id).sum() == 0)

    def test_intern_vl_pretrain_video(self):
        data_path = 'tests/resource/mllm_pretrain_video_example_data.jsonl'
        total_step = 6
        with open(data_path, encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= total_step:
                    break
                raw_data = json.loads(line)

                ret = self.tokenize_fn(raw_data, media_root=VIDEO_ROOT)
                labels_xtuner = torch.tensor(ret['labels'])
                self.assertTrue((labels_xtuner == self.tokenize_fn.video_context_token_id).sum() == 0)

