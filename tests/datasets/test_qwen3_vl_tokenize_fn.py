import os
from unittest import TestCase
from xtuner.v1.datasets import Qwen3VLTokenizeFnConfig, PretrainTokenizeFunction
from transformers import AutoTokenizer, AutoProcessor
import json
import torch
import parametrize

QWEN3_VL_PATH = os.environ["QWEN3_VL_MOE_PATH"]
VIDEO_ROOT = os.environ["VIDEO_ROOT"]


class TestMLLMTokenizeFn(TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained(QWEN3_VL_PATH)
        self.tokenize_fn = Qwen3VLTokenizeFnConfig(processor_path=QWEN3_VL_PATH, rand_video_max_frames=14).build(
            self.tokenizer)
        self.processor = AutoProcessor.from_pretrained(QWEN3_VL_PATH)

    def test_qwen3_vl_sft_single_image(self):
        data_path = 'tests/resource/mllm_sft_single_image_example_data.jsonl'
        total_step = 5
        with open(data_path) as f:
            for i, line in enumerate(f):
                if i >= total_step:
                    break
                raw_data = json.loads(line)

                ret = self.tokenize_fn(raw_data, media_root='tests/')
                input_ids_xtuner = ret['input_ids']
                pixel_values_xtuner: torch.Tensor = ret['pixel_values']
                image_grid_thw_xtuner: torch.Tensor = ret['image_grid_thw']

                # to hf openai format
                messages = raw_data['messages']
                messages[0]['content'][0]['type'] = 'image'
                messages[0]['content'][0]['path'] = 'tests/' + messages[0]['content'][0]['image_url']['url']
                del messages[0]['content'][0]['image_url']

                # <IMG_CONTEXT>\n 中的 \n 需要去掉，因为 qwen3 vl chat_template 里面不会加上 \n
                messages[0]['content'][1]['text'] = messages[0]['content'][1]['text'].replace('<IMG_CONTEXT>', '')
                for msg in messages:
                    if not isinstance(msg['content'], list):
                        msg['content'] = [{"type": "text", "text": msg['content']}]

                ret = self.processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=True,
                                                         return_dict=True)
                input_ids_hf = ret['input_ids'][0]
                pixel_values_hf = ret['pixel_values']
                image_grid_thw_hf = ret['image_grid_thw']
                self.assertEqual(input_ids_xtuner, input_ids_hf)
                self.assertTrue(torch.allclose(pixel_values_xtuner, pixel_values_hf))
                self.assertTrue(torch.allclose(image_grid_thw_xtuner, image_grid_thw_hf))

    @parametrize.parametrize("add_vision_id", [(True,), (False,)])
    def test_qwen3_vl_sft_multi_image(self, add_vision_id):
        tokenize_fn = Qwen3VLTokenizeFnConfig(processor_path=QWEN3_VL_PATH,
                                              add_vision_id=add_vision_id).build(self.tokenizer)
        data_path = 'tests/resource/mllm_sft_multi_image_example_data.jsonl'
        total_step = 5
        with open(data_path) as f:
            for i, line in enumerate(f):
                if i >= total_step:
                    break
                raw_data = json.loads(line)

                # \n 必须去掉，否则和 hf 无法对齐
                messages = raw_data['messages']
                messages[0]['content'][2]['text'] = messages[0]['content'][2]['text'].replace('\n', '')

                ret = tokenize_fn(raw_data, media_root='tests/')
                input_ids_xtuner = ret['input_ids']
                pixel_values_xtuner: torch.Tensor = ret['pixel_values']
                image_grid_thw_xtuner: torch.Tensor = ret['image_grid_thw']

                # to hf openai format
                messages = raw_data['messages']
                messages[0]['content'][0]['type'] = 'image'
                messages[0]['content'][0]['path'] = 'tests/' + messages[0]['content'][0]['image_url']['url']
                messages[0]['content'][1]['type'] = 'image'
                messages[0]['content'][1]['path'] = 'tests/' + messages[0]['content'][1]['image_url']['url']
                del messages[0]['content'][0]['image_url']
                del messages[0]['content'][1]['image_url']
                messages[0]['content'][2]['text'] = messages[0]['content'][2]['text'].replace('<IMG_CONTEXT>', '')
                for msg in messages:
                    if not isinstance(msg['content'], list):
                        msg['content'] = [{"type": "text", "text": msg['content']}]

                ret = self.processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=True,
                                                         return_dict=True, add_vision_id=add_vision_id)
                input_ids_hf = ret['input_ids'][0]
                pixel_values_hf = ret['pixel_values']
                image_grid_thw_hf = ret['image_grid_thw']
                self.assertEqual(input_ids_xtuner, input_ids_hf)
                self.assertTrue(torch.allclose(pixel_values_xtuner, pixel_values_hf))
                self.assertTrue(torch.allclose(image_grid_thw_xtuner, image_grid_thw_hf))

    def test_qwen3_vl_sft_pure_text(self):
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

    def test_calc_frame_info(self):
        self.tokenize_fn.state = "cache"
        data_path = 'tests/resource/mllm_video_frame_test_data.jsonl'
        with open(data_path) as f:
            for i, line in enumerate(f):
                raw_data = json.loads(line)
                self.tokenize_fn(raw_data)
                frames_indices_list, origin_fps_list, timestamps_list = self.tokenize_fn.calc_frame_info(raw_data)
                num_frames_list = []
                for frames_indices in frames_indices_list:
                    if isinstance(frames_indices, int):
                        num_frames_list.append(frames_indices)
                    else:
                        num_frames_list.append(len(frames_indices))
                if i == 0:
                    # case: 如果不存在 origin_fps ，则会基于预设的 rand_video_max_frames 参数随机采样
                    assert len(origin_fps_list) == len(timestamps_list) == 0
                    assert self.tokenize_fn.video_processor.min_frames <= num_frames_list[
                        0] <= self.tokenize_fn.rand_video_max_frames
                    assert self.tokenize_fn.video_processor.min_frames <= num_frames_list[
                        1] <= self.tokenize_fn.rand_video_max_frames
                elif i == 1:
                    # case: 如果存在 origin_fps ，则会基于 origin_fps 计算 timestamps
                    assert num_frames_list == [20, 4]
                    assert origin_fps_list == [10, 8]
                    assert timestamps_list == [[0.25, 1.3, 2.35, 3.35, 4.45, 5.45, 6.55, 7.55, 8.600000000000001, 9.65],
                                               [0.25, 1.125]]
                elif i == 2:
                    # case: 测试 origin_fps 为 1 且长度小于 4 时是否正常
                    assert num_frames_list == [20, 4]
                    assert origin_fps_list == [10, 1]
                    assert timestamps_list == [[0.25, 1.3, 2.35, 3.35, 4.45, 5.45, 6.55, 7.55, 8.600000000000001, 9.65],
                                               [0.0, 0.0]]
                elif i == 3:
                    # case: 测试存在 processed_fps 且一个能被 fps 整除，一个不能且视频长度大于 rand_video_max_frames
                    assert num_frames_list == [10, 14]
                    assert origin_fps_list == [20, 10]
                    assert timestamps_list == [[0.25, 1.35, 2.45, 3.55, 4.65],
                                               [0.3, 1.3, 2.4000000000000004, 3.5, 4.6, 5.7, 6.7]]
                elif i == 4:
                    # case: 测试存在 processed_fps 且一个能被 fps 整除，一个不能且视频长度小于 rand_video_max_frames
                    assert num_frames_list == [10, 12]
                    assert origin_fps_list == [20, 10]
                    assert timestamps_list == [[0.25, 1.35, 2.45, 3.55, 4.65],
                                               [0.1, 0.5, 0.9, 1.2999999999999998, 1.7000000000000002, 2.1]]
                elif i == 5:
                    # case: 测试存在 frames_timestamp，且一个能被 fps 整除，一个不能且视频长度小于 rand_video_max_frames
                    assert num_frames_list == [4, 14]
                    assert origin_fps_list == [20, 10]
                    assert timestamps_list == [[0.25, 1.5],
                                               [0.1, 0.5, 1.1, 1.5, 1.9, 2.5, 2.9]]
                elif i == 6:
                    # case: 测试存在 frames_timestamp，且一个能被 fps 整除，一个不能且视频长度小于 rand_video_max_frames
                    assert num_frames_list == [4, 12]
                    assert origin_fps_list == [20, 10]
                    assert timestamps_list == [[0.25, 1.5],
                                               [0.1, 0.5, 0.9, 1.2999999999999998, 1.7000000000000002, 2.1]]
                elif i == 7:
                    # case: 测试单视频
                    assert num_frames_list == [4]
                    assert origin_fps_list == [20]
                    assert timestamps_list == [[0.25, 1.5]]

    def test_qwen3_vl_sft_video(self):
        data_path = 'tests/resource/mllm_sft_video_example_data.jsonl'
        total_index = [1, 4, 5]
        with open(data_path) as f:
            for i, line in enumerate(f):
                if i not in total_index:
                    continue
                raw_data = json.loads(line)

                ret = self.tokenize_fn(raw_data, media_root=VIDEO_ROOT)
                input_ids_xtuner = ret['input_ids']
                pixel_values_xtuner: torch.Tensor = ret['pixel_values']
                image_grid_thw_xtuner: torch.Tensor = ret['image_grid_thw']

                # to hf openai format
                messages = raw_data['messages']
                messages[0]['content'][0]['type'] = 'video'
                messages[0]['content'][0]['path'] = VIDEO_ROOT + messages[0]['content'][0]['video_url']['url']
                messages[0]['content'][1]['text'] = messages[0]['content'][1]['text'].replace('<VIDEO_CONTEXT>', '')
                del messages[0]['content'][0]['video_url']
                for msg in messages:
                    if not isinstance(msg['content'], list):
                        msg['content'] = [{"type": "text", "text": msg['content']}]

                ret = self.processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=True,
                                                         return_dict=True, return_tensors="pt")
                input_ids_hf = ret['input_ids'][0]
                pixel_values_hf = ret['pixel_values_videos']
                image_grid_thw_hf = ret['video_grid_thw']
                if i == 1:
                    # 不应该包括 seconds> 内容
                    text = self.tokenize_fn.tokenizer.decode(input_ids_xtuner)
                    self.assertTrue('seconds>' not in text)
                else:
                    self.assertEqual(input_ids_xtuner, input_ids_hf.tolist())
                    text = self.tokenize_fn.tokenizer.decode(input_ids_xtuner)
                    self.assertTrue('seconds>' in text)
                    self.assertTrue(torch.allclose(pixel_values_xtuner, pixel_values_hf))
                    self.assertTrue(torch.allclose(image_grid_thw_xtuner, image_grid_thw_hf))

    def test_qwen3_vl_pretrain_pure_text(self):
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

    def test_qwen3_vl_pretrain_image(self):
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
                input_str = input_str.replace('<|image_pad|>', '')
                input_xtuner_str = input_str.replace('<|vision_start|><|vision_end|>', '<IMG_CONTEXT>')
                ground_truth_content = raw_data['messages'][0]
                for item in ground_truth_content['content']:
                    if item['type'] == 'text':
                        ground_truth_str = item['text'] + "<|im_end|>"
                image_cnt = ground_truth_str.count('<IMG_CONTEXT>')
                if image_cnt > 1:
                    for i in range(image_cnt):
                        ground_truth_str = ground_truth_str.replace('<IMG_CONTEXT>',
                                                                    f'Picture {i + 1}: <IMG_CONTEXT_1>', 1)
                    ground_truth_str = ground_truth_str.replace('<IMG_CONTEXT_1>', '<IMG_CONTEXT>')
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
