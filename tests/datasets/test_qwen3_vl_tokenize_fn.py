import os
from unittest import TestCase
from xtuner.v1.datasets import Qwen3VLTokenizeFnConfig, PretrainTokenizeFunction
from transformers import AutoTokenizer, AutoProcessor
import json
import torch
import parametrize
from xtuner.v1.utils.test_utils import add_video_root

QWEN3_VL_PATH = os.environ["QWEN3_VL_MOE_PATH"]
VIDEO_ROOT = os.environ["VIDEO_ROOT"]


class TestMLLMTokenizeFn(TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained(QWEN3_VL_PATH)
        self.tokenize_fn = Qwen3VLTokenizeFnConfig(processor_path=QWEN3_VL_PATH,
                                                   rand_video_max_frames=14,
                                                   add_vision_id=False).build(
            self.tokenizer)
        self.processor = AutoProcessor.from_pretrained(QWEN3_VL_PATH)

    def test_qwen3vl_tool_template(self):
        tokenize_fn = Qwen3VLTokenizeFnConfig(processor_path=QWEN3_VL_PATH).build(self.tokenizer)

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
        input_ids_ref = self.tokenizer.apply_chat_template(
            messages["messages"],
            tools=messages["tools"],
            tokenize=True,
            add_generation_prompt=False,
        )
        input_ids = tokenize_fn(messages)['input_ids']
        self.assertEqual(input_ids, input_ids_ref)

    @parametrize.parametrize("add_vision_id", [(True,), (False,)])
    def test_qwen3_vl_sft_single_image(self, add_vision_id):
        tokenize_fn = Qwen3VLTokenizeFnConfig(processor_path=QWEN3_VL_PATH,
                                              add_vision_id=add_vision_id).build(self.tokenizer)
        data_path = 'tests/resource/mllm_sft_single_image_example_data.jsonl'
        total_step = 5
        with open(data_path) as f:
            for i, line in enumerate(f):
                if i >= total_step:
                    break
                raw_data = json.loads(line)

                ret = tokenize_fn(raw_data, media_root='tests/')
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

                ret = self.processor.apply_chat_template(messages,
                                                         add_generation_prompt=False,
                                                         tokenize=True,
                                                         add_vision_id=add_vision_id,
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
        total_index = [0, 1, 2, 3, 4, 10]
        with open(data_path) as f:
            for i, line in enumerate(f):
                if i not in total_index:
                    continue
                raw_data = json.loads(line)

                # \n 必须去掉，否则和 hf 无法对齐
                messages = raw_data['messages']
                if i != 10:
                    messages[0]['content'][2]['text'] = messages[0]['content'][2]['text'].replace('\n', '')
                else:
                    messages[0]['content'][1]['text'] = messages[0]['content'][1]['text'].replace('\n', '')
                    messages[4]['content'][1]['text'] = messages[4]['content'][1]['text'].replace('\n', '')

                ret = tokenize_fn(raw_data, media_root='tests/')
                input_ids_xtuner = ret['input_ids']
                pixel_values_xtuner: torch.Tensor = ret['pixel_values']
                image_grid_thw_xtuner: torch.Tensor = ret['image_grid_thw']

                # to hf openai format
                messages = raw_data['messages']
                if i != 10:
                    messages[0]['content'][0]['type'] = 'image'
                    messages[0]['content'][0]['path'] = 'tests/' + messages[0]['content'][0]['image_url']['url']
                    messages[0]['content'][1]['type'] = 'image'
                    messages[0]['content'][1]['path'] = 'tests/' + messages[0]['content'][1]['image_url']['url']
                    del messages[0]['content'][0]['image_url']
                    del messages[0]['content'][1]['image_url']
                    messages[0]['content'][2]['text'] = messages[0]['content'][2]['text'].replace('<IMG_CONTEXT>', '')
                else:
                    messages[0]['content'][0]['type'] = 'image'
                    messages[0]['content'][0]['path'] = 'tests/' + messages[0]['content'][0]['image_url']['url']
                    del messages[0]['content'][0]['image_url']
                    messages[0]['content'][1]['text'] = messages[0]['content'][1]['text'].replace('<IMG_CONTEXT>', '')

                    messages[4]['content'][0]['type'] = 'image'
                    messages[4]['content'][0]['path'] = 'tests/' + messages[4]['content'][0]['image_url']['url']
                    del messages[4]['content'][0]['image_url']
                    messages[4]['content'][1]['text'] = messages[4]['content'][1]['text'].replace('<IMG_CONTEXT>', '')

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
                self.assertEqual(input_ids_xtuner, input_ids_hf)

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
                    self.assertEqual(num_frames_list, [20, 4])
                    self.assertEqual(origin_fps_list, [10, 8])
                    self.assertEqual(timestamps_list,
                                     [[0.25, 1.3, 2.35, 3.35, 4.45, 5.45, 6.55, 7.55, 8.600000000000001, 9.65],
                                      [0.25, 1.125]])
                elif i == 2:
                    # case: 测试 origin_fps 为 1 且长度小于 4 时是否正常
                    self.assertEqual(num_frames_list, [20, 4])
                    self.assertEqual(origin_fps_list, [10, 1])
                    self.assertEqual(timestamps_list,
                                     [[0.25, 1.3, 2.35, 3.35, 4.45, 5.45, 6.55, 7.55, 8.600000000000001, 9.65],
                                      [0.0, 0.0]])
                elif i == 3:
                    # case: 测试存在 processed_fps 且一个能被 fps 整除，一个不能且视频长度大于 rand_video_max_frames
                    self.assertEqual(num_frames_list, [10, 14])
                    self.assertEqual(origin_fps_list, [20, 10])
                    self.assertEqual(timestamps_list, [[0.25, 1.35, 2.45, 3.55, 4.65],
                                                       [0.3, 1.3, 2.4000000000000004, 3.5, 4.6, 5.7, 6.7]])
                elif i == 4:
                    # case: 测试存在 processed_fps 且一个能被 fps 整除，一个不能且视频长度小于 rand_video_max_frames
                    self.assertEqual(num_frames_list, [10, 12])
                    self.assertEqual(origin_fps_list, [20, 10])
                    self.assertEqual(timestamps_list, [[0.25, 1.35, 2.45, 3.55, 4.65],
                                                       [0.1, 0.5, 0.9, 1.2999999999999998, 1.7000000000000002, 2.1]])
                elif i == 5:
                    # case: 测试存在 frames_timestamp，且一个能被 fps 整除，一个不能且视频长度小于 rand_video_max_frames
                    self.assertEqual(num_frames_list, [4, 14])
                    self.assertEqual(origin_fps_list, [20, 10])
                    self.assertEqual(timestamps_list, [[0.25, 1.5],
                                                       [0.1, 0.5, 1.1, 1.5, 1.9, 2.5, 2.9]])
                elif i == 6:
                    # case: 测试存在 frames_timestamp，且一个能被 fps 整除，一个不能且视频长度小于 rand_video_max_frames
                    self.assertEqual(num_frames_list, [4, 12])
                    self.assertEqual(origin_fps_list, [20, 10])
                    self.assertEqual(timestamps_list, [[0.25, 1.5],
                                                       [0.1, 0.5, 0.9, 1.2999999999999998, 1.7000000000000002, 2.1]])
                elif i == 7:
                    # case: 测试单视频
                    self.assertEqual(num_frames_list, [4])
                    self.assertEqual(origin_fps_list, [20])
                    self.assertEqual(timestamps_list, [[0.25, 1.5]])

    @parametrize.parametrize("add_vision_id", [(True,), (False,)])
    def test_qwen3_vl_sft_video(self, add_vision_id):
        tokenize_fn = Qwen3VLTokenizeFnConfig(processor_path=QWEN3_VL_PATH, rand_video_max_frames=14,
                                              add_vision_id=add_vision_id).build(
            self.tokenizer)
        data_path = 'tests/resource/mllm_sft_video_example_data.jsonl'
        hf_data_path = 'tests/resource/mllm_sft_video_hf_example_data.jsonl'
        hf_raw_datas = []
        with open(hf_data_path) as f:
            for line in f:
                hf_raw_datas.append(json.loads(line))

        total_index = [1, 4, 5, 6, 7, 8, 9]
        with open(data_path) as f:
            for i, line in enumerate(f):
                if i not in total_index:
                    continue
                raw_data = json.loads(line)
                hf_raw_data = hf_raw_datas[i]

                if i in [7]:
                    # transformers 当输入视频文件夹时候，无法支持采样，有多少视频就读多少视频
                    do_sample_frames = False
                    tokenize_fn.video_processor.fps = 3
                    tokenize_fn.rand_video_max_frames = 24  # 设置为大于采样后视频，防止进行采样
                else:
                    do_sample_frames = True
                    tokenize_fn.video_processor.fps = 2
                    tokenize_fn.rand_video_max_frames = 14

                ret = tokenize_fn(raw_data, media_root=VIDEO_ROOT)
                input_ids_xtuner = ret['input_ids']
                pixel_values_xtuner: torch.Tensor = ret['pixel_values']
                image_grid_thw_xtuner: torch.Tensor = ret['image_grid_thw']

                # to hf openai format
                messages = hf_raw_data['messages']
                add_video_root(messages, VIDEO_ROOT)

                if i not in [8, 9]:
                    ret = self.processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=True,
                                                             do_sample_frames=do_sample_frames,
                                                             return_dict=True, add_vision_id=add_vision_id,
                                                             return_tensors="pt")
                    input_ids_hf = ret['input_ids'][0]
                    pixel_values_hf = ret['pixel_values_videos']
                    image_grid_thw_hf = ret['video_grid_thw']

                text = self.tokenize_fn.tokenizer.decode(input_ids_xtuner)

                if i == 1:
                    # 不应该包括 seconds> 内容
                    self.assertTrue('seconds>' not in text)
                else:
                    if i == 8:
                        # 测试能整除下均匀采样
                        self.assertEqual(pixel_values_xtuner.size(), (45760, 1536))
                        self.assertEqual(text.count('seconds>'), 13)
                    elif i == 9:
                        # 测试无法整除且超过最大帧数情况下，均匀采样
                        self.assertEqual(pixel_values_xtuner.size(), (24640, 1536))
                        self.assertEqual(text.count('seconds>'), 7)
                        print(pixel_values_xtuner.size(), image_grid_thw_xtuner, text.count('seconds>'), 'xxx')
                    else:
                        if i == 7:
                            self.assertEqual(len(input_ids_xtuner), len(input_ids_hf))
                        else:
                            self.assertEqual(input_ids_xtuner, input_ids_hf.tolist())
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

    @parametrize.parametrize("add_vision_id", [(True,), (False,)])
    def test_qwen3_vl_pretrain_image(self, add_vision_id):
        tokenize_fn = Qwen3VLTokenizeFnConfig(processor_path=QWEN3_VL_PATH,
                                              add_vision_id=add_vision_id).build(self.tokenizer)
        data_path = 'tests/resource/mllm_pretrain_image_example_data.jsonl'
        total_step = 6
        with open(data_path, encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= total_step:
                    break
                raw_data = json.loads(line)

                ret = tokenize_fn(raw_data, media_root='tests/')
                input_ids_xtuner = ret['input_ids']
                labels_xtuner = torch.tensor(ret['labels'])
                input_str = tokenize_fn.tokenizer.decode(input_ids_xtuner, skip_special_tokens=False)
                input_str = input_str.replace('<|image_pad|>', '')
                input_xtuner_str = input_str.replace('<|vision_start|><|vision_end|>', '<IMG_CONTEXT>')
                ground_truth_content = raw_data['messages'][0]
                for item in ground_truth_content['content']:
                    if item['type'] == 'text':
                        ground_truth_str = item['text'] + "<|im_end|>"
                image_cnt = ground_truth_str.count('<IMG_CONTEXT>')
                if add_vision_id:
                    for i in range(image_cnt):
                        ground_truth_str = ground_truth_str.replace('<IMG_CONTEXT>',
                                                                    f'Picture {i + 1}: <IMG_CONTEXT_1>', 1)
                    ground_truth_str = ground_truth_str.replace('<IMG_CONTEXT_1>', '<IMG_CONTEXT>')
                self.assertEqual(input_xtuner_str.strip(), ground_truth_str.strip())
                self.assertTrue((labels_xtuner == self.tokenize_fn.img_context_token_id).sum() == 0)

    def test_qwen3_vl_pretrain_video(self):
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
