import os
from unittest import TestCase
from xtuner.v1.datasets import Qwen3VLTokenizeFnConfig, PretrainTokenizeFunction
from transformers import AutoTokenizer, AutoProcessor,Qwen3VLProcessor
import json
import torch
import parametrize
from xtuner.v1.utils.test_utils import add_video_root
from packaging.version import Version
from transformers import __version__ as transformers_version
import unittest
from xtuner.v1.data_proto.messages.qwen35_chat import qwen35_tokenize_fn_slowspeed

VIDEO_ROOT = os.environ["VIDEO_ROOT"]


@unittest.skipIf(
    Version(transformers_version) < Version("5.2.0"),
    f"transformers >= 5.2.0 is required, but got {transformers_version}"
)
class TestMLLMTokenizeFn(TestCase):
    def setUp(self):
        QWEN35_VL_PATH = os.environ["QWEN3_5_MOE_PATH"]
        self.tokenizer = AutoTokenizer.from_pretrained(QWEN35_VL_PATH)
        self.tokenize_fn = Qwen3VLTokenizeFnConfig(processor_path=QWEN35_VL_PATH, 
                                                   chat_template="qwen3.5-vl",
                                                   rand_video_max_frames=14,
                                                   add_vision_id=False).build(
            self.tokenizer)
        self.processor = AutoProcessor.from_pretrained(QWEN35_VL_PATH)

    def test_qwen35vl_text(self):
        QWEN35_VL_PATH = os.environ["QWEN3_5_MOE_PATH"]
        tokenize_fn = Qwen3VLTokenizeFnConfig(processor_path=QWEN35_VL_PATH, chat_template="qwen3.5-vl", add_vision_id=True).build(self.tokenizer)

        data_path = 'tests/resource/qwen35_tokenize_data.jsonl'
        all_data= []
        with open(data_path, 'r') as f:
            for line in f:
                all_data.append(json.loads(line))
        
        for j, data in enumerate(all_data):
            if j>=12:
                break
            gt_token_ids, gt_labels = qwen35_tokenize_fn_slowspeed(self.tokenizer, data['messages'], tools=data.get('tools'), add_vision_id=True)
            ret = tokenize_fn(data)
            input_ids_xtuner = ret['input_ids']
            labels_xtuner = ret['labels']
            self.assertEqual(input_ids_xtuner, gt_token_ids)
            self.assertEqual(labels_xtuner, gt_labels)

            enable_thinking = any("reasoning_content" in msg for msg in data['messages'])
            decode_str = self.tokenizer.decode(input_ids_xtuner, skip_special_tokens=False)
            hf_text = self.tokenizer.apply_chat_template(data['messages'],   
                                               tools=data.get('tools'),       
                                               add_vision_id=True,   
                                               tokenize=False,
                                               enable_thinking=enable_thinking,
                                               add_generation_prompt=False)
            self.assertEqual(decode_str, hf_text)

    @parametrize.parametrize("add_vision_id", [(True,), (False,)])
    def test_qwen35_vl_sft_single_image(self, add_vision_id):
        QWEN35_VL_PATH = os.environ["QWEN3_5_MOE_PATH"]
        tokenize_fn = Qwen3VLTokenizeFnConfig(processor_path=QWEN35_VL_PATH, chat_template="qwen3.5-vl",
                                              add_vision_id=add_vision_id).build(self.tokenizer)
        data_path = 'tests/resource/mllm_sft_single_image_example_data_new.jsonl'
        total_step = 50
        with open(data_path) as f:
            for i, line in enumerate(f):
                if i >=total_step:
                    break
                raw_data = json.loads(line)
                
                ret = tokenize_fn(raw_data, media_root='tests/')
                input_ids_xtuner = ret['input_ids']
                pixel_values_xtuner: torch.Tensor = ret['pixel_values']
                image_grid_thw_xtuner: torch.Tensor = ret['image_grid_thw']

                # to hf openai format
                messages = raw_data['messages']
                messages[0]['content'][0]['type'] = 'image'
                messages[0]['content'][0]['path'] = 'tests/' + messages[0]['content'][0]['image']['url']
                del messages[0]['content'][0]['image']

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
        QWEN35_VL_PATH = os.environ["QWEN3_5_MOE_PATH"]
        tokenize_fn = Qwen3VLTokenizeFnConfig(processor_path=QWEN35_VL_PATH,
                                              chat_template="qwen3.5-vl",
                                              add_vision_id=add_vision_id).build(self.tokenizer)
        data_path = 'tests/resource/mllm_sft_multi_image_example_data_new.jsonl'
        total_index = [0, 1, 2, 3, 4, 10]
        with open(data_path) as f:
            for i, line in enumerate(f):
                if i not in total_index:
                    continue
                raw_data = json.loads(line)

                ret = tokenize_fn(raw_data, media_root='tests/')
                input_ids_xtuner = ret['input_ids']
                pixel_values_xtuner: torch.Tensor = ret['pixel_values']
                image_grid_thw_xtuner: torch.Tensor = ret['image_grid_thw']

                # to hf openai format
                messages = raw_data['messages']
                if i != 10:
                    messages[0]['content'][0]['type'] = 'image'
                    messages[0]['content'][0]['path'] = 'tests/' + messages[0]['content'][0]['image']['url']
                    messages[0]['content'][1]['type'] = 'image'
                    messages[0]['content'][1]['path'] = 'tests/' + messages[0]['content'][1]['image']['url']
                    del messages[0]['content'][0]['image']
                    del messages[0]['content'][1]['image']
                else:
                    messages[0]['content'][0]['type'] = 'image'
                    messages[0]['content'][0]['path'] = 'tests/' + messages[0]['content'][0]['image']['url']
                    del messages[0]['content'][0]['image']

                    messages[4]['content'][0]['type'] = 'image'
                    messages[4]['content'][0]['path'] = 'tests/' + messages[4]['content'][0]['image']['url']
                    del messages[4]['content'][0]['image']

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

    def test_calc_frame_info(self):
        self.tokenize_fn.state = "cache"
        data_path = 'tests/resource/mllm_video_frame_test_data_new.jsonl'
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
        QWEN35_VL_PATH = os.environ["QWEN3_5_MOE_PATH"]
        tokenize_fn = Qwen3VLTokenizeFnConfig(processor_path=QWEN35_VL_PATH, rand_video_max_frames=14,
                                              chat_template="qwen3.5-vl",
                                              add_vision_id=add_vision_id).build(
            self.tokenizer)
        data_path = 'tests/resource/mllm_sft_video_example_data_new.jsonl'
        hf_data_path = 'tests/resource/mllm_sft_video_hf_example_data_new.jsonl'
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
                    else:
                        if i == 7:
                            self.assertEqual(len(input_ids_xtuner), len(input_ids_hf))
                        else:
                            self.assertEqual(input_ids_xtuner, input_ids_hf.tolist())
                        self.assertTrue('seconds>' in text)
                        self.assertTrue(torch.allclose(pixel_values_xtuner, pixel_values_hf))
                        self.assertTrue(torch.allclose(image_grid_thw_xtuner, image_grid_thw_hf))

    @parametrize.parametrize("add_vision_id", [(True,), (False,)])
    def test_qwen3_vl_pretrain_image(self, add_vision_id):
        QWEN35_VL_PATH = os.environ["QWEN3_5_MOE_PATH"]
        tokenize_fn = Qwen3VLTokenizeFnConfig(processor_path=QWEN35_VL_PATH,
                                              chat_template="qwen3.5-vl",
                                              add_vision_id=add_vision_id).build(self.tokenizer)
        data_path = 'tests/resource/mllm_pretrain_image_example_data_new.jsonl'
        total_step = 60
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
                input_xtuner_str = input_str.replace('<|vision_start|><|vision_end|>', '<|vision_start|><|image_pad|><|vision_end|>')
                
                messages = raw_data['messages']
                messages[0]['role'] = 'user'
                hf_text = self.tokenizer.apply_chat_template(raw_data['messages'],          
                                               add_vision_id=add_vision_id,   
                                               tokenize=False,
                                               enable_thinking=False,
                                               add_generation_prompt=False)
                hf_text = hf_text.replace('<|im_start|>user\n', '')   
                hf_text = hf_text[:-1]  # remove \n                   
                self.assertEqual(input_xtuner_str, hf_text)
                self.assertTrue((labels_xtuner == self.tokenize_fn.img_context_token_id).sum() == 0)
    
    @parametrize.parametrize("add_vision_id", [(True,), (False,)])
    def test_qwen3_vl_pretrain_video(self, add_vision_id):
        QWEN35_VL_PATH = os.environ["QWEN3_5_MOE_PATH"]
        tokenize_fn = Qwen3VLTokenizeFnConfig(processor_path=QWEN35_VL_PATH,
                                              chat_template="qwen3.5-vl",
                                              add_vision_id=add_vision_id).build(self.tokenizer)
        data_path = 'tests/resource/mllm_pretrain_video_example_data_new.jsonl'
        total_step = 60
        with open(data_path, encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= total_step:   
                    break
                raw_data = json.loads(line)
                ret = tokenize_fn(raw_data, media_root=VIDEO_ROOT)
                labels_xtuner = torch.tensor(ret['labels'])
                self.assertTrue((labels_xtuner == tokenize_fn.video_context_token_id).sum() == 0)
