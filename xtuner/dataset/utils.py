# Copyright (c) OpenMMLab. All rights reserved.
import base64
import copy
import io
from io import BytesIO
from itertools import chain

import requests
from PIL import Image

from xtuner.utils import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX


def encode_fn(example,
              tokenizer,
              max_length,
              input_ids_with_output=True,
              with_image_token=False):
    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        bos_token_id = []
        eos_token_id = tokenizer.eos_token_id
    elif tokenizer.__class__.__name__ == 'ChatGLMTokenizer':
        bos_token_id = [64790, 64792]
        eos_token_id = tokenizer.eos_token_id
    else:
        bos_token_id = tokenizer.bos_token_id
        eos_token_id = tokenizer.eos_token_id
    if isinstance(bos_token_id, int):
        bos_token_id = [bos_token_id]
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    is_multi_turn_conversation = len(example['conversation']) > 1
    if is_multi_turn_conversation:
        assert input_ids_with_output

    input_ids, labels = [], []
    is_last_turn_handled = False  # 在函数开始处初始化变量    
    for idx, single_turn_conversation in enumerate(example['conversation']):
        # 判断是否为最后一轮对话
        is_last_turn = idx == len(example['conversation']) - 1
        input = single_turn_conversation['input']

        # 处理带有图片标记的输入
        if DEFAULT_IMAGE_TOKEN in input and with_image_token:
            chunk_encode = [
                tokenizer(chunk, add_special_tokens=False)
                for chunk in input.split('<image>')
            ]
            assert len(chunk_encode) == 2
            input_encode = {'input_ids': []}
            for idx, cur_chunk_encode in enumerate(chunk_encode):
                input_encode['input_ids'].extend(cur_chunk_encode['input_ids'])
                if idx != len(chunk_encode) - 1:
                    input_encode['input_ids'].append(IMAGE_TOKEN_INDEX)
        else:
            input_encode = tokenizer(f'{input}', add_special_tokens=False)

        # 添加开始标记和编码后的输入
        input_ids += bos_token_id + input_encode['input_ids']
        labels += [IGNORE_INDEX] * (len(bos_token_id + input_encode['input_ids']))

        # 判断最后一轮对话的 "input" 部分长度是否等于 max_length
        # 检查添加这轮对话后长度是否超过 max_length
        if len(input_ids) + len(bos_token_id) + len(input_encode['input_ids']) > max_length:
            # 如果是最后一轮，设置标志
            is_last_turn_handled = is_last_turn
            break

        # 处理输出
        if input_ids_with_output:
            output = single_turn_conversation['output']
            output_encode = tokenizer(f'{output}', add_special_tokens=False)
            input_ids += output_encode['input_ids'] + eos_token_id
            labels += output_encode['input_ids'] + eos_token_id

    # 处理超过最大长度的输入
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

    return {'input_ids': input_ids, 'labels': labels, 'is_last_turn_handled': is_last_turn_handled}


class Packer:
    # modified from
    # https://github.com/facebookresearch/llama-recipes/blob/main/ft_datasets/utils.py

    def __init__(self, chunk_size=2048):
        self.chunk_size = chunk_size
        self.residual = {'input_ids': [], 'labels': []}

    def __call__(self, batch):
        # 处理残留的部分和当前批次
        concatenated_samples = {
            k: v + list(chain(*[example[k] for example in batch]))
            for k, v in self.residual.items()
        }

        # 检查最后一轮对话是否被处理
        is_last_turn_handled = any(example['is_last_turn_handled'] for example in batch)

        total_length = len(concatenated_samples['input_ids'])

        if total_length >= self.chunk_size:
            chunk_num = total_length // self.chunk_size
            result = {
                k: [
                    v[i:i + self.chunk_size]
                    for i in range(0, chunk_num * self.chunk_size, self.chunk_size)
                ] for k, v in concatenated_samples.items()
            }

            if is_last_turn_handled:
                # 如果最后一轮对话恰好等于 max_length，处理特殊情况
                last_input_ids = batch[-1]['input_ids']
                last_labels = batch[-1]['labels']
                self.residual = {'input_ids': last_input_ids, 'labels': last_labels}
            else:
                # 正常情况下的残留处理
                self.residual = {
                    k: v[(chunk_num * self.chunk_size):]
                    for k, v in concatenated_samples.items()
                }
        else:
            result = {k: [v] for k, v in concatenated_samples.items()}
            self.residual = {k: [] for k in concatenated_samples.keys()}
        print("Current residual:", self.residual)      
        return result

class InternRepoPacker:
    """Only used for packing data in InternLM repo
    (https://github.com/InternLM/InternLM) format."""

    def __init__(self, chunk_size=2048):
        self.chunk_size = chunk_size
        self.residual = []

    def __call__(self, batch):
        concatenated_samples = self.residual + list(chain(*batch['input_ids']))

        total_length = len(concatenated_samples)

        if total_length >= self.chunk_size:
            chunk_num = total_length // self.chunk_size
            input_ids = [
                concatenated_samples[i:i + self.chunk_size]
                for i in range(0, chunk_num * self.chunk_size, self.chunk_size)
            ]
            result = {'input_ids': input_ids}
            self.residual = concatenated_samples[(chunk_num *
                                                  self.chunk_size):]
        else:
            input_ids = [concatenated_samples]
            result = {'input_ids': input_ids}
            self.residual = []

        return result


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image
