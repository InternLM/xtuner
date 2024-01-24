# Copyright (c) OpenMMLab. All rights reserved.
import base64
import copy
import io
from io import BytesIO
from itertools import chain

import numpy as np
import requests
from PIL import Image

from xtuner.utils import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX


def encode_fn(example,
              tokenizer,
              max_length,
              input_ids_with_output=True,
              with_image_token=False):
    """We only support the following three scenarios:

    1. Incremental pretraining dataset.
        example['conversation'] = [
                {
                    'input': '',
                    'output': '### Human: Can you write xxx'
                }
            ]

    2. Single-turn conversation dataset.
        example['conversation'] = [
                {
                    'input': 'Give three tips for staying healthy.',
                    'output': '1.Eat a balanced diet xxx'
                }
            ]

    3. Multi-turn conversation dataset.
        example['conversation'] = [
                {
                    'input': 'Give three tips for staying healthy.',
                    'output': '1.Eat a balanced diet xxx'
                },
                {
                    'input': 'Please expand on the second point.',
                    'output': 'Here is an expanded explanation of the xxx'
                }
            ]
    """
    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        bos_token_id = []
        eos_token_id = tokenizer.eos_token_id
        assert eos_token_id is not None, \
            'Please set eos_token for Qwen tokenizer!'
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
    next_needs_bos_token = True
    for single_turn_conversation in example['conversation']:
        input = single_turn_conversation['input']
        if DEFAULT_IMAGE_TOKEN in input and with_image_token:
            chunk_encode = [
                tokenizer.encode(chunk, add_special_tokens=False)
                for chunk in input.split('<image>')
            ]
            assert len(chunk_encode) == 2
            input_encode = []
            for idx, cur_chunk_encode in enumerate(chunk_encode):
                input_encode.extend(cur_chunk_encode)
                if idx != len(chunk_encode) - 1:
                    input_encode.append(IMAGE_TOKEN_INDEX)
        else:
            input_encode = tokenizer.encode(input, add_special_tokens=False)
        if next_needs_bos_token:
            input_ids += bos_token_id
            labels += [IGNORE_INDEX] * len(bos_token_id)
        input_ids += input_encode
        labels += [IGNORE_INDEX] * len(input_encode)
        if input_ids_with_output:
            # Add output (with loss)
            output = single_turn_conversation['output']
            output_encode = tokenizer.encode(output, add_special_tokens=False)
            input_ids += output_encode
            labels += copy.deepcopy(output_encode)
            # Add EOS_TOKEN (with loss)
            if single_turn_conversation['need_eos_token']:
                next_needs_bos_token = True
                input_ids += eos_token_id
                labels += copy.deepcopy(eos_token_id)
            else:
                next_needs_bos_token = False
            # Add SEP (without loss)
            sep = single_turn_conversation['sep']
            if sep != '':
                sep_encode = tokenizer.encode(sep, add_special_tokens=False)
                input_ids += sep_encode
                labels += [IGNORE_INDEX] * len(sep_encode)

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    return {'input_ids': input_ids, 'labels': labels}


class Packer:
    """Only used for packing data in InternLM repo
    (https://github.com/InternLM/InternLM) format."""

    def __init__(self,
                 chunk_size=2048,
                 use_varlen_attn=False,
                 drop_last=False):
        use_varlen_attn = True
        drop_last = True
        self.chunk_size = chunk_size
        self.residual = {'input_ids': [], 'labels': []}
        self.use_varlen_attn = use_varlen_attn
        self.drop_last = drop_last
        if use_varlen_attn:
            self.residual_cumulative_len = [0]

    def get_cumulative_len(self, chunk_num):
        ptr_l = 0
        cumulative_len = []
        for chunk_idx in range(chunk_num):
            length_train = (chunk_idx + 1) * self.chunk_size
            ptr_r = np.searchsorted(
                self.residual_cumulative_len, length_train, side='left')
            if self.residual_cumulative_len[ptr_r] == length_train:
                cumulative_len_cur = \
                    self.residual_cumulative_len[ptr_l:ptr_r + 1]
                ptr_l = ptr_r + 1
            else:
                cumulative_len_cur = self.residual_cumulative_len[
                    ptr_l:ptr_r] + [length_train]
                ptr_l = ptr_r
            cumulative_len_cur = [
                num - chunk_idx * self.chunk_size for num in cumulative_len_cur
            ]
            if cumulative_len_cur[0] != 0:
                cumulative_len_cur = [0] + cumulative_len_cur

            cumulative_len.append(cumulative_len_cur)

        self.residual_cumulative_len = [
            num - length_train for num in self.residual_cumulative_len[ptr_l:]
        ]
        if len(self.residual_cumulative_len) == 0:
            self.residual_cumulative_len = [0]
        elif self.residual_cumulative_len[0] != 0:
            self.residual_cumulative_len = [0] + self.residual_cumulative_len

        return cumulative_len

    def __call__(self, batch):
        concatenated_samples = {
            k: v + list(chain(*batch[k]))
            for k, v in self.residual.items()
        }

        if self.use_varlen_attn:
            for input_id in batch['input_ids']:
                self.residual_cumulative_len.append(
                    self.residual_cumulative_len[-1] + len(input_id))

        total_length = len(concatenated_samples[list(
            concatenated_samples.keys())[0]])

        if total_length >= self.chunk_size:
            chunk_num = total_length // self.chunk_size
            result = {
                k: [
                    v[i:i + self.chunk_size] for i in range(
                        0,
                        chunk_num *  # noqa: W504
                        self.chunk_size,
                        self.chunk_size)
                ]
                for k, v in concatenated_samples.items()
            }
            self.residual = {
                k: v[(chunk_num * self.chunk_size):]
                for k, v in concatenated_samples.items()
            }

            if self.use_varlen_attn:
                result['cumulative_len'] = self.get_cumulative_len(chunk_num)
        else:
            if self.drop_last:
                result = {k: [] for k, v in concatenated_samples.items()}
            else:
                result = {k: [v] for k, v in concatenated_samples.items()}
            self.residual = {k: [] for k in concatenated_samples.keys()}
            if self.use_varlen_attn:
                result['cumulative_len'] = [] if self.drop_last else [
                    self.residual_cumulative_len
                ]
                self.residual_cumulative_len = [0]

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
