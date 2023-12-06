# Copyright (c) OpenMMLab. All rights reserved.
import copy
from itertools import chain

import numpy as np

from xtuner.utils import IGNORE_INDEX


def encode_fn(example, tokenizer, max_length, input_ids_with_output=True):
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
    for single_turn_conversation in example['conversation']:
        input = single_turn_conversation['input']
        input_encode = tokenizer(f'{input}', add_special_tokens=False)
        input_ids += bos_token_id + input_encode['input_ids']
        labels += [IGNORE_INDEX] * (
            len(bos_token_id + input_encode['input_ids']))
        if input_ids_with_output:
            output = single_turn_conversation['output']
            output_encode = tokenizer(f'{output}', add_special_tokens=False)
            input_ids += output_encode['input_ids'] + eos_token_id
            labels += copy.deepcopy(output_encode['input_ids'] + eos_token_id)

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    return {'input_ids': input_ids, 'labels': labels}


class Packer:
    # modified from
    # https://github.com/facebookresearch/llama-recipes/blob/main/ft_datasets/utils.py

    def __init__(self, chunk_size=2048):
        self.chunk_size = chunk_size
        self.residual = {'input_ids': [], 'labels': []}

    def __call__(self, batch):
        concatenated_samples = {
            k: v + list(chain(*batch[k]))
            for k, v in self.residual.items()
        }

        total_length = len(concatenated_samples[list(
            concatenated_samples.keys())[0]])

        if total_length >= self.chunk_size:
            chunk_num = total_length // self.chunk_size
            result = {
                k: [
                    v[i:i + self.chunk_size]
                    for i in range(0, chunk_num *
                                   self.chunk_size, self.chunk_size)
                ]
                for k, v in concatenated_samples.items()
            }
            self.residual = {
                k: v[(chunk_num * self.chunk_size):]
                for k, v in concatenated_samples.items()
            }
        else:
            result = {k: [v] for k, v in concatenated_samples.items()}
            self.residual = {k: [] for k in concatenated_samples.keys()}

        return result


class InternRepoPacker:
    """Only used for packing data in InternLM repo
    (https://github.com/InternLM/InternLM) format."""

    def __init__(self, chunk_size=2048):
        self.chunk_size = chunk_size
        self.residual = []
        self.residual_cumulative_len = [0]

    def __call__(self, batch):
        concatenated_samples = copy.deepcopy(self.residual)
        for input_ids in batch['input_ids']:
            assert input_ids[0] == 1 and input_ids[1] < 0
            input_ids[0] = -100000000
            assert isinstance(input_ids, list)
            concatenated_samples += input_ids

        # concatenated_samples = self.residual + list(chain(*batch['input_ids']))
        for input_id in batch['input_ids']:
            self.residual_cumulative_len.append(
                self.residual_cumulative_len[-1] + len(input_id))

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

            ptr_l = 0
            cumulative_len = []
            for chunk_idx in range(chunk_num):
                length_train = (chunk_idx + 1) * self.chunk_size
                ptr_r = np.searchsorted(
                    self.residual_cumulative_len, length_train, side='left')
                if self.residual_cumulative_len[ptr_r] == length_train:
                    cumulative_len_cur = self.residual_cumulative_len[
                        ptr_l:ptr_r + 1]
                    ptr_l = ptr_r + 1
                else:
                    cumulative_len_cur = self.residual_cumulative_len[
                        ptr_l:ptr_r] + [length_train]
                    ptr_l = ptr_r
                cumulative_len_cur = [
                    num - chunk_idx * self.chunk_size
                    for num in cumulative_len_cur
                ]
                if cumulative_len_cur[0] != 0:
                    cumulative_len_cur = [0] + cumulative_len_cur

                cumulative_len.append(cumulative_len_cur)
            result['cumulative_len'] = cumulative_len

            self.residual_cumulative_len = [
                num - length_train
                for num in self.residual_cumulative_len[ptr_l:]
            ]
            if len(self.residual_cumulative_len) == 0:
                self.residual_cumulative_len = [0]
            elif self.residual_cumulative_len[0] != 0:
                self.residual_cumulative_len = [
                    0
                ] + self.residual_cumulative_len
        else:
            # input_ids = [concatenated_samples]
            # result = {'input_ids': input_ids}
            # result['cumulative_len'] = [self.residual_cumulative_len]

            # Make sure the length of each packed data is equal to chunk_size
            result = {}
            self.residual = []
            self.residual_cumulative_len = [0]

        return result
