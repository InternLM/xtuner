# Copyright (c) OpenMMLab. All rights reserved.
import copy
from itertools import chain

from mmchat.utils import IGNORE_INDEX


def encode_fn(example, tokenizer, max_length, input_with_labels=True):
    """We only support the following three scenarios:

    1. Incremental pretraining dataset, where example['input'] is empty and
        example['output'] contains the text of the dataset.

    2. Single-turn conversation dataset, in which example['input'] and
        example['output'] represent one question and its corresponding answer
        pair.

    3. Multi-turn conversation dataset, where example['input'] and
        example['output'] consist of a series of question-answer pairs.
        It is required that the length of example['input'] matches the length
        of example['output'], and `input_with_labels` is set to True.
    """
    encode_kwargs = {}
    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        encode_kwargs['disallowed_special'] = ()

    is_multi_turn_conversation = len(example['input']) > 1
    if is_multi_turn_conversation:
        assert input_with_labels
        assert len(example['input']) == len(
            example['output']
        ), 'In a multi-turn conversation, the number of inputs should be ' \
            'equal to the number of outputs.'

    input_ids, labels = [], []
    for input, output in zip(example['input'], example['output']):
        input_encode = tokenizer(
            f'{tokenizer.bos_token}{input}',
            add_special_tokens=False,
            **encode_kwargs)
        input_ids += input_encode['input_ids']
        labels += [IGNORE_INDEX] * len(input_encode['input_ids'])
        if input_with_labels:
            output_encode = tokenizer(
                f'{output}{tokenizer.eos_token}',
                add_special_tokens=False,
                **encode_kwargs)
            input_ids += output_encode['input_ids']
            labels += copy.deepcopy(output_encode['input_ids'])

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    return {'input_ids': input_ids, 'labels': labels}


class Concatenator:
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
