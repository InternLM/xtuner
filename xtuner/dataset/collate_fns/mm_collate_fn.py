# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from xtuner.utils import DEFAULT_PAD_TOKEN_INDEX, IGNORE_INDEX


def mm_collate_fn(instances: Sequence[Dict],
                  pad_index: int = DEFAULT_PAD_TOKEN_INDEX,
                  return_hf_format: bool = False,
                  extra_collate_keys=None):
    input_ids = []
    labels = []
    cumulative_len = []
    position_ids = []

    has_image = any(inst.get('pixel_values') is not None for inst in instances)
    has_labels = any(inst.get('labels') is not None for inst in instances)
    mode = 'train' if has_labels else 'eval'

    if has_image:
        pixel_values = []

    for i, data in enumerate(instances):
        input_ids.append(torch.LongTensor(data['input_ids']))
        if mode == 'train':
            labels.append(torch.LongTensor(data['labels']))

        if 'cumulative_len' in data:
            cumulative_len.append(torch.IntTensor(data['cumulative_len']))

        if has_image:
            pixel_values.append(data['pixel_values'])

    ori_length = [len(ids) for ids in input_ids]
    if len(instances) > 1:
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=pad_index)
        labels = pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)
    else:
        input_ids = torch.stack(input_ids)
        if mode == 'train':
            labels = torch.stack(labels)

    # Some tokenizers have the same eos token and pad token, so input_ids
    # cannot be masked directly based on the pad token id.
    attention_mask = torch.zeros_like(input_ids).bool()
    for i in ori_length:
         attention_mask[:i] = True

    if mode == 'train':
        bs, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len).unsqueeze(0).long().repeat(bs, 1)

    if len(cumulative_len) == 0:
        cumulative_len = None

    if mode == 'train':
        data_dict = {
            'input_ids': input_ids,
            'position_ids': position_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'cumulative_len': cumulative_len,
        }
    else:
        data_dict = {
            'input_ids': input_ids,
        }

    if has_image:
        # if all images have the same size, stack them into a single tensor
        # else, keep them as a list of tensors
        if all(x.shape == pixel_values[0].shape for x in pixel_values):
            pixel_values = torch.stack(pixel_values, dim=0)
        data_dict['pixel_values'] = pixel_values

    if extra_collate_keys is not None:
        for key in extra_collate_keys:
            data_dict[key] = [inst[key] for inst in instances]

    if return_hf_format:
        return data_dict
    else:
        return {'data': data_dict, 'data_samples': None}
