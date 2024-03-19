# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence

from xtuner.utils import DEFAULT_PAD_TOKEN_INDEX, IGNORE_INDEX


def hybrid_collate_fn(instances: Sequence[Dict],
                      pad_index: int = DEFAULT_PAD_TOKEN_INDEX,
                      return_hf_format: bool = False):

    input_ids = []
    labels = []
    pixel_values = []
    cumulative_len = []
    image_ranges = []
    image_belong = []
    position_ids = []

    for i, data in enumerate(instances):
        input_ids.append(torch.LongTensor(data['input_ids']))
        labels.append(torch.LongTensor(data['labels']))
        position_ids.append(torch.IntTensor(data['position_ids']))

        if 'cumulative_len' in data:
            cumulative_len.append(torch.IntTensor(data['cumulative_len']))

        image_belong.append(i)
        pixel_values.extend(data['pixel_values'])
        image_ranges.extend(torch.IntTensor(data['image_ranges']))

    if len(pixel_values) > 0:
        assert len(image_ranges) > 0
        assert len(image_belong) > 0

        pixel_values = torch.stack(pixel_values)
        image_ranges = torch.stack(image_ranges)
        image_belong = torch.IntTensor(image_belong)
    else:
        pixel_values = None
        image_ranges = None
        image_belong = None

    if len(instances) > 1:
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=pad_index)
        labels = pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)
        position_ids = pad_sequence(labels, batch_first=True, padding_value=0)
    else:
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        position_ids = torch.stack(position_ids)

    if len(cumulative_len) == 0:
        cumulative_len = None

    data_dict = {
        'input_ids': input_ids,
        'position_ids': position_ids,
        'attention_mask': input_ids.ne(pad_index),
        'labels': labels,
        'pixel_values': pixel_values,
        'cumulative_len': cumulative_len,
        'image_ranges': image_ranges,
        'image_belong': image_belong
    }

    if return_hf_format:
        return data_dict
    else:
        return {'data': data_dict, 'data_samples': None}
