# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence

from xtuner.utils import DEFAULT_PAD_TOKEN_INDEX, IGNORE_INDEX
from xtuner.types import RawTrainingData


def hybrid_collate_fn(instances: Sequence[Dict],
                       pad_index: int = DEFAULT_PAD_TOKEN_INDEX,
                       return_hf_format: bool = False):

    input_ids = []
    labels = []
    pixel_values = []
    cumulative_len = []
    image_ranges = []
    # indexes = []
    
    
    for item in instances:
        input_ids.append(torch.LongTensor(item['input_ids']))
        labels.append(torch.LongTensor(item['labels']))
        
        if 'cumulative_len' in item:
            cumulative_len.append(torch.IntTensor(item['cumulative_len']))
        
        pixel_values.extend(item['pixel_values'])
        # image_ranges.extend(torch.IntTensor(item['image_ranges']))

    if len(pixel_values) > 0:
        pixel_values = torch.stack(pixel_values)
    else:
        pixel_values = None
    
    if len(instances) > 1:
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=pad_index)
        labels = pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)
    else:
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)

    # if len(image_ranges) > 0:
    #     image_ranges = torch.stack(image_ranges)
    # else:
    #     image_ranges = None
    
    if len(cumulative_len) == 0:
        cumulative_len = None
        
    data_dict = {
        'input_ids': input_ids,
        'attention_mask': input_ids.ne(pad_index),
        'labels': labels,
        'pixel_values': pixel_values,
        'cumulative_len': cumulative_len,
        # 'image_ranges': image_ranges,
    }


    if return_hf_format:
        return data_dict
    else:
        return {'data': data_dict, 'data_samples': None}
