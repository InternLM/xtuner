# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence

from xtuner.utils import DEFAULT_PAD_TOKEN_INDEX, IGNORE_INDEX


def default_collate_fn(
        instances: Sequence[Dict],
        pad_index: int = DEFAULT_PAD_TOKEN_INDEX,
        return_hf_format: bool = False) -> Dict[str, torch.Tensor]:
    input_ids = []
    labels = []
    has_image = any(inst.get('pixel_values') is not None for inst in instances)
    if has_image:
        pixel_values = []
    for example in instances:
        input_ids.append(torch.tensor(example['input_ids']))
        labels.append(torch.tensor(example['labels']))
        if has_image:
            pixel_values.append(example['pixel_values'])
    if len(instances) > 1:
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=pad_index)
        labels = pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)
    else:
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)

    data_dict = {
        'input_ids': input_ids,
        'attention_mask': input_ids.ne(pad_index),
        'labels': labels
    }
    if has_image:
        pixel_values = torch.stack(pixel_values)
        data_dict['pixel_values'] = pixel_values

    if return_hf_format:
        return data_dict
    else:
        return {'data': data_dict, 'data_samples': None}
