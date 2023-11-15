# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence

from xtuner.utils import DEFAULT_PAD_TOKEN_INDEX, IGNORE_INDEX


def mmlu_collate_fn(instances: Sequence[Dict],
                    pad_index: int = DEFAULT_PAD_TOKEN_INDEX,
                    return_hf_format: bool = False) -> Dict[str, torch.Tensor]:
    input_ids = []
    labels = []
    data_samples = {'labels': [], 'subjects': []}
    for example in instances:
        input_ids.append(torch.tensor(example['input_ids']))
        labels.append(torch.tensor(example['labels']))
        data_samples['labels'].append(example['output'])
        data_samples['subjects'].append(example['subject'])
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

    if return_hf_format:
        return data_dict
    else:
        return {'data': data_dict, 'data_samples': data_samples}
