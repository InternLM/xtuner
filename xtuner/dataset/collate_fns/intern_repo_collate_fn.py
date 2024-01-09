# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Sequence

import torch


def intern_repo_collate_fn(instances: Sequence[Dict],
                           packed_length: int,
                           return_hf_format: bool = False,
                           use_local_attn: bool = True):

    input_ids, labels = [], []
    if use_local_attn:
        cumulative_len, indexes = [], []
        assert len(instances) == 1, (
            f'If utilizing local attention, the batch size should be'
            f' set to 1, but got {len(instances)}')

    for example in instances:
        assert (len(example['input_ids']) == packed_length), (
            f'length of a sample should be equal to packed_length, '
            f"but got {len(example['input_ids'])} and {packed_length})")
        assert (len(example['labels']) == packed_length), (
            f'length of a sample should be equal to packed_length, '
            f"but got {len(example['labels'])} and {packed_length})")

        input_ids_per_sample = [abs(w) for w in example['input_ids']]
        labels_per_sample = [w if w > 0 else -100 for w in example['labels']]

        input_ids.append(torch.LongTensor(input_ids_per_sample))
        labels.append(torch.LongTensor(labels_per_sample))
        if use_local_attn:
            cumulative_len.append(torch.IntTensor(example['cumulative_len']))
            indexes.append(torch.LongTensor(example['indexes']))

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True)
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100)
    assert input_ids.shape[1] == packed_length, (input_ids.shape[1],
                                                 packed_length)

    if use_local_attn:
        indexes = torch.stack(indexes, dim=0)
        max_seqlen = (cumulative_len[0][1:] -
                      cumulative_len[0][:-1]).max().item()
        data_dict = {
            'input_ids': input_ids,
            'cumulative_len': cumulative_len,
            'indexes': indexes,
            'labels': labels,
            'max_seqlen': max_seqlen
        }
    else:
        data_dict = {'input_ids': input_ids, 'labels': labels}

    if return_hf_format:
        return data_dict
    else:
        return {'data': data_dict, 'data_samples': None}
