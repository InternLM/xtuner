# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence

from xtuner.utils import DEFAULT_PAD_TOKEN_INDEX, IGNORE_INDEX


def default_collate_fn(instances: Sequence[Dict],
                       pad_index: int = DEFAULT_PAD_TOKEN_INDEX,
                       return_hf_format: bool = False,
                       use_varlen_attn: bool = False):

    input_ids, labels = [], []
    input_chosen_ids, chosen_labels = [], []
    input_reject_ids, reject_labels = [], []
    has_image = any(inst.get('pixel_values') is not None for inst in instances)
    if use_varlen_attn:
        cumulative_len, indexes = [], []
        assert len(instances) == 1, (
            f'If utilizing local attention, the batch size should be'
            f' set to 1, but got {len(instances)}')
        assert not has_image, 'Currently, it is not configured to '
        'accommodate the use of varlen Attention in multimodal training'

    if has_image:
        pixel_values = []

    for example in instances:
        if 'input_chosen_ids' in example:
            with_dpo = True
        if with_dpo:
            input_chosen_ids.append(torch.LongTensor(example['input_chosen_ids']))
            chosen_labels.append(torch.LongTensor(example['chosen_labels']))
            input_reject_ids.append(torch.LongTensor(example['input_reject_ids']))
            reject_labels.append(torch.LongTensor(example['reject_labels']))
        else:
            input_ids.append(torch.LongTensor(example['input_ids']))
            labels.append(torch.LongTensor(example['labels']))
        if use_varlen_attn:
            cumulative_len.append(torch.IntTensor(example['cumulative_len']))
            indexes.append(torch.LongTensor(example['indexes']))

        if has_image:
            pixel_values.append(example['pixel_values'])

    if len(instances) > 1:
        if with_dpo:
            input_chosen_ids = pad_sequence(
                input_chosen_ids, batch_first=True, padding_value=pad_index)
            chosen_labels = pad_sequence(
                chosen_labels, batch_first=True, padding_value=IGNORE_INDEX)
            input_reject_ids = pad_sequence(
                input_reject_ids, batch_first=True, padding_value=pad_index)
            reject_labels = pad_sequence(
                reject_labels, batch_first=True, padding_value=IGNORE_INDEX)
        else:
            input_ids = pad_sequence(
                input_ids, batch_first=True, padding_value=pad_index)
            labels = pad_sequence(
                labels, batch_first=True, padding_value=IGNORE_INDEX)
    else:
        if not with_dpo:
            input_ids = torch.stack(input_ids)
            labels = torch.stack(labels)
        else:
            input_chosen_ids = torch.stack(input_chosen_ids)
            chosen_labels = torch.stack(chosen_labels)
            input_reject_ids = torch.stack(input_reject_ids)
            reject_labels = torch.stack(reject_labels)

    if use_varlen_attn:
        indexes = torch.stack(indexes, dim=0)
        max_seqlen = (
            cumulative_len[0][1:] -  # noqa: W504
            cumulative_len[0][:-1]).max().item()
        data_dict = {
            'input_ids': input_ids,
            'cumulative_len': cumulative_len,
            'indexes': indexes,
            'labels': labels,
            'max_seqlen': max_seqlen
        }
    else:
        if with_dpo:
            data_dict = {
                'input_chosen_ids': input_chosen_ids,
                'chosen_attention_mask': input_chosen_ids.ne(pad_index),
                'chosen_labels': chosen_labels,
                'input_reject_ids': input_reject_ids,
                'reject_attention_mask': input_reject_ids.ne(pad_index),
                'reject_labels': reject_labels
            }
        else:
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
