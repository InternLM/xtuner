# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence

from xtuner.utils import DEFAULT_PAD_TOKEN_INDEX, IGNORE_INDEX


def intern_repo_collate_fn(
        instances: Sequence[Dict],
        pad_index: int = DEFAULT_PAD_TOKEN_INDEX,
        return_hf_format: bool = False) -> Dict[str, torch.Tensor]:
    input_ids = []
    labels = []
    cumulative_len = []
    indexes = []
    max_seqlen = 0
    for example in instances:
        cur_input_ids = torch.tensor(example['input_ids'])
        cur_labels = copy.deepcopy(cur_input_ids)
        cur_input_ids = cur_input_ids.abs()
        cur_labels[cur_labels < 0] = IGNORE_INDEX
        input_ids.append(cur_input_ids)
        labels.append(cur_labels)

        cumulative_len_cur = torch.IntTensor(example['cumulative_len'])
        cumulative_len.append(cumulative_len_cur)
        index_cur = []
        for i in range(len(cumulative_len_cur) - 1):
            index_cur.extend(
                list(range(cumulative_len_cur[i + 1] - cumulative_len_cur[i])))
        indexes.append(torch.LongTensor(index_cur))
        max_seqlen = max(max_seqlen, (cumulative_len_cur[1:] -
                                      cumulative_len_cur[:-1]).max().item())

    if len(instances) > 1:
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=pad_index)
        labels = pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)
    else:
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)

    indexes = torch.stack(indexes, dim=0)
    # if len(set(map(len, cumulative_len))) == 1:
    #     # if has uniform length, then stack to save device transfer time
    #     cumulative_len = torch.stack(cumulative_len, dim=0)

    data_dict = {
        'input_ids': input_ids,
        'attention_mask': input_ids.ne(pad_index),
        'labels': labels,
        'cumulative_len': cumulative_len,
        'indexes': indexes,
        'max_seqlen': max_seqlen
    }

    if return_hf_format:
        return data_dict
    else:
        return {'data': data_dict, 'data_samples': None}
