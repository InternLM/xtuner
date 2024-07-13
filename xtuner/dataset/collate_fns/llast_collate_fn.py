# Copyright (c) LLaST. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence

from xtuner.utils import (DEFAULT_PAD_TOKEN_INDEX, IGNORE_INDEX,
                          LLAST_AUDIO_PADDING_TOKEN_INDEX)


def llast_audiomask_mel_collate_fn(
        instances: Sequence[Dict],
        pad_index: int = DEFAULT_PAD_TOKEN_INDEX,
        return_hf_format: bool = False) -> Dict[str, torch.Tensor]:
    """Add audio tokens and conduct padding operation."""
    input_ids = []
    labels = []
    feats_lens = []
    has_audio = any(inst.get('audio_tokens') is not None for inst in instances)

    if has_audio:
        audio_tokens = []
    for example in instances:
        input_ids.append(torch.tensor(example['input_ids']))
        labels.append(torch.tensor(example['labels']))
        if has_audio:
            audio_tokens.append(example['audio_tokens'])
        feats_lens.append(torch.tensor(example['audio_lens']))
    if len(instances) > 1:
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=pad_index)
        labels = pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)
        # padding audio tokens
        padded_audio_tokens = pad_sequence(
            audio_tokens,
            batch_first=True,
            padding_value=LLAST_AUDIO_PADDING_TOKEN_INDEX)

    else:
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        padded_audio_tokens = torch.stack(audio_tokens)

    data_dict = {
        'input_ids': input_ids,
        'attention_mask': input_ids.ne(pad_index),
        'labels': labels
    }

    if has_audio:
        audio_lens = torch.stack(feats_lens)
        data_dict['audio_tokens'] = padded_audio_tokens
        data_dict['audio_lens'] = audio_lens

    if return_hf_format:
        return data_dict
    else:
        return {'data': data_dict, 'data_samples': instances}
