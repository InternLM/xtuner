from dataclasses import dataclass
from typing import Any, Dict, Sequence

import torch

from mmchat.registry import TOKENIZER
from mmchat.utils import DEFAULT_PAD_TOKEN_INDEX


@dataclass
class CollatorMMLU:

    tokenizer: Any
    max_len: int

    def __post_init__(self):
        self.tokenizer = TOKENIZER.build(self.tokenizer)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({
                'pad_token':
                self.tokenizer.convert_ids_to_tokens(DEFAULT_PAD_TOKEN_INDEX)
            })

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        inputs = []
        data_samples = {'labels': [], 'subjects': []}
        for example in instances:
            inputs.append(f"{self.tokenizer.bos_token}{example['input']}")
            data_samples['labels'].append(f"{example['output']}")
            data_samples['subjects'].append(f"{example['subject']}")
        # Tokenize
        data_dict = self.tokenizer(
            inputs,
            max_length=self.max_len,
            truncation=True,
            padding=True,
            add_special_tokens=False,
            return_tensors='pt')
        return {'data': data_dict, 'data_samples': data_samples}
