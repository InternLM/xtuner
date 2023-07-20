import copy
from dataclasses import dataclass
from typing import Any, Dict, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence

from mmchat.registry import TOKENIZER
from mmchat.utils import DEFAULT_PAD_TOKEN_INDEX, IGNORE_INDEX


@dataclass
class CollatorWithPadding:

    tokenizer: Any
    source_max_len: int
    target_max_len: int
    train_on_source: bool = False
    predict_with_generate: bool = False

    def __post_init__(self):
        self.tokenizer = TOKENIZER.build(self.tokenizer)

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [
            f"{self.tokenizer.bos_token}{example['input']}"
            for example in instances
        ]
        targets = [
            f"{example['output']}{self.tokenizer.eos_token}"
            for example in instances
        ]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
                tokenized_sources_with_prompt['input_ids'],
                tokenized_targets['input_ids']):
            if not self.predict_with_generate:
                input_ids.append(
                    torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor([
                            IGNORE_INDEX for _ in range(len(tokenized_source))
                        ] + copy.deepcopy(tokenized_target)))
                else:
                    labels.append(
                        torch.tensor(
                            copy.deepcopy(tokenized_source +
                                          tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))

        # Apply padding
        if self.tokenizer.pad_token_id is not None:
            pad_index = self.tokenizer.pad_token_id
        else:
            pad_index = DEFAULT_PAD_TOKEN_INDEX
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=pad_index)
        labels = pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        ) if not self.predict_with_generate else None
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(pad_index),
        }

        if labels is not None:
            data_dict['labels'] = labels

        return {'data': data_dict, 'data_samples': None}
