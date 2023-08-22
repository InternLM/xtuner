# Copyright (c) OpenMMLab. All rights reserved.
import copy
import json
import os

import torch
from mmengine.config import Config, ConfigDict
from mmengine.logging import print_log
from torch.utils.data import Dataset
from tqdm import tqdm

from xtuner.registry import BUILDER


class MOSSSFTDataset(Dataset):

    def __init__(self, data_file, tokenizer, max_length=2048, bot_name=None):
        super().__init__()
        self.bot_name = bot_name
        self.src_data_file = data_file
        if isinstance(tokenizer, dict) or isinstance(
                tokenizer, Config) or isinstance(tokenizer, ConfigDict):
            self.tokenizer = BUILDER.build(tokenizer)
        else:
            self.tokenizer = tokenizer
        self.max_length = max_length

        self.data = []
        # We do not calculate losses for the meta instruction or results
        # returned by plugins
        # The token spans with label -100, [(span_start, span_end), ...]
        self.no_loss_spans = []
        self.labels = []

        self.pre = len(
            self.tokenizer.encode('<|Results|>:', add_special_tokens=False))
        self.post = len(
            self.tokenizer.encode('<eor>\n', add_special_tokens=False))

        self.load_data()
        self.process_data()

    def load_data(self):
        print_log('Loading MOSS SFT data...', 'current')
        name = f'{self.tokenizer.__class__.__name__}_{self.bot_name}'
        data_file = self.src_data_file.replace('.jsonl', f'_data_{name}')
        no_loss_spans_file = self.src_data_file.replace(
            '.jsonl', f'_no_loss_spans_{name}')
        if os.path.exists(data_file) and os.path.exists(no_loss_spans_file):
            self.data = torch.load(data_file, map_location='cpu')
            self.no_loss_spans = torch.load(
                no_loss_spans_file, map_location='cpu')
        else:
            with open(self.src_data_file) as f:
                for line in tqdm(f):
                    sample = json.loads(line)

                    chat = sample['chat']
                    num_turns = int(sample['num_turns'])

                    meta_instruction = sample['meta_instruction']
                    if self.bot_name is not None:
                        meta_instruction = meta_instruction.replace(
                            'MOSS', self.bot_name)
                    instruction_ids = self.tokenizer.encode(meta_instruction)
                    assert isinstance(instruction_ids,
                                      list) and len(instruction_ids) > 0

                    input_ids = copy.deepcopy(instruction_ids)
                    no_loss_spans = [(0, len(instruction_ids))]
                    try:
                        for i in range(num_turns):
                            cur_turn_ids = []
                            cur_no_loss_spans = []
                            cur_turn = chat[f'turn_{i+1}']
                            for key, value in cur_turn.items():
                                if self.bot_name is not None:
                                    value = value.replace(
                                        'MOSS', self.bot_name)
                                cur_ids = self.tokenizer.encode(
                                    value, add_special_tokens=False)
                                if key == 'Tool Responses':
                                    # The format tokens
                                    # (<|Results|>:...<eor>\n)
                                    # should have losses.
                                    cur_no_loss_spans.append(
                                        (len(input_ids + cur_turn_ids) +
                                         self.pre,
                                         len(input_ids + cur_turn_ids +
                                             cur_ids) - self.post))

                                assert isinstance(cur_ids,
                                                  list) and len(cur_ids) > 0

                                cur_turn_ids.extend(cur_ids)

                            if len(input_ids + cur_turn_ids) > self.max_length:
                                break

                            input_ids.extend(cur_turn_ids)
                            no_loss_spans.extend(cur_no_loss_spans)
                        if len(input_ids) == len(instruction_ids):
                            continue

                        assert len(input_ids) > 0 and len(
                            input_ids) <= self.max_length

                        self.data.append(input_ids)
                        self.no_loss_spans.append(no_loss_spans)
                    except Exception:
                        pass
            torch.save(self.data, data_file)
            torch.save(self.no_loss_spans, no_loss_spans_file)
        print_log(
            f'Load data successfully, total {len(self.data)} training samples',
            'current')

    def process_data(self):
        for item, no_loss in zip(self.data, self.no_loss_spans):
            label = copy.deepcopy(item)
            for loc in no_loss:
                label[loc[0]:loc[1]] = [-100] * (loc[1] - loc[0])
            self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {'input_ids': self.data[index], 'labels': self.labels[index]}
