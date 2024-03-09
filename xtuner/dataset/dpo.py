# Copyright (c) OpenMMLab. All rights reserved.
import json
import os

import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from mmengine.config import Config, ConfigDict
from PIL import Image
from torch.utils.data import Dataset

from xtuner.registry import BUILDER
from .huggingface import process_hf_dataset
from .utils import expand2square
from xtuner.utils import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX

class DPODataset(Dataset):
    def __init__(self, dpo_dataset_dict, tokenizer):
        self.prompt_data = dpo_dataset_dict['prompt']
        self.chosen_data = dpo_dataset_dict['chosen']
        self.rejected_data = dpo_dataset_dict['rejected']
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.prompt_data)

    def __getitem__(self, index):
        prompt = self.prompt_data[index]
        chosen = self.chosen_data[index]
        rejected = self.rejected_data[index]

        # 编码文本
        prompt_encoding = self.tokenizer(prompt, truncation=True, return_tensors='pt')
        chosen_encoding = self.tokenizer(chosen, truncation=True, return_tensors='pt')
        rejected_encoding = self.tokenizer(rejected, truncation=True, return_tensors='pt')


        chosen_labels = [IGNORE_INDEX] * len(prompt_encoding['input_ids'][0]) + chosen_encoding['input_ids'][0].tolist()
        rejected_labels = [IGNORE_INDEX] * len(prompt_encoding['input_ids'][0]) + rejected_encoding['input_ids'][0].tolist()

        # 构造返回的数据字典
        data_dict = {
            'input_chosen_ids': torch.cat((prompt_encoding['input_ids'], chosen_encoding['input_ids']), dim=1).squeeze(0),
            'chosen_labels': torch.tensor(chosen_labels, dtype=torch.long),
            'input_reject_ids': torch.cat((prompt_encoding['input_ids'], rejected_encoding['input_ids']), dim=1).squeeze(0),
            'reject_labels': torch.tensor(rejected_labels, dtype=torch.long),
        }

        return data_dict



