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


class DPODataset(Dataset):

    def __init__(self,
                 data_path,
                 tokenizer,
                 max_dataset_length=None,
                 dataset_map_fn=None,
                 template_map_fn=None,
                 max_length=2048):
        super().__init__()
        # TODO
        pass

    def __len__(self):
        # TODO
        pass

    def __getitem__(self, index):
        # TODO(lsh) 对于DPO数据集，分词编码返回预处理过的data_dict
        pass
