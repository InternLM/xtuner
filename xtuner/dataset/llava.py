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


class LLaVADataset(Dataset):

    def __init__(self,
                 data_path,
                 image_folder,
                 tokenizer,
                 processor,
                 max_dataset_length=None,
                 dataset_map_fn=None,
                 template_map_fn=None,
                 max_length=2048,
                 image_aspect_ratio='pad'):
        super().__init__()

        json_data = json.load(open(data_path))
        for idx in range(len(json_data)):
            if isinstance(json_data[idx]['id'], int):
                json_data[idx]['id'] = str(json_data[idx]['id'])
        json_data = DatasetDict({'train': HFDataset.from_list(json_data)})
        self.text_data = process_hf_dataset(
            dataset=json_data,
            tokenizer=tokenizer,
            max_length=max_length,
            dataset_map_fn=dataset_map_fn,
            template_map_fn=template_map_fn,
            split='train',
            max_dataset_length=max_dataset_length,
            remove_unused_columns=False,
            pack_to_max_length=False,
            with_image_token=True)

        self.image_folder = image_folder
        if isinstance(processor, dict) or isinstance(
                processor, Config) or isinstance(processor, ConfigDict):
            self.processor = BUILDER.build(processor)
        else:
            self.processor = processor
        self.image_aspect_ratio = image_aspect_ratio

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, index):
        data_dict = self.text_data[index]
        if data_dict.get('image', None) is not None:
            image_file = data_dict['image']
            image = Image.open(os.path.join(self.image_folder,
                                            image_file)).convert('RGB')
            if self.image_aspect_ratio == 'pad':
                image = expand2square(
                    image,
                    tuple(int(x * 255) for x in self.processor.image_mean))
                image = self.processor.preprocess(
                    image, return_tensors='pt')['pixel_values'][0]
            else:
                image = self.processor.preprocess(
                    image, return_tensors='pt')['pixel_values'][0]
            data_dict['pixel_values'] = image
        else:
            crop_size = self.processor.crop_size
            data_dict['pixel_values'] = torch.zeros(3, crop_size['height'],
                                                    crop_size['width'])
        return data_dict
