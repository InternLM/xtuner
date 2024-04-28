# Copyright (c) OpenMMLab. All rights reserved.
import json
import logging
import os

import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk
from mmengine import print_log
from mmengine.config import Config, ConfigDict
from PIL import Image
from torch.utils.data import Dataset

from xtuner.registry import BUILDER
from .huggingface import process_hf_dataset
from .utils import expand2square, process_anyres_image, total_image_token, dynamic_preprocess


def load_jsonl(json_file):
    with open(json_file) as f:
        lines = f.readlines()
    data = []
    for line in lines:
        data.append(json.loads(line))
    return data


class LLaVADataset(Dataset):

    def __init__(self,
                 image_folder,
                 image_processor,
                 data_path=None,
                 tokenizer=None,
                 offline_processed_text_folder=None,
                 max_dataset_length=None,
                 dataset_map_fn=None,
                 template_map_fn=None,
                 max_length=2048,
                 s2_scales=None,  # [1, 2] or [1,2,3]
                 pad_image_to_square=False):
        super().__init__()

        assert offline_processed_text_folder or (data_path and tokenizer)
        if offline_processed_text_folder and data_path:
            print_log(
                'Both `offline_processed_text_folder` and '
                '`data_path` are set, and we load dataset from'
                '`offline_processed_text_folder` '
                f'({offline_processed_text_folder})',
                logger='current',
                level=logging.WARNING)

        if offline_processed_text_folder is not None:
            self.text_data = load_from_disk(offline_processed_text_folder)
        else:
            if data_path.endswith('.json'):
                json_data = json.load(open(data_path))
            elif data_path.endswith('.jsonl'):
                json_data = load_jsonl(data_path)
            else:
                raise NotImplementedError

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
        if isinstance(image_processor, dict) or isinstance(
                image_processor, Config) or isinstance(image_processor,
                                                       ConfigDict):
            self.image_processor = BUILDER.build(image_processor)
        else:
            self.image_processor = image_processor
        self.pad_image_to_square = pad_image_to_square

        self.max_s2_scale = s2_scales
        if s2_scales is not None:
            self.max_s2_scale = max(s2_scales)
            if hasattr(self.image_processor, 'crop_size'):
                self.image_processor.crop_size['height'] *= self.max_s2_scale
                self.image_processor.crop_size['width'] *= self.max_s2_scale
                self.image_processor.size['shortest_edge'] *= self.max_s2_scale
            else:
                self.image_processor.size['height'] *= self.max_s2_scale
                self.image_processor.size['width'] *= self.max_s2_scale

    @property
    def modality_length(self):
        length_list = []
        for data_dict in self.text_data:
            cur_len = len(data_dict['input_ids'])
            if data_dict.get('image', None) is None:
                cur_len = -cur_len
            length_list.append(cur_len)
        return length_list

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, index):
        data_dict = self.text_data[index]
        if data_dict.get('image', None) is not None:
            image_file = data_dict['image']
            image = Image.open(os.path.join(self.image_folder,
                                            image_file)).convert('RGB')
            if self.pad_image_to_square:
                image = expand2square(
                    image,
                    tuple(
                        int(x * 255) for x in self.image_processor.image_mean))
            image = self.image_processor.preprocess(
                image, return_tensors='pt')['pixel_values'][0]
            data_dict['pixel_values'] = image
        else:
            if hasattr(self.image_processor, 'crop_size'):
                crop_size = self.image_processor.crop_size
            else:
                crop_size = self.image_processor.size
            data_dict['pixel_values'] = torch.zeros(3, crop_size['height'],
                                                    crop_size['width'])
        return data_dict


class AnyResLLaVADataset(LLaVADataset):

    def __init__(self, image_grid_pinpoints, *args, **kwargs):
        self.image_grid_pinpoints = image_grid_pinpoints
        super().__init__(*args, **kwargs)
        # TODO: Assuming they are all squares.
        if hasattr(self.image_processor, 'crop_size'):
            self._crop_size = self.image_processor.crop_size
        else:
            self._crop_size = self.image_processor.size
        self._patch_size = self._crop_size['height']
        self._shortest_edge = self._crop_size['height']

    def __getitem__(self, index):
        data_dict = self.text_data[index]
        if data_dict.get('image', None) is not None:
            image_file = data_dict['image']
            image = Image.open(os.path.join(self.image_folder,
                                            image_file)).convert('RGB')
            orig_size = image.size
            # use to remove padding
            data_dict['orig_size'] = orig_size
            image = process_anyres_image(image, self.image_processor,
                                         self.image_grid_pinpoints,
                                         self._patch_size, self._shortest_edge,
                                         pad_mean=tuple(int(x * 255) for x in self.image_processor.image_mean),
                                         # keep the same as the original implementation
                                         orig_img_pad_to_square=self.pad_image_to_square)
            data_dict['pixel_values'] = image
        else:
            data_dict['orig_size'] = self._crop_size
            data_dict['pixel_values'] = torch.zeros(1, 3, self._crop_size['height'],
                                                    self._crop_size['width'])
        return data_dict


class InternVL_V1_5_LLaVADataset(LLaVADataset):
    def __init__(self, min_num, max_num, downsample_ratio=0.5, image_size=336, *args, **kwargs):
        self.min_num = min_num
        self.max_num = max_num
        self.downsample_ratio = downsample_ratio
        super().__init__(*args, **kwargs)

        if hasattr(self.image_processor, 'crop_size'):
            self._crop_size = self.image_processor.crop_size
        else:
            self._crop_size = self.image_processor.size
        self._patch_size = self._crop_size['height']
        self._shortest_edge = self._crop_size['height']

        # clip
        self._image_size = image_size
        self._patch_size = (self._image_size // 14) * downsample_ratio  # 12

    @property
    def modality_length(self):
        print_log('start calculating modality length', logger='current'),
        length_list = []
        for data_dict in self.text_data:
            cur_len = len(data_dict['input_ids'])
            if data_dict.get('image', None) is None:
                image_file = data_dict['image']
                image = Image.open(os.path.join(self.image_folder,
                                                image_file))
                num_image_token = total_image_token(image.size, self.min_num, self.max_num, self._image_size,
                                                    self._patch_size)
                cur_len += num_image_token
                cur_len = -cur_len
            length_list.append(cur_len)
        print_log('end calculating modality length', logger='current'),
        return length_list

    def __getitem__(self, index):
        data_dict = self.text_data[index]
        if data_dict.get('image', None) is not None:
            image_file = data_dict['image']
            image = Image.open(os.path.join(self.image_folder,
                                            image_file)).convert('RGB')
            images = dynamic_preprocess(image, self.min_num, self.max_num, self._image_size)
            for image in images:
                image = self.image_processor.preprocess(
                    image, return_tensors='pt')['pixel_values'][0]
                images.append(image)
            images = torch.stack(images, dim=0)
            data_dict['pixel_values'] = images
        else:
            data_dict['pixel_values'] = torch.zeros(1, 3, self._crop_size['height'],
                                                    self._crop_size['width'])
        return data_dict
