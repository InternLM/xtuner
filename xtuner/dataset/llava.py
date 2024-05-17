# Copyright (c) OpenMMLab. All rights reserved.
import json
import logging
import os
import io
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
from mmengine.fileio import get
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import numpy as np

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
                 encode_map_fn=None,
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
                encode_map_fn=encode_map_fn,
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

    def get_image(self, path):
        if path.startswith('s3://'):
            img_bytes = get(path)
            with io.BytesIO(img_bytes) as buff:
                img = Image.open(buff).convert('RGB')
            return img
        else:
            return Image.open(path).convert('RGB')

    def __getitem__(self, index):
        data_dict = self.text_data[index]
        if data_dict.get('image', None) is not None:
            image_file = data_dict['image']
            image = self.get_image(os.path.join(self.image_folder, image_file))
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
            image = self.get_image(os.path.join(self.image_folder, image_file))
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
    def __init__(self, min_num, max_num, downsample_ratio=0.5, image_size=336, use_patch=True, *args, **kwargs):
        self.min_num = min_num
        self.max_num = max_num
        self.downsample_ratio = downsample_ratio
        self.use_patch = use_patch
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

        self.max_refetch = 1000

    def __calc_fn(self, data_dict):
        cur_len = len(data_dict['input_ids'])
        if data_dict.get('image', None) is not None:
            cur_len = len(data_dict['input_ids'])
            if data_dict.get('image', None) is not None:
                image_file = data_dict['image']
                assert 'image_wh' in data_dict
                if 'image_wh' in data_dict:
                    size = data_dict['image_wh'][0]
                else:
                    try:
                        image = self.get_image(os.path.join(self.image_folder, image_file))
                        size = image.size
                    except Exception as e:
                        print(f'Error: {e}', flush=True)
                        print_log(f'Error: {e}', logger='current')
                        size = [1, 1]
                if self.use_patch:
                    num_image_token = total_image_token(size, self.min_num, self.max_num, self._image_size,
                                                        self._patch_size)
                else:
                    num_image_token = self._patch_size * self._patch_size
                cur_len += num_image_token
                cur_len = -cur_len
        return cur_len

    @property
    def modality_length(self):
        print_log('start calculating modality length', logger='current'),
        with ThreadPoolExecutor(max_workers=16) as executor:
            length_list = list(
                tqdm(
                    executor.map(self.__calc_fn, self.text_data),
                    desc='Calculating modality length',
                    total=len(self.text_data)))
        print_log('end calculating modality length', logger='current'),
        return length_list

    def __getitem__(self, index):
        for _ in range(self.max_refetch + 1):
            data = self.prepare_data(index)
            # Broken images may cause the returned data to be None
            if data is None:
                index = self._rand_another()
                continue
            return data

    def prepare_data(self, index):
        data_dict = self.text_data[index]
        if data_dict.get('image', None) is not None:
            image_file = data_dict['image']
            try:
                image = self.get_image(os.path.join(self.image_folder, image_file))
            except Exception as e:
                print(f'Error: {e}', flush=True)
                print_log(f'Error: {e}', logger='current')
                return None
            images = dynamic_preprocess(image, self.min_num, self.max_num, self._image_size, use_patch=self.use_patch)
            for i, image in enumerate(images):
                image = self.image_processor.preprocess(
                    image, return_tensors='pt')['pixel_values'][0]
                images[i] = image
            images = torch.stack(images, dim=0)
            data_dict['pixel_values'] = images
        else:
            data_dict['pixel_values'] = torch.zeros(1, 3, self._crop_size['height'],
                                                    self._crop_size['width'])
        return data_dict

    def _rand_another(self) -> int:
        return np.random.randint(0, len(self.text_data))
