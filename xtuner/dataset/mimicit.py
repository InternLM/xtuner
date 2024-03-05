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
from .utils import decode_base64_to_image, expand2square


def combine_related_instructions(data):
    data = data['data']
    data_new = []
    ids = set(data.keys())

    for id_, qa_pair in data.items():
        messages = []
        image_ids = qa_pair['image_ids']
        for cur_id in [id_] + qa_pair['rel_ins_ids']:
            if cur_id in ids:
                cur_qa_pair = data[cur_id]
                messages.append({
                    'role': 'user',
                    'content': cur_qa_pair['instruction']
                })
                messages.append({
                    'role': 'assistant',
                    'content': cur_qa_pair['answer']
                })
                assert image_ids == cur_qa_pair['image_ids']
                ids.remove(cur_id)

        if len(messages) > 0:
            data_new.append({'messages': messages, 'image_ids': image_ids})
    return data_new


class MIMICITDataset(Dataset):

    def __init__(self,
                 tokenizer,
                 image_processor,
                 instruction_json,
                 per_image_length,
                 image_json=None,
                 use_coco_image=False,
                 coco_image_path=None,
                 max_dataset_length=None,
                 dataset_map_fn=None,
                 template_map_fn=None,
                 max_length=2048,
                 pad_image_to_square=False):
        super().__init__()
        self.use_coco_image = use_coco_image
        self.coco_image_path = coco_image_path
        self.per_image_length = per_image_length
        json_data = json.load(open(instruction_json))
        json_data = combine_related_instructions(json_data)
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
            with_image_token=True,
            per_image_length=self.per_image_length)
        if not self.use_coco_image and image_json is not None:
            self.image_json_data = json.load(open(image_json))

        if isinstance(image_processor, dict) or isinstance(
                image_processor, Config) or isinstance(image_processor,
                                                       ConfigDict):
            self.image_processor = BUILDER.build(image_processor)
        else:
            self.image_processor = image_processor
        self.pad_image_to_square = pad_image_to_square

    @property
    def modality_length(self):
        length_list = []
        for data_dict in self.text_data:
            cur_len = len(data_dict['input_ids'])
            if data_dict.get('image_ids', None) is None:
                cur_len = -cur_len
            else:
                n_images = len(data_dict['image_ids'])
                cur_len = cur_len - n_images + self.per_image_length * n_images
            length_list.append(cur_len)
        return length_list

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, index):
        data_dict = self.text_data[index]
        if data_dict.get('image_ids', None) is not None:
            image_ids = data_dict['image_ids']
            image_list = []
            for image_id in image_ids:
                if self.use_coco_image:
                    image_path = os.path.join(self.coco_image_path,
                                              image_id[-12:] + '.jpg')
                    image = Image.open(image_path).convert('RGB')
                else:
                    image = decode_base64_to_image(
                        self.image_json_data[image_id])
                if self.pad_image_to_square:
                    image = expand2square(
                        image,
                        tuple(
                            int(x * 255)
                            for x in self.image_processor.image_mean))
                image = self.image_processor.preprocess(
                    image, return_tensors='pt')['pixel_values'][0]
                image_list.append(image)
            data_dict['pixel_values'] = image_list
        else:
            crop_size = self.image_processor.crop_size
            data_dict['pixel_values'] = torch.zeros(3, crop_size['height'],
                                                    crop_size['width'])
        return data_dict
