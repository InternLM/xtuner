import bisect
import itertools
import random

import torch
from transformers.utils.import_utils import is_flash_attn_2_available
from xtuner.utils import DEFAULT_PAD_TOKEN_INDEX, IGNORE_INDEX
from torch.nn.utils.rnn import pad_sequence
import itertools
from PIL import Image
import os
from .text import TextDataset, SoftPackTextDataset, HardPackTextDataset
from torch.utils.data import Dataset
from xtuner._lite.chat import ChatMessages
from .format import OPENAI_FORMAT_MAP

def distinguish_image_or_text(item):
    if item['num_img_tokens'] > 0:
        item['modality_length'] = -item['num_tokens']
    else:
        item['modality_length'] = item['num_tokens']
    return item

class LlavaTokenizeFunction():

    def __init__(self, tokenizer, chat_template, image_dir, raw_format='llava'):
        
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.image_dir = image_dir
        self.raw_format = raw_format
        
    def __call__(self, item):
        
        formatter = OPENAI_FORMAT_MAP[self.raw_format]
        msg = ChatMessages.from_dict(formatter(item))
        tokenized = msg.tokenize(self.tokenizer, self.chat_template)
        
        if 'image_urls' in tokenized:
            image_urls = tokenized['image_urls']
            image_urls = [os.path.join(self.image_dir, url) for url in image_urls]
            num_images = len(image_urls)
            num_img_tokens = [576 for url in image_urls]
            tokenized['num_tokens'] += sum(num_img_tokens) - num_images
            tokenized['num_img_tokens'] = sum(num_img_tokens)
            tokenized['image_urls'] = image_urls
            
        return tokenized

        
        
class LlavaTokenizedDataset(TextDataset):
    
    def __init__(self, dataset, image_processor):
        super().__init__(dataset)
        self.image_processor = image_processor
        
    
    def process_tokenized_data(self, tokenized_data):
        images = []
        for url in tokenized_data['image_urls']:
            img = Image.open(url)
            images.append(img)
        
        if len(images):   
            outputs = self.image_processor(images, return_tensors='pt')
            pixel_values = outputs["pixel_values"]
        else:
            pixel_values = None
    
        data = {
            'input_ids': tokenized_data['input_ids'],
            'labels': tokenized_data['labels'],
            'pixel_values': pixel_values,
            'num_tokens': [tokenized_data['num_tokens']],
            'num_img_tokens': [tokenized_data['num_img_tokens']],
        }
        
        return data
    
    def __getitem__(self, item):
        """Returns a dict containing packed data in the given item.

        Args:
            item: An index to retrieve packed data.

        Returns:
            A dict including packed input_ids, labels, and cumulative_len.
        """
        tokenized_data = self.dataset[item]
        

        return self.process_tokenized_data(tokenized_data)



class LlavaRawDataset(LlavaTokenizedDataset):
    
    def __init__(self, dataset, image_processor, tokenize_fn):
        super().__init__(dataset, image_processor)
        
        self.tokenize_fn = tokenize_fn
    def __getitem__(self, item):
        
        raw_data = self.dataset[item]
        tokenized_data = self.tokenize_fn(raw_data)
        return self.process_tokenized_data(tokenized_data)

        
        

class SoftPackerForLlava(SoftPackTextDataset):
    
    def __init__(self, dataset, image_processor, max_length=2048, use_varlen_attn=True):
        super().__init__(dataset, max_length, use_varlen_attn)
        self.image_processor = image_processor

    def __getitem__(self, item):
        """Returns a dict containing packed data in the given item.

        Args:
            item: An index to retrieve packed data.

        Returns:
            A dict including packed input_ids, labels, and cumulative_len.
        """
        
        packed_items = self.pack_lut[item]
        
        packed_input_ids = []
        packed_labels = []
        packed_img_urls = []
        packed_num_tokens = []
        packed_num_img_tokens = []
        for i in packed_items:
            packed_input_ids.extend(self.dataset[i]['input_ids'])
            packed_labels.extend(self.dataset[i]['labels'])
            packed_img_urls.extend(self.dataset[item]['image_urls'])
            
            _num_tokens = self.dataset[i]['num_tokens']
            _num_img_tokens = self.dataset[i]['num_img_tokens']
            packed_num_tokens.append(_num_tokens)
            packed_num_img_tokens.append(_num_img_tokens)
            
        images = []
        for url in packed_img_urls:
            img = Image.open(url)
            images.append(img)
        
        if len(images):   
            outputs = self.image_processor(images, return_tensors='pt')
            pixel_values = outputs["pixel_values"]
        else:
            pixel_values = None

        packed = {
            'input_ids': packed_input_ids,
            'labels': packed_labels,
            'pixel_values': pixel_values,
            'num_tokens': packed_num_tokens,
            'num_img_tokens': packed_num_tokens
        }

        return packed
    


class LlavaCollator():
    
    def __init__(self, pack_batch=False):
        self.pack_batch = pack_batch
        
    def __call__(self, instances):
        
        pad_index = DEFAULT_PAD_TOKEN_INDEX

        input_ids = []
        labels = []
        attention_mask = []
        pixel_values = []
        num_tokens = []
        num_img_tokens = []

        for data in instances:
            input_ids.append(torch.LongTensor(data['input_ids']))
            labels.append(torch.LongTensor(data['labels']))
            num_tokens.extend(data['num_tokens'])
            num_img_tokens.extend(data['num_img_tokens'])
            if data['pixel_values'] is not None:
                pixel_values.append(data['pixel_values'])
            # breakpoint()
        attention_mask = [torch.ones_like(ids)for ids in input_ids]

        num_tokens = torch.IntTensor(num_tokens)
        num_img_tokens = torch.IntTensor(num_img_tokens)
        
        if len(instances) > 1 and self.pack_batch:
            
            input_ids = torch.cat(input_ids, dim=0).unsqueeze(0)
            labels = torch.cat(labels, dim=0).unsqueeze(0)
            attention_mask = torch.cat(attention_mask, dim=0).unsqueeze(0)
            
        elif len(instances) > 1 and not self.pack_batch:

            input_ids = pad_sequence(
                input_ids, batch_first=True, padding_value=pad_index)
            labels = pad_sequence(
                labels, batch_first=True, padding_value=IGNORE_INDEX)
            attention_mask = pad_sequence(
                attention_mask, batch_first=True, padding_value=0)
        else:
            input_ids = torch.stack(input_ids)
            labels = torch.stack(labels)
            attention_mask = torch.stack(attention_mask)
        
        if len(pixel_values) > 0:
            pixel_values = torch.cat(pixel_values, dim=0)
        else:
            pixel_values = None
        
        # TODO support sp
        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'pixel_values': pixel_values,
            'num_tokens': num_tokens, 
            'num_img_tokens': num_img_tokens,
            'attention_mask': attention_mask.bool()
        }

        return data_dict




