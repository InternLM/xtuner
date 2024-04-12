from xtuner.dataset.utils import expand2square
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
import torch
from PIL import Image
import os
from xtuner.tools.utils import is_cn_string
import numpy as np


class MiniGeminiProxyEvalDataset:
    def __init__(self, eval_dataset, image_size_aux=768):
        self.eval_ds = eval_dataset

        self._model_name = type(eval_dataset.image_processor).__name__

        if self._model_name == 'CLIPImageProcessor':
            self.crop_size_raw = eval_dataset.image_processor.crop_size.copy()
            self.eval_ds.image_processor.crop_size['height'] = image_size_aux
            self.eval_ds.image_processor.crop_size['width'] = image_size_aux
            self.eval_ds.image_processor.size['shortest_edge'] = image_size_aux
        else:
            self.aux_mean = np.array([0.48145466, 0.4578275, 0.40821073])
            self.aux_std = np.array([0.26862954, 0.26130258, 0.27577711])

    def getitem(self, idx, data):
        data_dict = {'img_id': data['img_id']}

        # 1 prepare text
        if self.eval_ds.metainfo['name'] == 'multiple_choice':
            # MultipleChoiceDataset
            if data['context'] is not None:
                text = data['context'] + '\n' + data[
                    'question'] + '\n' + data['options']
            else:
                text = data['question'] + '\n' + data['options']
            text = DEFAULT_IMAGE_TOKEN + '\n' + text

            if is_cn_string(text):
                text = text + '请直接回答选项字母。'
            else:
                text = text + ("Answer with the option's letter from the "
                               'given choices directly.')
        else:
            text = data['question']
            text = DEFAULT_IMAGE_TOKEN + '\n' + text

        if self.eval_ds.use_system:
            inputs = self.eval_ds.template.get('SYSTEM', '{system}').format(system='')
        else:
            inputs = ''
        inputs += self.eval_ds.template['INSTRUCTION'].format(input=text, round=1)

        # 2 tokenize inputs
        chunk_encode = []
        for idx, chunk in enumerate(inputs.split(DEFAULT_IMAGE_TOKEN)):
            if idx == 0:
                cur_encode = self.eval_ds.tokenizer.encode(chunk)
            else:
                cur_encode = self.eval_ds.tokenizer.encode(chunk, add_special_tokens=False)
            chunk_encode.append(cur_encode)
        assert len(chunk_encode) == 2
        ids = []
        for idx, cur_chunk_encode in enumerate(chunk_encode):
            ids.extend(cur_chunk_encode)
            if idx != len(chunk_encode) - 1:
                ids.append(IMAGE_TOKEN_INDEX)
        ids = torch.tensor(ids)
        data_dict['input_ids'] = ids

        # 3 process image
        if self.eval_ds.metainfo['name'] in ['mme', 'textvqa', 'gqa']:
            # MMEDataset or TextVQADataset
            image = Image.open(os.path.join(self.eval_ds.image_folder,
                                            data['image_path'])).convert('RGB')
        else:
            image = self.eval_ds.get_image(data['img']).convert('RGB')

        if self._model_name == 'CLIPImageProcessor':
            # clip 和 convnext 均值和方差一样，前处理相同，但是 siglip 不一致
            if self.eval_ds.pad_image_to_square:
                image = expand2square(image, tuple(int(x * 255) for x in self.eval_ds.image_processor.image_mean))

            image_aux = self.eval_ds.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            data_dict['pixel_values_aux'] = image_aux

            image = image_aux.clone()
            image = torch.nn.functional.interpolate(
                image[None], size=[self.crop_size_raw['height'], self.crop_size_raw['width']], mode='bilinear',
                align_corners=False
            )[0]
            data_dict['pixel_values'] = image
        else:
            raise NotImplementedError

        return data_dict
