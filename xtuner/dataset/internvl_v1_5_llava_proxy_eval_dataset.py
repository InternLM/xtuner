from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
import torch
from PIL import Image
import os
from xtuner.tools.utils import is_cn_string
from .utils import dynamic_preprocess


class InternVL_v1_5_LLaVAProxyEvalDataset:
    def __init__(self, eval_dataset, min_num, max_num):
        self.eval_ds = eval_dataset
        self.min_num = min_num
        self.max_num = max_num

        # TODO: Assuming they are all squares.
        if hasattr(eval_dataset.image_processor, 'crop_size'):
            self._crop_size = eval_dataset.image_processor.crop_size
        else:
            self._crop_size = eval_dataset.image_processor.size
        self._image_size = self._crop_size['height']

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
                # TODO prompt are different of vlmevalkit
                text = text + ("Answer with the option's letter from the "
                               'given choices directly.')
        elif self.eval_ds.metainfo['name'] in ['chartqa', 'gvqa']:
            # TODO prompt are different of vlmevalkit
            text = data['question'] + '\nAnswer the question using a single word or phrase.'
            text = DEFAULT_IMAGE_TOKEN + '\n' + text
        elif self.eval_ds.metainfo['name'] in ['hullusion', 'pope']:
            # TODO prompt are different of vlmevalkit
            text = data['question'] + '\nPlease answer yes or no.'
            text = DEFAULT_IMAGE_TOKEN + '\n' + text
        else:
            text = data['question']
            text = DEFAULT_IMAGE_TOKEN + '\n' + text

        if self.eval_ds.use_system:
            inputs = self.eval_ds.template.get('SYSTEM', '{system}').format(system='')
        else:
            inputs = ''
        inputs += self.eval_ds.template['INSTRUCTION'].format(input=text, round=1)

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
        if self.eval_ds.metainfo['name'] in ['mme', 'textvqa', 'gqa', 'vqa_v2', 'chartqa']:
            # MMEDataset or TextVQADataset
            image = Image.open(os.path.join(self.eval_ds.image_folder,
                                            data['image_path'])).convert('RGB')
        else:
            image = self.eval_ds.get_image(data['img']).convert('RGB')

        images = dynamic_preprocess(image, self.min_num, self.max_num, self._image_size)
        for i, image in enumerate(images):
            image = self.eval_ds.image_processor.preprocess(
                image, return_tensors='pt')['pixel_values'][0]
            images[i] = image
        images = torch.stack(images, dim=0)
        data_dict['pixel_values'] = images
        return data_dict
