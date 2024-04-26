from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
import torch
from PIL import Image
import os
from xtuner.tools.utils import is_cn_string
from .utils import process_anyres_image


class AnyResLLaVAProxyEvalDataset:
    def __init__(self, eval_dataset, image_grid_pinpoints):
        self.eval_ds = eval_dataset
        self.image_grid_pinpoints = image_grid_pinpoints

        # TODO: Assuming they are all squares.
        if hasattr(eval_dataset.image_processor, 'crop_size'):
            self._crop_size = eval_dataset.image_processor.crop_size
        else:
            self._crop_size = eval_dataset.image_processor.size
        self._patch_size = self._crop_size['height']
        self._shortest_edge = self._crop_size['height']

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
        if self.eval_ds.metainfo['name'] in ['mme', 'textvqa', 'gqa', 'vqa_v2', 'chartqa']:
            # MMEDataset or TextVQADataset
            image = Image.open(os.path.join(self.eval_ds.image_folder,
                                            data['image_path'])).convert('RGB')
        else:
            image = self.eval_ds.get_image(data['img']).convert('RGB')

        orig_size = image.size
        # use to remove padding
        data_dict['orig_size'] = orig_size
        image = process_anyres_image(image, self.eval_ds.image_processor,
                                     self.image_grid_pinpoints,
                                     self._patch_size, self._shortest_edge,
                                     pad_mean=tuple(int(x * 255) for x in self.eval_ds.image_processor.image_mean),
                                     orig_img_pad_to_square=self.eval_ds.pad_image_to_square)
        data_dict['pixel_values'] = image

        return data_dict
