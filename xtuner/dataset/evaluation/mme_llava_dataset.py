import os
import os.path as osp

import pandas as pd
import torch
from mmengine.dist import (master_only)
from .base_eval_dataset import BaseEvalDataset

from xtuner.dataset.utils import decode_base64_to_image, expand2square
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from xtuner.registry import BUILDER
from mmengine.logging import print_log
from PIL import Image
from .utils import YOrN_Extraction, MME_rating


class MMELLaVADataset(BaseEvalDataset):

    def __init__(self, data_file, image_folder, prompt_template, image_processor, tokenizer, pad_image_to_square=True,
                 use_system=False, for_llava_prompt=False, metainfo=None):
        super().__init__(metainfo)
        self.image_folder = image_folder
        self.use_system = use_system
        self.for_llava_prompt = for_llava_prompt
        self.data_file = data_file
        self.df = pd.read_csv(data_file, sep='\t')

        skip_noimg = True
        if skip_noimg:
            self.df = self.df[~pd.isna(self.df['image'])]

        template = prompt_template
        self.template = template

        self.tokenizer = BUILDER.build(tokenizer)
        self.image_processor = BUILDER.build(image_processor)
        self.pad_image_to_square = pad_image_to_square
        self.name = os.path.splitext(os.path.basename(data_file))[0]
        self.results_xlsx_path = os.path.splitext(os.path.basename(data_file))[0] + '-results.xlsx'
        self.data = self.load_data_list()

    def load_data_list(self):
        data_list = []
        for idx in range(len(self.df)):
            index = self.df.iloc[idx]['index']
            image = self.df.iloc[idx]['image']
            image_path = self.df.iloc[idx]['image_path']

            question = self.df.iloc[idx]['question']
            if self.for_llava_prompt:
                question = question.replace(' Please answer yes or no.',
                                            '\nAnswer the question using a single word or phrase.')

            category = self.df.iloc[idx]['category']
            answer = self.df.iloc[idx]['answer'] if 'answer' in self.df.iloc[
                0].keys() else None

            data = {
                'img': image,
                'image_path': image_path,
                'question': question,
                'answer': answer,
                'category': category,
                'index': index,
                'img_id': idx
            }
            data_list.append(data)
        return data_list

    def get_image(self, image):
        while len(image) < 16:
            image = self.df[self.df['index'] == int(image)]['image'].values
            assert len(image) == 1
            image = image[0]
        image = decode_base64_to_image(image)
        return image

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.data[idx]
        data_dict = {'img_id': data['img_id']}

        text = data['question']
        text = DEFAULT_IMAGE_TOKEN + '\n' + text

        if self.use_system:
            inputs = self.template.get('SYSTEM', '{system}').format(system='')
        else:
            inputs = ''
        inputs += self.template['INSTRUCTION'].format(input=text, round=1)

        chunk_encode = []
        for idx, chunk in enumerate(inputs.split(DEFAULT_IMAGE_TOKEN)):
            if idx == 0:
                cur_encode = self.tokenizer.encode(chunk)
            else:
                cur_encode = self.tokenizer.encode(chunk, add_special_tokens=False)
            chunk_encode.append(cur_encode)
        assert len(chunk_encode) == 2
        ids = []
        for idx, cur_chunk_encode in enumerate(chunk_encode):
            ids.extend(cur_chunk_encode)
            if idx != len(chunk_encode) - 1:
                ids.append(IMAGE_TOKEN_INDEX)
        ids = torch.tensor(ids)
        data_dict['input_ids'] = ids

        # 发现重新生成数据集后，感知部分还是对不上，推理部分对的上，暂时不清楚原因
        # image = self.get_image(data['img']).convert('RGB')
        image = Image.open(os.path.join(self.image_folder,
                                        data['image_path'])).convert('RGB')
        if self.pad_image_to_square:
            image = expand2square(
                image,
                tuple(
                    int(x * 255) for x in self.image_processor.image_mean))
        image = self.image_processor.preprocess(
            image, return_tensors='pt')['pixel_values'][0]
        data_dict['pixel_values'] = image

        return data_dict

    @master_only
    def evaluate(self, results, work_dir):
        orig_index = [x['img_id'] for x in self.data]
        new_results = []
        for pred_dict in results:
            index = pred_dict['img_id']
            new_index = orig_index.index(index)
            filtered_rows = self.data[new_index]

            cur_result = {}
            cur_result['question'] = filtered_rows.get('question')
            cur_result['prediction'] = pred_dict['prediction']
            cur_result['category'] = filtered_rows['category']
            cur_result['index'] = filtered_rows.get('index')
            cur_result['answer'] = filtered_rows.get('answer')
            cur_result['image_path'] = filtered_rows.get('image_path')
            new_results.append(cur_result)

        results_df = pd.DataFrame(new_results)
        with pd.ExcelWriter(osp.join(work_dir, self.results_xlsx_path), engine='openpyxl') as writer:
            results_df.to_excel(writer, index=False)

        data = results_df.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]

        ans_map = {k: YOrN_Extraction(v) for k, v in zip(data['index'], data['prediction'])}
        # 不使用 gpt
        data['extracted'] = [ans_map[x] for x in data['index']]
        data['score'] = (data['answer'] == data['extracted'])

        results_df = pd.DataFrame(data)
        with pd.ExcelWriter(osp.join(work_dir, self.results_xlsx_path), engine='openpyxl') as writer:
            results_df.to_excel(writer, index=False)

        score = MME_rating(data)
        print_log('============================================', 'current')
        print_log(score, 'current')
        print_log('============================================', 'current')
        print_log(f'MME YOrN_eval successfully finished evaluating', 'current')
        return score
