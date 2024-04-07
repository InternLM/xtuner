import os
import os.path as osp

import pandas as pd
from mmengine.dist import (master_only)
from .base_eval_dataset import BaseEvalDataset

from xtuner.dataset.utils import decode_base64_to_image
from xtuner.registry import BUILDER
from mmengine.logging import print_log
from .utils import YOrN_Extraction, Hallusion_rating
from ..llava_proxy_eval_dataset import LLaVAProxyEvalDataset


class HallusionDataset(BaseEvalDataset):

    METAINFO: dict = dict(name='hullusion')

    def __init__(self, data_file, prompt_template, image_processor, tokenizer, pad_image_to_square=True,
                 use_system=False, metainfo=None, proxy_eval_dataset=dict(type=LLaVAProxyEvalDataset)):
        super().__init__(metainfo)
        self.use_system = use_system
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

        proxy_eval_dataset['eval_dataset'] = self
        self.proxy_eval_dataset = BUILDER.build(proxy_eval_dataset)

    def get_image(self, image):
        while len(image) < 16:
            image = self.df[self.df['index'] == int(image)]['image'].values
            assert len(image) == 1
            image = image[0]
        image = decode_base64_to_image(image)
        return image

    def __len__(self):
        return len(self.df)

    def load_data_list(self):
        data_list = []
        for idx in range(len(self.df)):
            index = self.df.iloc[idx]['index']
            image = self.df.iloc[idx]['image']
            image_path = self.df.iloc[idx]['image_path']
            question = self.df.iloc[idx]['question']
            category = self.df.iloc[idx]['category']
            l2_category = self.df.iloc[idx]['l2-category']
            answer = self.df.iloc[idx]['answer'] if 'answer' in self.df.iloc[
                0].keys() else None

            data = {
                'img': image,
                'image_path': image_path,
                'question': question,
                'answer': answer,
                'category': category,
                'index': index,
                'l2-category': l2_category,
                'img_id': idx
            }
            data_list.append(data)
        return data_list

    def __getitem__(self, idx):
        data = self.data[idx]
        data_dict = self.proxy_eval_dataset.getitem(idx, data)
        return data_dict

    @master_only
    def evaluate(self, result, work_dir):
        orig_index = [x['img_id'] for x in self.data]
        results = []
        for pred_dict in result:
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
            cur_result['l2-category'] = filtered_rows.get('l2-category')
            results.append(cur_result)

        results_df = pd.DataFrame(results)
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

        score = Hallusion_rating(data)
        print_log('============================================', 'current')
        print_log(score, 'current')
        print_log('============================================', 'current')
        print_log(f'YOrN_eval successfully finished evaluating', 'current')
        return score

