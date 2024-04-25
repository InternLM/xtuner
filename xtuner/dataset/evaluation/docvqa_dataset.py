import os
import os.path as osp
from typing import Optional
import json
from mmengine.dist import master_only
from xtuner.dataset.evaluation.base_eval_dataset import BaseEvalDataset

from xtuner.registry import BUILDER
from mmengine.logging import print_log
from xtuner.dataset.llava_proxy_eval_dataset import LLaVAProxyEvalDataset
import pandas as pd
from xtuner.dataset.utils import decode_base64_to_image
import numpy as np


def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def anls_compute(groundtruth, prediction):
    gt_answer = ' '.join(groundtruth.strip().lower().split())
    det_answer = ' '.join(prediction.strip().lower().split())
    dist = levenshtein_distance(gt_answer, det_answer)
    length = max(len(groundtruth.upper()), len(prediction.upper()))
    values = 0.0 if length == 0 else float(dist) / float(length)
    return values


def hit_calculate(result, dataset_name, anls_threshold=0.5):
    if dataset_name == 'DocVQA':
        # return [1 - np.min(x['match']) >= anls_threshold for x in result]
        return [0.0 if 1 - np.min(x['match']) < anls_threshold else 1 - np.min(x['match']) for x in result]
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported for hit calculation")


class DocVQADataset(BaseEvalDataset):
    METAINFO: dict = dict(name='docvqa')

    def __init__(self, data_file, prompt_template, image_processor, tokenizer, pad_image_to_square=True,
                 anls_threshold=0.5, use_system=False, metainfo=None,
                 proxy_eval_dataset=dict(type=LLaVAProxyEvalDataset)):
        super().__init__(metainfo)
        self.anls_threshold = anls_threshold
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
            question = self.df.iloc[idx]['question']
            split = self.df.iloc[idx]['split']
            answer = self.df.iloc[idx]['answer'] if 'answer' in self.df.iloc[
                0].keys() else None

            data = {
                'img': image,
                'question': question,
                'split': split,
                'answer': answer,
                'index': index,
                'img_id': idx
            }
            data_list.append(data)
        return data_list

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
            cur_result['split'] = filtered_rows.get('split')
            cur_result['prediction'] = pred_dict['prediction']
            cur_result['index'] = filtered_rows.get('index')
            cur_result['answer'] = filtered_rows.get('answer')
            new_results.append(cur_result)

        results_df = pd.DataFrame(new_results)
        with pd.ExcelWriter(osp.join(work_dir, self.results_xlsx_path), engine='openpyxl') as writer:
            results_df.to_excel(writer, index=False)

        match = [anls_compute(results['answer'], results['prediction']) for results in new_results]

        splits = set([results['split'] for results in new_results])
        ret = dict()
        for sp in splits:
            sub = [match[i] for i, x in enumerate(new_results) if x['split'] == sp]
            hit = hit_calculate(sub, 'DocVQA')
            ret[sp] = np.mean(hit) * 100
