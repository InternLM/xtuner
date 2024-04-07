import os

import pandas as pd
from mmengine.dist import master_only
from PIL import Image

from xtuner.registry import BUILDER
from mmengine.logging import print_log
from .base_eval_dataset import BaseEvalDataset

from .utils import YOrN_Extraction, load_jsonl
from ..llava_proxy_eval_dataset import LLaVAProxyEvalDataset


def eval_func(pred_list, label_list):
    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print_log('TP\tFP\tTN\tFN\t', 'current')
    print_log(f'{TP}\t{FP}\t{TN}\t{FN}', 'current')

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print_log(f'Accuracy: {acc}', 'current')
    print_log(f'Precision: {precision}', 'current')
    print_log(f'Recall: {recall}', 'current')
    print_log(f'F1 score: {f1}', 'current')
    print_log(f'Yes ratio: {yes_ratio}', 'current')
    return f1


class POPEDataset(BaseEvalDataset):
    METAINFO: dict = dict(name='pope')

    def __init__(self, data_file, coco_val_path, prompt_template, image_processor, tokenizer, pad_image_to_square=True,
                 use_system=False, metainfo=None, proxy_eval_dataset=dict(type=LLaVAProxyEvalDataset)):
        super().__init__(metainfo)
        self.use_system = use_system
        if isinstance(data_file, str):
            data_file = [data_file]
        self.raw_data = [load_jsonl(f) for f in data_file]

        self.name = [
            os.path.splitext(os.path.basename(f))[0] for f in data_file
        ]

        self.coco_val_path = coco_val_path

        template = prompt_template
        self.template = template

        self.tokenizer = BUILDER.build(tokenizer)
        self.image_processor = BUILDER.build(image_processor)
        self.pad_image_to_square = pad_image_to_square

        self.results_xlsx_path = 'pope-results.xlsx'
        self.data = self.load_data_list()

        proxy_eval_dataset['eval_dataset'] = self
        self.proxy_eval_dataset = BUILDER.build(proxy_eval_dataset)

    def get_image(self, image):
        image = Image.open(os.path.join(self.coco_val_path, image))
        return image

    def __len__(self):
        return len(self.data)

    def load_data_list(self):
        data_list = []
        idx = 0
        for data_idx in range(len(self.raw_data)):
            for sample_idx in range(len(self.raw_data[data_idx])):
                sample = self.raw_data[data_idx][sample_idx]
                index = sample['question_id']
                image_path = sample['image']
                question = sample['text']
                answer = sample['label']
                category = self.name[data_idx]
                assert answer in ['yes', 'no']
                data = {
                    'img_id': idx,
                    'index': index,
                    'img': image_path,
                    'question': question,
                    'answer': answer,
                    'category': category
                }
                data_list.append(data)
                idx += 1
        return data_list

    def __getitem__(self, idx):
        data = self.data[idx]
        data_dict = self.proxy_eval_dataset.getitem(idx, data)
        return data_dict

    @master_only
    def evaluate(self, result, work_dir, show=True):
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
            results.append(cur_result)

        results_df = pd.DataFrame(results)
        with pd.ExcelWriter(
                os.path.join(work_dir, self.results_xlsx_path),
                engine='openpyxl') as writer:
            results_df.to_excel(writer, index=False)

        score = 0
        for sub_name in self.name:
            sub_results = [x for x in results if x['category'] == sub_name]
            pred_list = [
                int(YOrN_Extraction(x['prediction']) == 'Yes')
                for x in sub_results
            ]
            label_list = [
                int(YOrN_Extraction(x['answer']) == 'Yes') for x in sub_results
            ]
            print_log('============================================', 'current')
            print_log('Category: {}, # samples: {}'.format(sub_name,
                                                           len(sub_results)), 'current')
            cur_f1 = eval_func(pred_list, label_list)
            score += cur_f1

        score /= len(self.name)
        print_log('============================================', 'current')
        print_log(f'Average F1-score: {score}', 'current')
        print_log('============================================', 'current')
        print_log('POPE successfully finished evaluating', 'current')
        return score
