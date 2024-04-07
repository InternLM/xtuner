import os
import os.path as osp
import re
import string

import numpy as np
import pandas as pd
from mmengine.dist import (master_only)
from rich.console import Console
from rich.table import Table
from .base_eval_dataset import BaseEvalDataset

from xtuner.dataset.utils import decode_base64_to_image
from xtuner.registry import BUILDER
from mmengine.logging import print_log
from ..llava_proxy_eval_dataset import LLaVAProxyEvalDataset


class MultipleChoiceDataset(BaseEvalDataset):
    # 'mmbench', 'seedbench', 'ccbench', 'mmmu', 'scienceqa', 'ai2d'
    METAINFO: dict = dict(name='multiple_choice')

    def __init__(self, data_file, prompt_template, image_processor, tokenizer, pad_image_to_square=True,
                 use_system=False, metainfo=None, proxy_eval_dataset=dict(type=LLaVAProxyEvalDataset)):
        super().__init__(metainfo)
        self.use_system = use_system
        self.data_file = data_file
        self.df = pd.read_csv(data_file, sep='\t')
        self.split = 'dev' if 'answer' in self.df.iloc[0].keys() else 'test'
        self.has_l2_category = 'l2-category' in self.df.columns.to_list()

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
            answer = self.df.iloc[idx]['answer'] if 'answer' in self.df.iloc[
                0].keys() else None
            category = self.df.iloc[idx]['category']

            options = {
                cand: self.load_from_df(idx, cand)
                for cand in string.ascii_uppercase
                if self.load_from_df(idx, cand) is not None
            }
            options_prompt = ''
            for key, item in options.items():
                options_prompt += f'{key}. {item}\n'

            hint = self.load_from_df(idx, 'hint')
            data = {
                'img': image,
                'question': question,
                'answer': answer,
                'options': options_prompt,
                'category': category,
                'options_dict': options,
                'index': index,
                'context': hint,
                'img_id': idx
            }
            if self.has_l2_category:
                data.update({'l2-category': self.df.iloc[idx]['l2-category']})
            data_list.append(data)
        return data_list

    def __getitem__(self, idx):
        data = self.data[idx]
        data_dict = self.proxy_eval_dataset.getitem(idx, data)
        return data_dict

    def load_from_df(self, idx, key):
        if key in self.df.iloc[idx] and not pd.isna(self.df.iloc[idx][key]):
            return self.df.iloc[idx][key]
        else:
            return None

    @master_only
    def evaluate(self, results, work_dir):

        def calc_acc(df, group='category'):
            assert group in ['overall', 'category', 'l2-category']
            if group == 'overall':
                res = {'Average': np.mean(df['hit'])}
            else:
                res = {}
                abilities = list(set(df[group]))
                abilities.sort()
                for ab in abilities:
                    sub_df = df[df[group] == ab]
                    res[ab] = np.mean(sub_df['hit'])
            return res

        def eval_sub_data(sub_data, answer_map):
            lt = len(sub_data)
            for i in range(lt):
                item = sub_data.iloc[i]
                match = re.search(r'([A-D]+)', item['prediction'])
                pred = match.group(1) if match else ''
                gt = answer_map[item['index']]
                if gt != pred:
                    return 0
            return 1

        def show_result(ret_json):
            show_dict = ret_json.copy()
            table = Table(title=f' Multiple Choice ({self.data_file}) ')
            console = Console()
            table.add_column('Category', justify='left')
            table.add_column('Accuracy (%)', justify='right')
            average = show_dict.pop('Average') * 100
            table.add_row('Average', f'{average:.1f}')
            table.add_section()
            for cat_name, cat_acc in show_dict.items():
                table.add_row(cat_name, f'{cat_acc * 100:.1f}')
            with console.capture() as capture:
                console.print(table, end='')
            print_log('\n' + capture.get(), 'current')
            print_log('Note: Please be cautious if you use the results in papers, '
                      "since we don't use ChatGPT as a helper for choice "
                      'extraction', 'current')

        orig_index = [x['img_id'] for x in self.data]
        new_results = []
        for pred_dict in results:
            index = pred_dict['img_id']
            new_index = orig_index.index(index)
            filtered_rows = self.data[new_index]

            cur_result = {}
            cur_result['question'] = filtered_rows.get('question')
            cur_result.update(filtered_rows.get('options_dict'))
            cur_result['prediction'] = pred_dict['prediction']
            if filtered_rows.get('category') is not None:
                cur_result['category'] = filtered_rows.get('category')
            if filtered_rows.get('l2-category') is not None:
                cur_result['l2-category'] = filtered_rows.get('l2-category')
            cur_result['index'] = filtered_rows.get('index')
            cur_result['split'] = filtered_rows.get('split')
            cur_result['answer'] = filtered_rows.get('answer')
            new_results.append(cur_result)

        results_df = pd.DataFrame(new_results)
        with pd.ExcelWriter(osp.join(work_dir, self.results_xlsx_path), engine='openpyxl') as writer:
            results_df.to_excel(writer, index=False)

        if self.split != 'dev':
            print_log('Test set does not have answers, skip evaluation', 'current')
            return {'Average': 0}

        data = results_df.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]
        for k in data.keys():
            data[k.lower() if k not in 'ABCD' else k] = data.pop(k)

        data_main = data[data['index'] < int(1e6)]
        cate_map = {
            i: c
            for i, c in zip(self.df['index'], self.df['category'])
        }
        if self.has_l2_category:
            l2_cate_map = {
                i: c
                for i, c in zip(self.df['index'], self.df['l2-category'])
            }
        answer_map = {
            i: c
            for i, c in zip(self.df['index'], self.df['answer'])
        }

        lt = len(data_main)
        hit, tot = 0, 0
        result = {}
        for i in range(lt):
            item_main = data_main.iloc[i]
            idx = item_main['index']
            assert idx not in result
            sub_data = data[data['index'] % int(1e6) == idx]
            ret = eval_sub_data(sub_data, answer_map)
            result[idx] = ret
            hit += ret
            tot += 1

        indices = data_main['index']
        data_main = data_main.copy()
        data_main['hit'] = [result[i] for i in indices]
        main_idx = data_main['index']
        data_main['category'] = [cate_map[i] for i in main_idx]

        ret_json = calc_acc(data_main, 'overall')

        if self.has_l2_category:
            data_main['l2-category'] = [l2_cate_map[i] for i in main_idx]
            l2 = calc_acc(data_main, 'l2-category')
            ret_json.update(l2)
        else:
            leaf = calc_acc(data_main, 'category')
            ret_json.update(leaf)
        print_log('============================================', 'current')
        show_result(ret_json)
        print_log('============================================', 'current')
        print_log('Multiple Choice successfully finished evaluating' 'current')
        return ret_json
