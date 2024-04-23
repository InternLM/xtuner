# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import math
import os
import os.path as osp
import re
import string
import time

import numpy as np
import pandas as pd
import torch
import tqdm
from huggingface_hub import snapshot_download
from mmengine import mkdir_or_exist
from mmengine.dist import (collect_results, get_dist_info, get_rank, init_dist,
                           master_only)
from mmengine.utils.dl_utils import set_multi_processing
from peft import PeftModel
from rich.console import Console
from rich.table import Table
from torch.utils.data import Dataset
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)

from xtuner.dataset.utils import decode_base64_to_image, expand2square
from xtuner.model.utils import LoadWoInit, prepare_inputs_labels_for_multimodal
from xtuner.tools.utils import get_stop_criteria, is_cn_string
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                          PROMPT_TEMPLATE)
from tqdm import tqdm
TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')


def parse_args():
    parser = argparse.ArgumentParser(description='MMBench')
    parser.add_argument('--data-path', default=None, help='data path')
    parser.add_argument('--work-dir', help='the dir to save results')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for reproducible text generation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    args = parser.parse_args()
    return args


@master_only
def master_print(msg):
    print(msg)


class MMBenchDataset(Dataset):
    ABBRS = {
        'coarse_perception': 'CP',
        'finegrained_perception (instance-level)': 'FP-S',
        'finegrained_perception (cross-instance)': 'FP-C',
        'logic_reasoning': 'LR',
        'relation_reasoning': 'RR',
        'attribute_reasoning': 'AR',
        'sketch_reasoning': 'Sketch Reasoning',
        'scenery_building': 'Scenery & Building',
        'food_clothes': 'Food & Clothes',
        'historical_figure': 'Historical Figure',
        'traditional_show': 'Traditional Show',
        'calligraphy_painting': 'Calligraphy Painting',
        'cultural_relic': 'Cultural Relic'
    }

    def __init__(self, data_file):
        self.data_file = data_file
        self.df = pd.read_csv(data_file, sep='\t')
        self.split = 'dev' if 'answer' in self.df.iloc[0].keys() else 'test'
        self.has_l2_category = 'l2-category' in self.df.columns.to_list()

    def get_image(self, image):
        while len(image) < 16:
            image = self.df[self.df['index'] == int(image)]['image'].values
            assert len(image) == 1
            image = image[0]
        # image = decode_base64_to_image(image)
        return f'data:image/jpeg;base64,{image}'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        index = self.df.iloc[idx]['index']
        image = self.df.iloc[idx]['image']
        image = self.get_image(image)
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
        }
        if self.has_l2_category:
            data.update({'l2-category': self.df.iloc[idx]['l2-category']})
        return data

    def load_from_df(self, idx, key):
        if key in self.df.iloc[idx] and not pd.isna(self.df.iloc[idx][key]):
            return self.df.iloc[idx][key]
        else:
            return None

    @master_only
    def eval_result(self, result_df, show=True):

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
                    ab = self.ABBRS[ab] if ab in self.ABBRS else ab
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
            table = Table(title=f' MMBench ({self.data_file}) ')
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
            print('\n' + capture.get())
            print('Note: Please be cautious if you use the results in papers, '
                  "since we don't use ChatGPT as a helper for choice "
                  'extraction')

        data = result_df.sort_values(by='index')
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
        if show:
            show_result(ret_json)
        return ret_json


def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    if args.launcher != 'none':
        set_multi_processing(distributed=True)
        init_dist(args.launcher)

        rank, world_size = get_dist_info()
        torch.cuda.set_device(rank)
    else:
        rank = 0
        world_size = 1

    from xtuner.types import ChatTemplate
    chat_template = ChatTemplate(
        system='<|im_start|>system\n{system}<|im_end|>\n',
        user='<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n',
        assistant='{assistant}<|im_end|>\n',
        stop_words=['<|im_end|>'],
        image_token='<image>',
        function_call='{assistant}<|action_start|><|plugin|>\n{function_call}<|action_end|><|im_end|>\n',  # noqa: E501, E251
        function_result='<|im_start|>environment name=<|plugin|>\n{function_result}<|im_end|>\n<|im_start|>assistant\n',  # noqa: E501, E251
        functions='<|im_start|>system name=<|plugin|>\n{functions}<|im_end|>\n'
    )
    
    from xtuner.chat.backend import HFBackend, LMDeployBackend
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from xtuner.chat.backend import VisionEncoderForDeploy
    from xtuner.chat.conversation import Conversation
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # llm = AutoModelForCausalLM.from_pretrained('/mnt/petrelfs/share_data/caoweihan/llava', trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained('/mnt/petrelfs/share_data/caoweihan/llava', trust_remote_code=True)
    vision_tower = VisionEncoderForDeploy(
        model_name_or_path = 'openai/clip-vit-large-patch14-336',
        # adapter_name_or_path= '/mnt/petrelfs/share_data/linzhihao/model/models--xtuner--llava-internlm2-7b/snapshots/f363b45ce4787bd0a21d43ed724a70ee40ce69b2/visual_encoder_adapter',
        projector_name_or_path='/mnt/petrelfs/share_data/linzhihao/model/models--xtuner--llava-internlm2-7b/snapshots/f363b45ce4787bd0a21d43ed724a70ee40ce69b2/projector'
        
    )
    # backend = HFBackend(
    #     chat_template, 
    #     llm, 
    #     tokenizer,
    #     vision_tower,
    # )
    backend = LMDeployBackend(
        chat_template, 
        '/mnt/petrelfs/share_data/linzhihao/model/models--internlm--internlm2-chat-7b/snapshots/2292b86b21cb856642782cebed0a453997453b1f/',
        vision_tower
    )
    
    # mm_backend = HFBackend(
    #     chat_template, 
    #     '/mnt/petrelfs/share_data/linzhihao/model/models--internlm--internlm2-chat-7b/snapshots/2292b86b21cb856642782cebed0a453997453b1f/',
    #     '/mnt/petrelfs/share_data/linzhihao/model/models--xtuner--llava-internlm2-7b/snapshots/f363b45ce4787bd0a21d43ed724a70ee40ce69b2/llm_adapter',
    #     'openai/clip-vit-large-patch14-336',
    #     '/mnt/petrelfs/share_data/linzhihao/model/models--xtuner--llava-internlm2-7b/snapshots/f363b45ce4787bd0a21d43ed724a70ee40ce69b2/projector',
    #     '/mnt/petrelfs/share_data/linzhihao/model/models--xtuner--llava-internlm2-7b/snapshots/f363b45ce4787bd0a21d43ed724a70ee40ce69b2/visual_encoder_adapter',
    # )
    
    # work_dir
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        save_dir = args.work_dir
    else:
        # use config filename as default work_dir
        save_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.data_path))[0])
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    save_dir = osp.join(save_dir, timestamp)

    if rank == 0:
        mkdir_or_exist(osp.abspath(save_dir))
        print('=======================================================')
        print(f'Dataset path: {osp.abspath(args.data_path)}\n'
              f'Results will be saved to {osp.abspath(save_dir)}')
        print('=======================================================')

        args_path = osp.join(save_dir, 'args.json')
        with open(args_path, 'w', encoding='utf-8') as f:
            json.dump(args.__dict__, f, indent=2)

    results_xlsx_path = osp.join(save_dir, 'mmbench_result.xlsx')
    results_json_path = osp.join(save_dir, 'mmbench_result.json')

    dataset = MMBenchDataset(args.data_path)

    results = []
    n_samples = len(dataset)
    per_rank_samples = math.ceil(n_samples / world_size)

    per_rank_ids = range(per_rank_samples * rank,
                         min(n_samples, per_rank_samples * (rank + 1)))
    
    
    def _eval_fn(data_sample):
        if data_sample['context'] is not None:
            text = data_sample['context'] + '\n' + data_sample[
                'question'] + '\n' + data_sample['options']
        else:
            text = data_sample['question'] + '\n' + data_sample['options']

        
        from xtuner.types.messages import ChatMessages, ChatMsg, TextContentItem, ImageContentItem
        content = [
            TextContentItem(text=text),
            ImageContentItem(image_url=data_sample['img'])
        ]

        if is_cn_string(text):
            text = '请直接回答选项字母。'
        else:
            text = ("Answer with the option's letter from the "
                           'given choices directly.')

        content.append(TextContentItem(text=text))
        
        conv = Conversation(backend)        

        predict = conv.chat(content)
        # breakpoint()
        cur_result = {}
        cur_result['question'] = data_sample.get('question')
        cur_result.update(data_sample.get('options_dict'))
        cur_result['prediction'] = predict
        if data_sample.get('category') is not None:
            cur_result['category'] = data_sample.get('category')
        if data_sample.get('l2-category') is not None:
            cur_result['l2-category'] = data_sample.get('l2-category')
        cur_result['index'] = data_sample.get('index')
        cur_result['split'] = data_sample.get('split')
        cur_result['answer'] = data_sample.get('answer')
        return cur_result
    
    from concurrent.futures import ThreadPoolExecutor
    
    # with ThreadPoolExecutor(max_workers=16) as executor:
    #     results = list(
    #         tqdm(
    #             executor.map(_eval_fn, dataset),
    #             desc='Evalutate',
    #             total=len(dataset)))
    # from xtuner.types.chat import HybridChatMessages
    msg_list = []
    
    for i in tqdm(per_rank_ids, desc=f'Rank {rank}'):
        data_sample = dataset[i]
        if data_sample['context'] is not None:
            text = data_sample['context'] + '\n' + data_sample[
                'question'] + '\n' + data_sample['options']
        else:
            text = data_sample['question'] + '\n' + data_sample['options']

        
        from xtuner.types.messages import ChatMessages, ChatMsg, TextContentItem, ImageContentItem
        content = [
            ImageContentItem(image_url=data_sample['img']),
            TextContentItem(text=text),
        ]

        if is_cn_string(text):
            text = '请直接回答选项字母。'
        else:
            text = ("Answer with the option's letter from the "
                           'given choices directly.')

        content.append(TextContentItem(text=text))
        # msg_list.append(content)
        msg = ChatMsg(role='user', content=content)
        msg_list.append(ChatMessages(messages=[msg]))

        # conv = Conversation(backend)        

        # predict = conv.chat(content)
        # breakpoint()
        cur_result = {}
        cur_result['question'] = data_sample.get('question')
        cur_result.update(data_sample.get('options_dict'))
        # cur_result['prediction'] = predict
        if data_sample.get('category') is not None:
            cur_result['category'] = data_sample.get('category')
        if data_sample.get('l2-category') is not None:
            cur_result['l2-category'] = data_sample.get('l2-category')
        cur_result['index'] = data_sample.get('index')
        cur_result['split'] = data_sample.get('split')
        cur_result['answer'] = data_sample.get('answer')
        results.append(cur_result)
    
    # def _map_fn(msg):
    #     # conv = Conversation(backend)
    #     return backend.chat(msg)
        
    # with ThreadPoolExecutor(max_workers=16) as executor:
    #     predictions = list(
    #         tqdm(
    #             executor.map(_map_fn, msg_list),
    #             desc='Evalutate',
    #             total=len(msg_list)))
    predictions = backend.batch_infer(msg_list)
    # # breakpoint()
    for pred, res in zip(predictions, results):
        res['prediction'] = pred

    results = collect_results(results, n_samples)

    if get_rank() == 0:

        results_df = pd.DataFrame(results)
        with pd.ExcelWriter(results_xlsx_path, engine='openpyxl') as writer:
            results_df.to_excel(writer, index=False)

        if dataset.split == 'dev':
            results_dict = dataset.eval_result(results_df, show=True)
            with open(results_json_path, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, indent=2)
        else:
            print('All done!')


if __name__ == '__main__':

    main()
