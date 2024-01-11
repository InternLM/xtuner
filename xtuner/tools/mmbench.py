# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
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
from peft import PeftModel
from rich.console import Console
from rich.table import Table
from torch.utils.data import Dataset
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)

from xtuner.dataset.utils import decode_base64_to_image, expand2square
from xtuner.model.utils import prepare_inputs_labels_for_multimodal
from xtuner.tools.utils import get_stop_criteria, is_cn_string
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                          PROMPT_TEMPLATE)

TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')


def parse_args():
    parser = argparse.ArgumentParser(description='MMBench')
    parser.add_argument(
        'model_name_or_path', help='Hugging Face model name or path')
    parser.add_argument('--data-path', default=None, help='data path')
    parser.add_argument('--work-dir', help='the dir to save results')
    parser.add_argument('--llava', default=None, help='llava name or path')
    parser.add_argument(
        '--visual-encoder', default=None, help='visual encoder name or path')
    parser.add_argument(
        '--visual-select-layer', default=-2, help='visual select layer')
    parser.add_argument(
        '--prompt-template',
        choices=PROMPT_TEMPLATE.keys(),
        default=None,
        help='Specify a prompt template')
    parser.add_argument(
        '--stop-words', nargs='+', type=str, default=[], help='Stop words')
    parser.add_argument(
        '--torch-dtype',
        default='fp16',
        choices=TORCH_DTYPE_MAP.keys(),
        help='Override the default `torch.dtype` and load the model under '
        'a specific `dtype`.')
    parser.add_argument(
        '--bits',
        type=int,
        choices=[4, 8, None],
        default=None,
        help='LLM bits')
    parser.add_argument(
        '--bot-name', type=str, default='BOT', help='Name for Bot')
    parser.add_argument(
        '--offload-folder',
        default=None,
        help='The folder in which to offload the model weights (or where the '
        'model weights are already offloaded).')
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=100,
        help='Maximum number of new tokens allowed in generated text')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for reproducible text generation')
    args = parser.parse_args()
    return args


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
        image = decode_base64_to_image(image)
        return image

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
    mkdir_or_exist(osp.abspath(save_dir))
    print('=======================================================')
    print(f'Dataset path: {osp.abspath(args.data_path)}\n'
          f'Results will be saved to {osp.abspath(save_dir)}')
    print('=======================================================')
    results_xlsx_path = osp.join(save_dir, 'mmbench_result.xlsx')
    results_json_path = osp.join(save_dir, 'mmbench_result.json')
    args_path = osp.join(save_dir, 'args.json')
    with open(args_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # build llm
    quantization_config = None
    load_in_8bit = False
    if args.bits == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4')
    elif args.bits == 8:
        load_in_8bit = True
    model_kwargs = {
        'quantization_config': quantization_config,
        'load_in_8bit': load_in_8bit,
        'device_map': 'auto',
        'offload_folder': args.offload_folder,
        'trust_remote_code': True,
        'torch_dtype': TORCH_DTYPE_MAP[args.torch_dtype]
    }

    # build llm
    llm = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                               **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        encode_special_tokens=True)
    print(f'Load LLM from {args.model_name_or_path}')

    llava_path = snapshot_download(
        repo_id=args.llava) if not osp.isdir(args.llava) else args.llava

    # build visual_encoder
    if 'visual_encoder' in os.listdir(llava_path):
        assert args.visual_encoder is None, (
            "Please don't specify the `--visual-encoder` since passed "
            '`--llava` contains a visual encoder!')
        visual_encoder_path = osp.join(llava_path, 'visual_encoder')
    else:
        assert args.visual_encoder is not None, (
            'Please specify the `--visual-encoder`!')
        visual_encoder_path = args.visual_encoder
    visual_encoder = CLIPVisionModel.from_pretrained(
        visual_encoder_path, torch_dtype=TORCH_DTYPE_MAP[args.torch_dtype])
    image_processor = CLIPImageProcessor.from_pretrained(visual_encoder_path)
    print(f'Load visual_encoder from {visual_encoder_path}')

    # load adapter
    if 'llm_adapter' in os.listdir(llava_path):
        adapter_path = osp.join(llava_path, 'llm_adapter')
        llm = PeftModel.from_pretrained(
            llm, adapter_path, offload_folder=args.offload_folder)
        print(f'Load LLM adapter from {args.llava}')
    if 'visual_encoder_adapter' in os.listdir(llava_path):
        adapter_path = osp.join(llava_path, 'visual_encoder_adapter')
        visual_encoder = PeftModel.from_pretrained(
            visual_encoder, adapter_path, offload_folder=args.offload_folder)
        print(f'Load visual_encoder adapter from {args.llava}')

    # build projector
    projector_path = osp.join(llava_path, 'projector')
    projector = AutoModel.from_pretrained(
        projector_path, torch_dtype=TORCH_DTYPE_MAP[args.torch_dtype])
    print(f'Load projector from {args.llava}')

    projector.cuda()
    projector.eval()
    visual_encoder.cuda()
    visual_encoder.eval()
    llm.eval()

    stop_words = args.stop_words
    if args.prompt_template:
        template = PROMPT_TEMPLATE[args.prompt_template]
        stop_words += template.get('STOP_WORDS', [])
    stop_criteria = get_stop_criteria(
        tokenizer=tokenizer, stop_words=stop_words)

    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )

    dataset = MMBenchDataset(args.data_path)
    results = []
    n_samples = len(dataset)
    for i in tqdm.tqdm(range(n_samples)):
        data_sample = dataset[i]
        if data_sample['context'] is not None:
            text = data_sample['context'] + '\n' + data_sample[
                'question'] + '\n' + data_sample['options']
        else:
            text = data_sample['question'] + '\n' + data_sample['options']

        text = DEFAULT_IMAGE_TOKEN + '\n' + text

        if is_cn_string(text):
            text = text + '请直接回答选项字母。'
        else:
            text = text + ("Answer with the option's letter from the "
                           'given choices directly.')

        if args.prompt_template:
            prompt_text = ''
            template = PROMPT_TEMPLATE[args.prompt_template]
            prompt_text += template['INSTRUCTION'].format(
                input=text, round=1, bot_name=args.bot_name)
        else:
            prompt_text = text
        inputs = prompt_text

        image = data_sample['img'].convert('RGB')
        image = expand2square(
            image, tuple(int(x * 255) for x in image_processor.image_mean))
        image = image_processor.preprocess(
            image, return_tensors='pt')['pixel_values'][0]
        image = image.cuda().unsqueeze(0)
        visual_outputs = visual_encoder(image, output_hidden_states=True)
        pixel_values = projector(
            visual_outputs.hidden_states[args.visual_select_layer][:, 1:])

        chunk_encode = []
        for idx, chunk in enumerate(inputs.split(DEFAULT_IMAGE_TOKEN)):
            if idx == 0:
                cur_encode = tokenizer.encode(chunk)
            else:
                cur_encode = tokenizer.encode(chunk, add_special_tokens=False)
            chunk_encode.append(cur_encode)
        assert len(chunk_encode) == 2
        ids = []
        for idx, cur_chunk_encode in enumerate(chunk_encode):
            ids.extend(cur_chunk_encode)
            if idx != len(chunk_encode) - 1:
                ids.append(IMAGE_TOKEN_INDEX)
        ids = torch.tensor(ids).cuda().unsqueeze(0)
        mm_inputs = prepare_inputs_labels_for_multimodal(
            llm=llm, input_ids=ids, pixel_values=pixel_values)

        generate_output = llm.generate(
            **mm_inputs,
            generation_config=gen_config,
            streamer=None,
            bos_token_id=tokenizer.bos_token_id,
            stopping_criteria=stop_criteria)

        predict = tokenizer.decode(
            generate_output[0], skip_special_tokens=True).strip()
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
        results.append(cur_result)

    results_df = pd.DataFrame(results)
    with pd.ExcelWriter(results_xlsx_path, engine='openpyxl') as writer:
        results_df.to_excel(writer, index=False)

    if dataset.split == 'dev':
        results_dict = dataset.eval_result(results_df, show=True)
        with open(results_json_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
    else:
        print('All done!')


if __name__ == '__main__':
    main()
