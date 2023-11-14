# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import re
import time

import pandas as pd
import torch
import tqdm
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
from xtuner.tools.utils import get_chat_utils
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                          PROMPT_TEMPLATE)


def parse_args():
    parser = argparse.ArgumentParser(description='MMBench')
    parser.add_argument('--llm', help='Hugging Face model name or path')
    parser.add_argument('--data-path', default=None, help='data path')
    parser.add_argument('--work-dir', help='the dir to save results')
    parser.add_argument('--adapter', default=None, help='adapter name or path')
    parser.add_argument(
        '--visual-encoder', default=None, help='visual encoder name or path')
    parser.add_argument(
        '--visual-select-layer', default=-2, help='visual select layer')
    parser.add_argument(
        '--projector', default=None, help='projector name or path')
    parser.add_argument(
        '--prompt-template',
        choices=PROMPT_TEMPLATE.keys(),
        default=None,
        help='Specify a prompt template')
    parser.add_argument(
        '--system',
        default=("Answer with the option's letter from the given "
                 'choices directly.'),
        help='Specify the system text')
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
        default=10,
        help='Maximum number of new tokens allowed in generated text')
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.1,
        help='The value used to modulate the next token probabilities.')
    parser.add_argument(
        '--top-k',
        type=int,
        default=40,
        help='The number of highest probability vocabulary tokens to '
        'keep for top-k-filtering.')
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.75,
        help='If set to float < 1, only the smallest set of most probable '
        'tokens with probabilities that add up to top_p or higher are '
        'kept for generation.')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for reproducible text generation')
    args = parser.parse_args()
    return args


class MMBenchDataset(Dataset):

    def __init__(self, data_file, sys_prompt='There are several options:'):
        self.df = pd.read_csv(data_file, sep='\t')
        self.sys_prompt = sys_prompt
        self.split = 'dev' if 'answer' in self.df.iloc[0].keys() else 'test'

    def __len__(self):
        return 10
        return len(self.df)

    def __getitem__(self, idx):
        index = self.df.iloc[idx]['index']
        image = self.df.iloc[idx]['image']
        image = decode_base64_to_image(image)
        question = self.df.iloc[idx]['question']
        answer = self.df.iloc[idx]['answer'] if 'answer' in self.df.iloc[
            0].keys() else None
        catetory = self.df.iloc[idx]['category']
        l2_catetory = self.df.iloc[idx]['l2-category']

        option_candidate = ['A', 'B', 'C', 'D', 'E']
        options = {
            cand: self.load_from_df(idx, cand)
            for cand in option_candidate
            if self.load_from_df(idx, cand) is not None
        }
        options_prompt = f'{self.sys_prompt}\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'

        hint = self.load_from_df(idx, 'hint')
        data = {
            'img': image,
            'question': question,
            'answer': answer,
            'options': options_prompt,
            'category': catetory,
            'l2-category': l2_catetory,
            'options_dict': options,
            'index': index,
            'context': hint,
        }
        return data

    def load_from_df(self, idx, key):
        if key in self.df.iloc[idx] and not pd.isna(self.df.iloc[idx][key]):
            return self.df.iloc[idx][key]
        else:
            return None


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
    save_path = osp.join(save_dir, 'mmbench_result.xlsx')
    print('=======================================================')
    print(f'Dataset path: {osp.abspath(args.data_path)}\n'
          f'Results will be saved to {osp.abspath(save_path)}')
    print('=======================================================')

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
        'trust_remote_code': True
    }

    # build llm
    llm = AutoModelForCausalLM.from_pretrained(args.llm, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        args.llm, trust_remote_code=True, encode_special_tokens=True)
    if args.adapter is not None:
        llm = PeftModel.from_pretrained(
            llm, args.adapter, offload_folder=args.offload_folder)
        print(f'Load adapter from {args.adapter}')
    llm.eval()
    # build visual_encoder

    visual_encoder = CLIPVisionModel.from_pretrained(args.visual_encoder)
    processor = CLIPImageProcessor.from_pretrained(args.visual_encoder)
    visual_encoder.cuda()
    visual_encoder.eval()

    # build projector
    projector = AutoModel.from_pretrained(args.projector)
    projector.cuda()
    projector.eval()

    _, stop_criteria = get_chat_utils(llm)

    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
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
            text = data_sample['context'] + ' ' + data_sample[
                'question'] + ' ' + data_sample['options']
        else:
            text = data_sample['question'] + ' ' + data_sample['options']

        text = DEFAULT_IMAGE_TOKEN + '\n' + text

        if args.prompt_template:
            prompt_text = ''
            template = PROMPT_TEMPLATE[args.prompt_template]
            if 'SYSTEM' in template and args.system is not None:
                prompt_text += template['SYSTEM'].format(
                    system=args.system, round=1, bot_name=args.bot_name)
            prompt_text += template['INSTRUCTION'].format(
                input=text, round=1, bot_name=args.bot_name)
        else:
            prompt_text = text
        inputs = prompt_text

        image = data_sample['img'].convert('RGB')
        image = expand2square(
            image, tuple(int(x * 255) for x in processor.image_mean))
        image = processor.preprocess(
            image, return_tensors='pt')['pixel_values'][0]
        image = image.cuda().unsqueeze(0)
        visual_outputs = visual_encoder(image, output_hidden_states=True)
        pixel_values = projector(
            visual_outputs.hidden_states[args.visual_select_layer][:, 1:])

        chunk_encode = []
        for idx, chunk in enumerate(inputs.split(DEFAULT_IMAGE_TOKEN)):
            if idx == 0:
                cur_encode = tokenizer(chunk)
            else:
                cur_encode = tokenizer(chunk, add_special_tokens=False)
            chunk_encode.append(cur_encode)
        assert len(chunk_encode) == 2
        ids = []
        for idx, cur_chunk_encode in enumerate(chunk_encode):
            ids.extend(cur_chunk_encode['input_ids'])
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

        output_text = tokenizer.decode(generate_output[0])
        match = re.search(r'([A-D]+)', output_text)
        predict = match.group(1) if match else ''

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

    df = pd.DataFrame(results)
    with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    if dataset.split == 'dev':
        all_l2_category = set(df['l2-category'])
        table = Table(title=f' MMBench ({args.data_path}) ')
        console = Console()
        table.add_column('L2 Category', justify='left')
        table.add_column('Accuracy (%)', justify='right')
        for cat in all_l2_category:
            cat_df = df[df['l2-category'] == cat]
            cat_acc = sum(
                cat_df['answer'] == cat_df['prediction']) / len(cat_df) * 100
            cat_name = ' '.join(cat.split('_')).title()
            table.add_row(cat_name, f'{cat_acc:.1f}')
        table.add_section()
        average_acc = sum(df['answer'] == df['prediction']) / len(df) * 100
        table.add_row('Average', f'{average_acc:.1f}')
        with console.capture() as capture:
            console.print(table, end='')
        print('\n' + capture.get())

    print('All done!')


if __name__ == '__main__':
    main()
