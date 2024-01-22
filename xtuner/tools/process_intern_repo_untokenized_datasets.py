# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import ast
import multiprocessing
import os
import warnings
from functools import partial

from datasets import Dataset, DatasetDict, load_dataset
from mmengine import ConfigDict
from transformers import AutoTokenizer

from xtuner.dataset.huggingface import process
from xtuner.dataset.map_fns import openai_map_fn, template_map_fn_factory
from xtuner.utils import PROMPT_TEMPLATE

# ignore FutureWarning in hf datasets
warnings.simplefilter(action='ignore', category=FutureWarning)
"""
srun -p llm_razor --quotatype=auto --gres=gpu:1 --ntasks=1 \
    --ntasks-per-node=1 --cpus-per-task=5 --kill-on-bad-exit=1 \
    python xtuner/tools/process_intern_repo_untokenized_datasets.py \
        --data-folder /mnt/petrelfs/share_data/caoweihan/v1_sample_with_legal_cate \
        --save-folder ./processed \
        --tokenizer-path /mnt/petrelfs/share_data/caoweihan/official_Ampere_7B_1_0_0 \
        --prompt-template internlm2_chat
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-folder', help='Data folder')
    parser.add_argument('--save-folder', help='The folder to save data order.')
    parser.add_argument(
        '--tokenizer-path', help='The path to the hf tokenizer.')
    parser.add_argument(
        '--prompt-template',
        choices=PROMPT_TEMPLATE.keys(),
        help='Which prompt template need to be added to the dataset. '
        f'The available choices are {PROMPT_TEMPLATE.keys()}')
    parser.add_argument(
        '--max_length', default=32768, help='Max sequence length.')
    parser.add_argument(
        '--file-type',
        default='.json',
        help='We want to get the order of the file in this type.')
    args = parser.parse_args()
    return args


def process_one(fp, tokenizer, max_length, template_map_fn=None):
    dataset = []
    with open(fp) as file:
        lines = file.readlines()
        for line in lines:
            line = ast.literal_eval(line)
            dataset.append({'messages': line})
    dataset = Dataset.from_list(dataset)
    dataset = process(
        dataset,
        tokenizer,
        max_length,
        dataset_map_fn=openai_map_fn,
        template_map_fn=template_map_fn,
        remove_unused_columns=True,
        pack_to_max_length=False,
        map_num_proc=32)
    # fn = fp.replace('/', '__').replace('.', '_')
    return fp, dataset


def process_intern_repo_untokenized_dataset(folder,
                                            tokenizer,
                                            max_length,
                                            prompt_template,
                                            data_order_path=None,
                                            file_type='.json'):
    assert os.path.exists(folder), f'{folder} does not exist.'
    datasets_dict = {}

    if data_order_path is not None:
        data_order = load_dataset(
            'text', data_files=data_order_path, split='train')['text']
        for i, fp in enumerate(data_order):
            data_order[i] = os.path.join(folder, fp)
    else:
        triples = list(os.walk(folder, followlinks=True))
        data_order = []
        for root, dirs, files in triples:
            dirs.sort()
            for fn in sorted(files):
                if fn.endswith(file_type):
                    fp = os.path.join(root, fn)
                    data_order.append(fp)
    print(data_order)

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    template_map_fn = ConfigDict(
        type=template_map_fn_factory, template=prompt_template)
    process_single = partial(
        process_one,
        tokenizer=tokenizer,
        max_length=max_length,
        template_map_fn=template_map_fn)
    out = pool.map(process_single, data_order)
    for idx, (key, val) in enumerate(out):
        assert data_order[idx] == key
        datasets_dict[str(idx)] = val
    datasets_dict = DatasetDict(datasets_dict)
    pool.close()
    pool.join()
    return datasets_dict


if __name__ == '__main__':
    args = parse_args()
    tokenizer = ConfigDict(
        type=AutoTokenizer.from_pretrained,
        pretrained_model_name_or_path=args.tokenizer_path,
        trust_remote_code=True,
        padding_side='right')
    datasets_dict = process_intern_repo_untokenized_dataset(
        args.data_folder,
        tokenizer,
        args.max_length,
        PROMPT_TEMPLATE[args.prompt_template],
        file_type=args.file_type)
    datasets_dict.save_to_disk(args.save_folder)
