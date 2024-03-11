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
from xtuner.dataset.map_fns import (DATASET_FORMAT_MAPPING,
                                    template_map_fn_factory)
from xtuner.utils import PROMPT_TEMPLATE

# ignore FutureWarning in hf datasets
warnings.simplefilter(action='ignore', category=FutureWarning)
"""
ftdp dataset:
srun -p llm_razor --quotatype=auto --gres=gpu:1 --ntasks=1 \
    --ntasks-per-node=1 --cpus-per-task=5 --kill-on-bad-exit=1 \
    python xtuner/tools/process_untokenized_datasets.py \
        --data-folder /path/to/data/folder \
        --save-folder ./processed \
        --tokenizer-path pretrained_model_name_or_path \
        --prompt-template internlm2_chat \
        --dataset-format ftdp

normal json dataset:
srun -p llm_razor --quotatype=auto --gres=gpu:1 --ntasks=1 \
    --ntasks-per-node=1 --cpus-per-task=5 --kill-on-bad-exit=1 \
    python xtuner/tools/process_untokenized_datasets.py \
    --data-folder /path/to/data/folder \
    --save-folder ./processed \
    --tokenizer-path pretrained_model_name_or_path \
    --prompt-template internlm2_chat
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-folder', help='Data folder')
    parser.add_argument('--save-folder', help='The folder to save data order.')
    parser.add_argument(
        '--tokenizer-path', help='The path to the hf tokenizer.')
    parser.add_argument(
        '--dataset-format',
        choices=list(DATASET_FORMAT_MAPPING.keys()) + ['ftdp'],
        default=None,
        help='Which dataset format is this data. The available choices are '
        f"{list(DATASET_FORMAT_MAPPING.keys()) + ['ftdp']}. ")
    parser.add_argument(
        '--prompt-template',
        choices=PROMPT_TEMPLATE.keys(),
        help='Which prompt template need to be added to the dataset. '
        f'The available choices are {PROMPT_TEMPLATE.keys()}')
    parser.add_argument(
        '--max-length', default=32768, help='Max sequence length.')
    parser.add_argument(
        '--pack-to-max-length',
        action='store_true',
        help='Whether to pack the dataset to the `max_length `.')
    parser.add_argument(
        '--file-type',
        default='.json',
        help='We want to get the order of the file in this type.')
    parser.add_argument(
        '--data-order-path',
        default=None,
        help=('The path to a txt file which contains the a list of data path.'
              ' It can be obtain by xtuner/tools/get_data_order.py script.'))
    args = parser.parse_args()
    return args


def process_one(fp,
                tokenizer,
                max_length,
                pack_to_max_length,
                dataset_map_fn=None,
                template_map_fn=None,
                is_ftdp=False):
    dataset = []
    if is_ftdp:
        with open(fp) as file:
            lines = file.readlines()
            for line in lines:
                line = ast.literal_eval(line)
                dataset.append({'messages': line})
        dataset = Dataset.from_list(dataset)
    else:
        # load formal json data
        dataset = load_dataset('json', data_files=fp)
        dataset = dataset['train']
    dataset = process(
        dataset,
        tokenizer=tokenizer,
        max_length=max_length,
        dataset_map_fn=dataset_map_fn,
        template_map_fn=template_map_fn,
        remove_unused_columns=True,
        pack_to_max_length=pack_to_max_length,
        map_num_proc=32)
    return fp, dataset


def process_untokenized_dataset(folder,
                                tokenizer,
                                max_length,
                                pack_to_max_length,
                                dataset_map_fn,
                                prompt_template,
                                data_order_path=None,
                                file_type='.json',
                                is_ftdp=False):
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
    print('All file path: ', data_order)

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    template_map_fn = ConfigDict(
        type=template_map_fn_factory, template=prompt_template)
    process_single = partial(
        process_one,
        tokenizer=tokenizer,
        max_length=max_length,
        pack_to_max_length=pack_to_max_length,
        dataset_map_fn=dataset_map_fn,
        template_map_fn=template_map_fn,
        is_ftdp=is_ftdp)
    out = pool.map(process_single, data_order)
    pool.close()
    pool.join()
    for idx, (key, dataset) in enumerate(out):
        assert data_order[idx] == key
        dataset = dataset.remove_columns('length')
        datasets_dict[str(idx)] = dataset
    datasets_dict = DatasetDict(datasets_dict)
    return datasets_dict


if __name__ == '__main__':
    args = parse_args()
    tokenizer = ConfigDict(
        type=AutoTokenizer.from_pretrained,
        pretrained_model_name_or_path=args.tokenizer_path,
        trust_remote_code=True,
        padding_side='right')

    if args.dataset_format is None:
        dataset_map_fn = None
    elif args.dataset_format == 'ftdp':
        dataset_map_fn = DATASET_FORMAT_MAPPING['openai']
    else:
        dataset_map_fn = DATASET_FORMAT_MAPPING[args.dataset_format]

    datasets_dict = process_untokenized_dataset(
        args.data_folder,
        tokenizer,
        args.max_length,
        args.pack_to_max_length,
        dataset_map_fn,
        PROMPT_TEMPLATE[args.prompt_template],
        data_order_path=args.data_order_path,
        file_type=args.file_type,
        is_ftdp=args.dataset_format == 'ftdp')
    datasets_dict.save_to_disk(args.save_folder)
