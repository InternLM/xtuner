# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import math
import os.path as osp
from datetime import datetime

from datasets import load_dataset
from mmengine import mkdir_or_exist
from tqdm import tqdm

from xtuner.dataset.converters import ConverterMap


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Path or name of the dataset.')
    parser.add_argument(
        'format', choices=ConverterMap.keys(), help='Source dataset format.')
    parser.add_argument(
        '--save-dir', help='Save dir of the converted dataset.')
    parser.add_argument(
        '--data-dir', help='Reference all the files in a directory')
    parser.add_argument('--data-files', help='Path(s) to source data file(s).')
    parser.add_argument(
        '--shard-size',
        type=int,
        default=20000,
        help='The number of data in a sliced JSON file.')
    parser.add_argument(
        '--num-proc',
        type=int,
        default=8,
        help='Multithreaded data processing')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    dataset = load_dataset(
        path=args.path, data_dir=args.data_dir, data_files=args.data_files)
    converter = ConverterMap[args.format]
    converted = dataset.map(converter.convert, num_proc=args.num_proc)['train']

    num_shards = math.ceil(len(converted) / args.shard_size)
    num_digits = len(str(abs(num_shards)))
    timestamp = datetime.now().strftime('%Y%m%d%H%M')
    mkdir_or_exist(args.save_dir)

    for i in tqdm(range(num_shards), desc='Save'):
        _t = timestamp
        _d = num_digits
        shard_name = f'{_t}-shard-{i+1:0{_d}}-of-{num_shards:0{_d}}.json'
        save_path = osp.join(args.save_dir, shard_name)

        begin = i * args.shard_size
        end = min((i + 1) * args.shard_size, len(converted))

        shard = converted.select(range(begin, end)).to_list()
        with open(save_path, 'w') as f:
            json.dump(shard, f)

    print(f'Converted {len(converted)} pieces of data in {args.format} format '
          f'and saved them in {args.save_dir}.')


if __name__ == '__main__':
    main()
