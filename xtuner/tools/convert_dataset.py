# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import math
import os.path as osp

from datasets import load_dataset
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

    dataset = load_dataset(path=args.path)
    converter = ConverterMap[args.format]
    converted = dataset.map(converter.convert, num_proc=args.num_proc)['train']

    num_shards = math.ceil(len(converted) / args.shard_size)
    digits = len(str(abs(num_shards)))
    for shard in tqdm(range(num_shards), desc='Save'):
        shard_name = f'shard-{shard:0{digits}}-of-{num_shards:0{digits}}.json'
        save_path = osp.join(args.save_dir, shard_name)

        begin = shard * args.shard_size
        end = min((shard + 1) * args.shard_size, len(converted))

        converted.select(range(begin, end)).to_json(save_path)

    print(f'Converted {len(converted)} pieces of data in {args.format} format '
          'and saved them in save_dir.')


if __name__ == '__main__':
    main()
