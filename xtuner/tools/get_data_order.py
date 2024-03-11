# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-folder', help='Data folder')
    parser.add_argument('--save-folder', help='The folder to save data order.')
    parser.add_argument(
        '--file-type',
        default='.bin',
        help='We want to get the order of the file in this type.')
    args = parser.parse_args()
    return args


def save_data_order(data_folder, save_folder, file_type='.bin'):
    assert os.path.exists(data_folder), f'{data_folder} does not exist.'
    triples = list(os.walk(data_folder, followlinks=True))
    data_order = []
    for root, dirs, files in triples:
        dirs.sort()
        print(f'Reading {root}...')
        for fn in sorted(files):
            if fn.endswith(file_type):
                fp = os.path.join(root, fn)
                # Using relative paths so that you can get the same result
                # on different clusters
                fp = fp.replace(data_folder, '')[1:]
                data_order.append(fp)

    save_path = os.path.join(save_folder, 'data_order.txt')
    with open(save_path, 'w') as f:
        for fp in data_order:
            f.write(fp + '\n')


if __name__ == '__main__':
    args = parse_args()
    save_data_order(args.data_folder, args.save_folder, args.file_type)
