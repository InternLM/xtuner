# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import shutil

from mmengine.utils import mkdir_or_exist

from xtuner.configs import cfgs_name_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_name', help='config name')
    parser.add_argument('save_dir', help='save directory for copied config')
    args = parser.parse_args()
    return args


def add_copy_suffix(string):
    file_name, ext = os.path.splitext(string)
    return f'{file_name}_copy{ext}'


def main():
    args = parse_args()
    mkdir_or_exist(args.save_dir)
    config_path = cfgs_name_path[args.config_name]
    save_path = os.path.join(args.save_dir,
                             add_copy_suffix(os.path.basename(config_path)))
    shutil.copyfile(config_path, save_path)
    print(f'Copy to {save_path}')


if __name__ == '__main__':
    main()
