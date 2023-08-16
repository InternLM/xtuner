# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import shutil

from mmengine.utils import mkdir_or_exist

import xtuner.configs as configs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_name', help='config name')
    parser.add_argument('save_dir', help='save directory for copied config')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    mkdir_or_exist(args.save_dir)
    configs_name_path = {
        name: configs.__dict__[name].__file__
        for name in configs.__dict__ if not name.startswith('__')
        and configs.__dict__[name].__file__ is not None
    }
    config_path = configs_name_path[args.config_name]
    save_path = os.path.join(args.save_dir, os.path.basename(config_path))
    shutil.copyfile(config_path, save_path)
    print(f'Copy to {save_path}')


if __name__ == '__main__':
    main()
