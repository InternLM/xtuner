# Copyright (c) OpenMMLab. All rights reserved.
import argparse

from xtuner.configs import cfgs_name_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--pattern', default=None, help='Pattern for fuzzy matching')
    args = parser.parse_args()
    return args


def main(pattern=None):
    args = parse_args()
    configs_names = sorted(list(cfgs_name_path.keys()))
    print('==========================CONFIGS===========================')
    if args.pattern is not None:
        print(f'PATTERN: {args.pattern}')
        print('-------------------------------')
    for name in configs_names:
        if args.pattern is None or args.pattern.lower() in name.lower():
            print(name)
    print('=============================================================')


if __name__ == '__main__':
    main()
