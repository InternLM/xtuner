# Copyright (c) OpenMMLab. All rights reserved.
from xtuner.configs import cfgs_name_path


def main():
    configs_names = sorted(list(cfgs_name_path.keys()))
    print('=========================ALL CONFIGS=========================')
    for name in configs_names:
        print(name)
    print('=============================================================')


if __name__ == '__main__':
    main()
