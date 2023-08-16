# Copyright (c) OpenMMLab. All rights reserved.
import xtuner.configs as configs


def main():
    configs_names = sorted(name for name in configs.__dict__
                           if name.islower() and not name.startswith('__')
                           and configs.__dict__[name].__file__ is not None)
    print('=========================ALL CONFIGS=========================')
    for name in configs_names:
        print(name)
    print('=============================================================')


if __name__ == '__main__':
    main()
