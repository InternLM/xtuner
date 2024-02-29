# Copyright (c) OpenMMLab. All rights reserved.
from xtuner.dataset.map_fns import DATASET_FORMAT_MAPPING


def main():
    dataset_format = DATASET_FORMAT_MAPPING.keys()
    print('======================DATASET_FORMAT======================')
    for format in dataset_format:
        print(format)
    print('==========================================================')


if __name__ == '__main__':
    main()
