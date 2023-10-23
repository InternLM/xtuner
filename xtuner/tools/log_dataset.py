# Copyright (c) OpenMMLab. All rights reserved.
import argparse

from mmengine.config import Config

from xtuner.registry import BUILDER


def parse_args():
    parser = argparse.ArgumentParser(description='Log processed dataset.')
    parser.add_argument('config', help='config file name or path.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    tokenizer = BUILDER.build(cfg.tokenizer)
    if cfg.get('framework', 'mmengine').lower() == 'huggingface':
        train_dataset = BUILDER.build(cfg.train_dataset)
    else:
        train_dataset = BUILDER.build(cfg.train_dataloader.dataset)

    print('#' * 20 + '   text   ' + '#' * 20)
    print(tokenizer.decode(train_dataset[0]['input_ids']))
    print('#' * 20 + '   input_ids   ' + '#' * 20)
    print(train_dataset[0]['input_ids'])
    print('#' * 20 + '   labels   ' + '#' * 20)
    print(train_dataset[0]['labels'])


if __name__ == '__main__':
    main()
