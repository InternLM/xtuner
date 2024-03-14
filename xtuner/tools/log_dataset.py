# Copyright (c) OpenMMLab. All rights reserved.
import argparse

from mmengine.config import Config

from xtuner.registry import BUILDER


def parse_args():
    parser = argparse.ArgumentParser(description='Log processed dataset.')
    parser.add_argument('config', help='config file name or path.')
    # chose which kind of dataset style to show
    parser.add_argument(
        '--show',
        default='text',
        choices=['text', 'masked_text', 'input_ids', 'labels', 'all'],
        help='which kind of dataset style to show')
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

    if args.show == 'text' or args.show == 'all':
        print('#' * 20 + '   text   ' + '#' * 20)
        print(tokenizer.decode(train_dataset[0]['input_ids']))
    if args.show == 'masked_text' or args.show == 'all':
        print('#' * 20 + '   text(masked)   ' + '#' * 20)
        masked_text = ' '.join(
            ['[-100]' for i in train_dataset[0]['labels'] if i == -100])
        unmasked_text = tokenizer.decode(
            [i for i in train_dataset[0]['labels'] if i != -100])
        print(masked_text + ' ' + unmasked_text)
    if args.show == 'input_ids' or args.show == 'all':
        print('#' * 20 + '   input_ids   ' + '#' * 20)
        print(train_dataset[0]['input_ids'])
    if args.show == 'labels' or args.show == 'all':
        print('#' * 20 + '   labels   ' + '#' * 20)
        print(train_dataset[0]['labels'])


if __name__ == '__main__':
    main()
