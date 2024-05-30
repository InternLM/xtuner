# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import warnings

from mmengine import Config

from xtuner.registry import BUILDER

# ignore FutureWarning in hf datasets
warnings.simplefilter(action='ignore', category=FutureWarning)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='config file name or path.')
    parser.add_argument('--save-folder', help='The folder to save data order.')
    args = parser.parse_args()
    return args


def build_llava_dataset(config):
    dataset = BUILDER.build(config)
    return dataset


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)

    datasets = cfg.train_dataloader.datasets
    for dataset_cfg in datasets:
        llava_dataset = build_llava_dataset(dataset_cfg)
        text_data = llava_dataset.text_data
        variable_name = [k for k, v in locals().items() if v == dataset_cfg][0]
        save_folder = args.save_folder + f'/{variable_name}'
        text_data.save_to_disk(save_folder)
