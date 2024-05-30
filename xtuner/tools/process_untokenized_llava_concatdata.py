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
    args = parser.parse_args()
    return args


def build_llava_dataset(config):
    dataset = BUILDER.build(config)
    return dataset


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)

    datasets = cfg.train_dataloader.dataset.datasets
    for dataset_cfg in datasets:
        offline_processed_text_folder = dataset_cfg.pop('offline_processed_text_folder')
        llava_dataset = build_llava_dataset(dataset_cfg)
        text_data = llava_dataset.text_data
        text_data.save_to_disk(offline_processed_text_folder)
