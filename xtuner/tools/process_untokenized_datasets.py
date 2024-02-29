# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings

from mmengine import Config, ConfigDict
from mmengine.config.lazy import LazyObject

from xtuner.registry import BUILDER

# ignore FutureWarning in hf datasets
warnings.simplefilter(action='ignore', category=FutureWarning)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='config file name or path.')
    parser.add_argument('--save-folder', help='The folder to save data order.')
    args = parser.parse_args()
    return args


def modify_config(config, dataset_save_folder):
    dataset = ConfigDict(
        type=LazyObject('datasets', 'load_from_disk'),
        dataset_path=dataset_save_folder)
    train_dataset = ConfigDict(
        type=LazyObject('xtuner.dataset', 'process_hf_dataset'),
        dataset=dataset,
        do_dataset_tokenization=False,
        tokenizer=None,
        max_length=None,
        dataset_map_fn=None,
        template_map_fn=None,
        max_dataset_length=None,
        split=None,
        remove_unused_columns=False,
        rename_maps=[],
        pack_to_max_length=False,
        input_ids_with_output=False)
    config.train_dataloader.dataset = train_dataset
    return config


def process_untokenized_dataset(config):
    dataset = BUILDER.build(config.train_dataloader.dataset)
    return dataset


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)

    print('Start to process untokenized dataset...')
    processed_dataset = process_untokenized_dataset(cfg)
    print('Processing untokenized dataset finished.')

    processed_dataset_save_folder = args.save_folder
    if not os.path.isabs(processed_dataset_save_folder):
        processed_dataset_save_folder = os.path.join(
            os.getcwd(), processed_dataset_save_folder)
    modified_cfg = modify_config(cfg, processed_dataset_save_folder)

    print('Start to save processed dataset...')
    processed_dataset.save_to_disk(processed_dataset_save_folder)
    print(
        f'Processed dataset has been saved to {processed_dataset_save_folder}')

    cfg_folder, cfg_file_name = os.path.split(args.config)
    cfg_file_name = cfg_file_name.split('.')[0]
    cfg_file_name = f'{cfg_file_name}_modified.py'
    modified_cfg_save_path = os.path.join(cfg_folder, cfg_file_name)
    modified_cfg.dump(modified_cfg_save_path)
    print(f'Modified config has been saved to {modified_cfg_save_path}. '
          'Please use this new config for the next training phase.')
