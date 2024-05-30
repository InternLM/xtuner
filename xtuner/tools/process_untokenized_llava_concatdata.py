# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import warnings

from mmengine import Config
import numpy as np

from xtuner.registry import BUILDER
from tqdm import tqdm
from mmengine.logging import MMLogger

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

    logger = MMLogger.get_instance(
        name='xtuner',
        log_file='benchmark_test.log')

    datasets = cfg.train_dataloader.dataset.datasets
    for dataset_cfg in tqdm(datasets):
        offline_processed_text_folder = dataset_cfg.pop('offline_processed_text_folder')
        logger.info('=================================================================')
        logger.info(f'offline_processed_text_folder: {offline_processed_text_folder}')
        try:
            llava_dataset = build_llava_dataset(dataset_cfg)
            text_data = llava_dataset.text_data

            length_list = text_data['length']
            length_np = np.abs(length_list)
            min_, max_, mid_ = np.min(length_np), np.max(length_np), np.median(length_np)
            logger.info(f'token len({length_np.shape[0]}): max: {max_}, min: {min_}, mid: {mid_}')
            try:
                image_wh_list = text_data['image_wh']
                new_list = []
                for d in image_wh_list:
                    if d is not None:
                        if isinstance(d[0], list):
                            new_list.append(d[0])
                        else:
                            new_list.append(d)
                new_list = np.array(new_list).reshape(-1, 2)
                row_sums = np.sum(new_list, axis=1)
                max_idx = np.argmax(row_sums)
                min_idx = np.argmin(row_sums)
                mid_idx = np.argsort(row_sums)[len(row_sums) // 2]
                max_value = new_list[max_idx]
                min_value = new_list[min_idx]
                mid_value = new_list[mid_idx]
                logger.info(f'Image wh: max: {max_value}, min: {min_value}, mid: {mid_value}\n')

            except Exception as e:
                logger.error(f'=======Error: {e}')

            text_data.save_to_disk(offline_processed_text_folder)
        except Exception as e:
            logger.error(f'--------Error: {e}')
            raise NotImplementedError
