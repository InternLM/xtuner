# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import Config
from mmengine.runner import Runner


def get_train_dataloader(dataset_cfg_path, tokenizer):
    dataset_cfg = Config.fromfile(dataset_cfg_path)
    if dataset_cfg.train_dataloader.dataset.type.__name__ == 'ConcatDataset':
        dataset_cfg.train_dataloader.dataset.datasets_kwargs.tokenizer = \
            tokenizer
    else:
        dataset_cfg.train_dataloader.dataset.tokenizer = tokenizer
    train_dataloader = Runner.build_dataloader(dataset_cfg.train_dataloader)
    return train_dataloader
