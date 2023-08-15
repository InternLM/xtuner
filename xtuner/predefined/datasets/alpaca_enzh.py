from mmengine.config import Config
from mmengine.dataset import DefaultSampler
from mmengine.runner import Runner
from torch.utils.data import ConcatDataset

from xtuner.datasets.collate_fns import default_collate_fn
from .alpaca import alpaca_dataset
from .alpaca_zh import alpaca_zh_dataset


def alpaca_enzh_dataloader(tokenizer,
                           batch_size=1,
                           num_workers=0,
                           path_en='tatsu-lab/alpaca',
                           path_zh='silk-road/alpaca-data-gpt4-chinese',
                           max_length=2048,
                           concat_to_max_length=True):
    ds = alpaca_enzh_dataset(
        tokenizer,
        path_en=path_en,
        path_zh=path_zh,
        max_length=max_length,
        concat_to_max_length=concat_to_max_length)
    dl_cfg = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        dataset=ds,
        sampler=dict(type=DefaultSampler, shuffle=True),
        collate_fn=dict(type=default_collate_fn))
    dl_cfg = Config(dl_cfg)
    dl = Runner.build_dataloader(dl_cfg)
    return dl


def alpaca_enzh_dataset(tokenizer,
                        path_en='tatsu-lab/alpaca',
                        path_zh='silk-road/alpaca-data-gpt4-chinese',
                        max_length=2048,
                        concat_to_max_length=True):
    alpaca = alpaca_dataset(
        tokenizer,
        path=path_en,
        max_length=max_length,
        concat_to_max_length=concat_to_max_length)
    alpaca_zh = alpaca_zh_dataset(
        tokenizer,
        path=path_zh,
        max_length=max_length,
        concat_to_max_length=concat_to_max_length)
    ds = ConcatDataset([alpaca, alpaca_zh])
    return ds


def alpaca_enzh_data_collator():
    return default_collate_fn
