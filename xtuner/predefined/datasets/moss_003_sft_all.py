from mmengine.config import Config
from mmengine.dataset import DefaultSampler
from mmengine.runner import Runner
from torch.utils.data import ConcatDataset

from xtuner.datasets.collate_fns import default_collate_fn
from .moss_003_sft_no_plugins import moss_003_sft_no_plugins_dataset
from .moss_003_sft_plugins import moss_003_sft_plugins_dataset


def moss_003_sft_dataloader(tokenizer,
                            batch_size=1,
                            num_workers=0,
                            plugins_data_file=None,
                            no_plugins_data_file=None,
                            bot_name=None,
                            max_length=2048):
    ds = moss_003_sft_dataset(
        tokenizer,
        plugins_data_file=plugins_data_file,
        no_plugins_data_file=no_plugins_data_file,
        bot_name=bot_name,
        max_length=max_length)
    dl_cfg = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        dataset=ds,
        sampler=dict(type=DefaultSampler, shuffle=True),
        collate_fn=dict(type=default_collate_fn))
    dl_cfg = Config(dl_cfg)
    dl = Runner.build_dataloader(dl_cfg)
    return dl


def moss_003_sft_dataset(tokenizer,
                         plugins_data_file=None,
                         no_plugins_data_file=None,
                         bot_name=None,
                         max_length=2048):
    plugins = moss_003_sft_plugins_dataset(
        tokenizer,
        data_file=plugins_data_file,
        bot_name=bot_name,
        max_length=max_length)
    no_plugins = moss_003_sft_no_plugins_dataset(
        tokenizer,
        data_file=no_plugins_data_file,
        bot_name=bot_name,
        max_length=max_length)
    ds = ConcatDataset([plugins, no_plugins])
    return ds


def moss_003_sft_data_collator():
    return default_collate_fn
