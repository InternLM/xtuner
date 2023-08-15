from mmengine.config import Config
from mmengine.dataset import DefaultSampler
from mmengine.registry import DATASETS
from mmengine.runner import Runner

from xtuner.datasets import MOSSSFTDataset
from xtuner.datasets.collate_fns import default_collate_fn


def moss_003_sft_plugins_dataloader(tokenizer,
                                    batch_size=1,
                                    num_workers=0,
                                    data_file=None,
                                    bot_name=None,
                                    max_length=2048):
    ds = moss_003_sft_plugins_dataset(
        tokenizer,
        data_file=data_file,
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


def moss_003_sft_plugins_dataset(tokenizer,
                                 data_file=None,
                                 bot_name=None,
                                 max_length=2048):

    # Download data from https://huggingface.co/datasets/fnlp/moss-003-sft-data
    if data_file is None:
        data_file = './data/conversations_with_tools_with_inner_instruction_no_text2image_train_all_random_meta0.5_0.1_0.01_moss_0709.jsonl'  # noqa: E501
    ds_cfg = dict(
        type=MOSSSFTDataset,
        data_file=data_file,
        bot_name=bot_name,
        tokenizer=tokenizer,
        max_length=max_length)
    ds_cfg = Config(ds_cfg)
    ds = DATASETS.build(ds_cfg)
    return ds


def moss_003_sft_plugins_data_collator():
    return default_collate_fn
