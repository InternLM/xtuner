from mmengine.dataset import DefaultSampler

from xtuner.datasets import MOSSSFTDataset
from xtuner.datasets.collate_fns import default_collate_fn

data_root = './data/'

# Download data from https://huggingface.co/datasets/fnlp/moss-003-sft-data
data_file = 'moss-003-sft-no-tools.jsonl'

moss_sft_no_plugins = dict(
    type=MOSSSFTDataset,
    data_file=data_root + data_file,
    bot_name=None,
    tokenizer=None,
    max_length=2048)

train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    dataset=moss_sft_no_plugins,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn))
