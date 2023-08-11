from mmengine.config import read_base
from mmengine.dataset import DefaultSampler

from xtuner.datasets import ConcatDataset
from xtuner.datasets.collate_fns import default_collate_fn

with read_base():
    from .moss_003_sft_no_plugins import moss_sft_no_plugins
    from .moss_003_sft_plugins import moss_sft_plugins

train_dataset = dict(
    type=ConcatDataset,
    datasets_cfg=dict(
        moss_sft_no_plugins=moss_sft_no_plugins,
        moss_sft_plugins=moss_sft_plugins),
    datasets_kwargs=dict(tokenizer=None, bot_name=None))

train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    dataset=train_dataset,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn))
