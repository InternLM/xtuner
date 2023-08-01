from mmengine.dataset import DefaultSampler

from mmchat.datasets import MOSSPluginsDataset
from mmchat.datasets.collate_fns import default_collate_fn

data_root = './data/'

data_file = 'conversations_with_tools_with_inner_instruction_no_text2image_train_all_random_meta0.5_0.1_0.01_moss_0709.jsonl'  # noqa: E501

moss_plugins = dict(
    type=MOSSPluginsDataset,
    data_file=data_root + data_file,
    bot_name=None,
    tokenizer=None,
    max_length=2048)

train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    dataset=moss_plugins,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn))
