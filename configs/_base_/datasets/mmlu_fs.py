from datasets import load_dataset
from mmengine.dataset import DefaultSampler

from mmchat.datasets import process_hf_dataset
from mmchat.datasets.collators import CollatorMMLU

data_root = 'data/mmlu/'

mmlu_fs_dataset = dict(
    type=load_dataset,
    path='json',
    data_files=dict(
        val=data_root + 'five_shot_mmlu_val.json',
        test=data_root + 'five_shot_mmlu_test.json'))

val_mmlu_fs = dict(
    type=process_hf_dataset, dataset=mmlu_fs_dataset, mode='val')
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=val_mmlu_fs,
    sampler=dict(type=DefaultSampler, shuffle=False),
    collate_fn=dict(type=CollatorMMLU, tokenizer=None, max_len=2048))

test_mmlu_fs = dict(
    type=process_hf_dataset, dataset=mmlu_fs_dataset, mode='test')
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=test_mmlu_fs,
    sampler=dict(type=DefaultSampler, shuffle=False),
    collate_fn=dict(type=CollatorMMLU, tokenizer=None, max_len=2048))

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

val_evaluator = dict(type='MMLUMetric', tokenizer=None, prefix='mmlu_fs_val')
test_evaluator = dict(type='MMLUMetric', tokenizer=None, prefix='mmlu_fs_test')
