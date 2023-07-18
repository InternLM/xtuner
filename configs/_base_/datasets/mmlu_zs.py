from datasets import load_dataset
from mmengine.dataset import DefaultSampler

from mmchat.datasets import process_hf_dataset

data_root = 'data/mmlu/'

mmlu_zs_dataset = dict(
    type=load_dataset,
    path='json',
    data_files=dict(
        val=data_root + 'zero_shot_mmlu_val.json',
        test=data_root + 'zero_shot_mmlu_test.json'))

val_mmlu_zs = dict(
    type=process_hf_dataset, dataset=mmlu_zs_dataset, mode='val')
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=val_mmlu_zs,
    sampler=dict(type=DefaultSampler, shuffle=False))

test_mmlu_zs = dict(
    type=process_hf_dataset, dataset=mmlu_zs_dataset, mode='test')
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=test_mmlu_zs,
    sampler=dict(type=DefaultSampler, shuffle=False))

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

val_evaluator = dict(type='MMLUMetric', tokenizer=None, prefix='mmlu_zs_val')
test_evaluator = dict(type='MMLUMetric', tokenizer=None, prefix='mmlu_zs_test')
