from datasets import load_dataset
from mmengine.dataset import DefaultSampler

from mmchat.datasets import process_hf_dataset
from mmchat.datasets.collate_fns import mmlu_collate_fn

data_root = 'data/mmlu/'

mmlu_zs_val_dataset = dict(
    type=load_dataset,
    path='json',
    data_files=dict(val=data_root + 'zero_shot_mmlu_val.json'))

mmlu_zs_test_dataset = dict(
    type=load_dataset,
    path='json',
    data_files=dict(test=data_root + 'zero_shot_mmlu_test.json'))

val_mmlu_zs = dict(
    type=process_hf_dataset,
    dataset=mmlu_zs_val_dataset,
    mode='val',
    tokenizer=False,
    max_length=2048,
    concat_to_max_length=False,
    predict_with_generation=True)
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=val_mmlu_zs,
    sampler=dict(type=DefaultSampler, shuffle=False),
    collate_fn=dict(type=mmlu_collate_fn))

test_mmlu_zs = dict(
    type=process_hf_dataset,
    dataset=mmlu_zs_test_dataset,
    mode='test',
    tokenizer=False,
    max_length=2048,
    concat_to_max_length=False,
    predict_with_generation=True)
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=test_mmlu_zs,
    sampler=dict(type=DefaultSampler, shuffle=False),
    collate_fn=dict(type=mmlu_collate_fn))

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

val_evaluator = dict(type='MMLUMetric', tokenizer=None, prefix='mmlu_zs_val')
test_evaluator = dict(type='MMLUMetric', tokenizer=None, prefix='mmlu_zs_test')
