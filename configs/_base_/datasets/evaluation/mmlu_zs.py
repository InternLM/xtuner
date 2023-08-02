from datasets import load_dataset
from mmengine.dataset import DefaultSampler

from mmchat.datasets import process_hf_dataset
from mmchat.datasets.collate_fns import mmlu_collate_fn

data_root = './data/mmlu/'

# Download data from https://github.com/artidoro/qlora/tree/main/data/mmlu
mmlu_zs_val = dict(
    type=load_dataset,
    path='json',
    data_files=dict(val=data_root + 'zero_shot_mmlu_val.json'))

mmlu_zs_test = dict(
    type=load_dataset,
    path='json',
    data_files=dict(test=data_root + 'zero_shot_mmlu_test.json'))

mmlu_zs_val_dataset = dict(
    type=process_hf_dataset,
    dataset=mmlu_zs_val,
    split='val',
    tokenizer=False,
    max_length=2048,
    concat_to_max_length=False,
    input_with_labels=False)
val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    dataset=mmlu_zs_val_dataset,
    sampler=dict(type=DefaultSampler, shuffle=False),
    collate_fn=dict(type=mmlu_collate_fn))

mmlu_zs_test_dataset = dict(
    type=process_hf_dataset,
    dataset=mmlu_zs_test,
    split='test',
    tokenizer=False,
    max_length=2048,
    concat_to_max_length=False,
    input_with_labels=False)
test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    dataset=mmlu_zs_test_dataset,
    sampler=dict(type=DefaultSampler, shuffle=False),
    collate_fn=dict(type=mmlu_collate_fn))

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

val_evaluator = dict(type='MMLUMetric', tokenizer=None, prefix='mmlu_zs_val')
test_evaluator = dict(type='MMLUMetric', tokenizer=None, prefix='mmlu_zs_test')
