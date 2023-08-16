from datasets import load_dataset
from mmengine.dataset import DefaultSampler

from xtuner.datasets import process_hf_dataset
from xtuner.datasets.collate_fns import mmlu_collate_fn

data_url = 'https://raw.githubusercontent.com/artidoro/qlora/main/data/mmlu/'

mmlu_fs_val = dict(
    type=load_dataset,
    path='json',
    data_files=dict(val=data_url + 'five_shot_mmlu_val.json'))

mmlu_fs_test = dict(
    type=load_dataset,
    path='json',
    data_files=dict(test=data_url + 'five_shot_mmlu_test.json'))

mmlu_fs_val_dataset = dict(
    type=process_hf_dataset,
    dataset=mmlu_fs_val,
    split='val',
    tokenizer=None,
    max_length=2048,
    pack_to_max_length=False,
    input_ids_with_output=False)
val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    dataset=mmlu_fs_val_dataset,
    sampler=dict(type=DefaultSampler, shuffle=False),
    collate_fn=dict(type=mmlu_collate_fn))

mmlu_fs_test_dataset = dict(
    type=process_hf_dataset,
    dataset=mmlu_fs_test,
    split='test',
    tokenizer=None,
    max_length=2048,
    pack_to_max_length=False,
    input_ids_with_output=False)
test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    dataset=mmlu_fs_test_dataset,
    sampler=dict(type=DefaultSampler, shuffle=False),
    collate_fn=dict(type=mmlu_collate_fn))

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

val_evaluator = dict(type='MMLUMetric', tokenizer=None, prefix='mmlu_fs_val')
test_evaluator = dict(type='MMLUMetric', tokenizer=None, prefix='mmlu_fs_test')
