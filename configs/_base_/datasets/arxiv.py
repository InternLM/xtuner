from datasets import load_dataset
from mmengine.dataset import DefaultSampler

from mmchat.datasets import process_hf_dataset
from mmchat.datasets.collate_fns import default_collate_fn
from mmchat.datasets.map_fns import arxiv_map_fn

data_root = 'data/'
json_file = 'arxiv_postprocess_csAIcsCLcsCV_20200101.json'

arxiv = dict(
    type=load_dataset,
    path='json',
    data_files=dict(train=data_root + json_file))

arxiv_dataset = dict(
    type=process_hf_dataset,
    dataset=arxiv,
    mode='train',
    tokenizer=None,
    max_length=2048,
    map_fn=arxiv_map_fn,
    remove_columns=[
        'id', 'submitter', 'authors', 'title', 'comments', 'journal-ref',
        'doi', 'report-no', 'categories', 'license', 'abstract', 'versions',
        'update_date', 'authors_parsed'
    ],
    concat_to_max_length=True)

train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=arxiv_dataset,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn))
