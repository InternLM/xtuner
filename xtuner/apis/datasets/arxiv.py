from functools import partial

from datasets import load_dataset

from xtuner.dataset import process_hf_dataset
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.map_fns import arxiv_map_fn


def arxiv_dataset(tokenizer,
                  data_file=None,
                  max_length=2048,
                  pack_to_max_length=True):
    # 1. Download data from https://kaggle.com/datasets/Cornell-University/arxiv  # noqa: E501
    # 2. Process data with `./tools/data_preprocess/arxiv.py`
    if data_file is None:
        data_file = './data/arxiv_postprocess_csAIcsCLcsCV_20200101.json'
    dataset_org = load_dataset(path='json', data_files=dict(train=data_file))
    dataset = process_hf_dataset(
        dataset=dataset_org,
        tokenizer=tokenizer,
        max_length=max_length,
        map_fn=arxiv_map_fn,
        remove_columns=[
            'id', 'submitter', 'authors', 'title', 'comments', 'journal-ref',
            'doi', 'report-no', 'categories', 'license', 'abstract',
            'versions', 'update_date', 'authors_parsed'
        ],
        shuffle_before_pack=True,
        pack_to_max_length=pack_to_max_length)

    return dataset


def arxiv_data_collator(return_hf_format=False):
    return partial(default_collate_fn, return_hf_format=return_hf_format)
