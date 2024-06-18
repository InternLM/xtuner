from datasets import load_dataset

from xtuner.dataset import process_hf_dataset
from xtuner.dataset.map_fns import template_map_fn_factory
from xtuner.rlhf.dataset.utils import H4_summarize_map_fn, hhrlhf_map_fn
from xtuner.utils import PROMPT_TEMPLATE


def read_hf_dataset(tokenizer,
                    path: str = None,
                    data_dir: str = None,
                    dataset_map_fn=None,
                    max_length=8192,
                    split='train',
                    prompt_template=PROMPT_TEMPLATE.internlm_chat,
                    remove_unused_columns=False,
                    shuffle_before_pack=False,
                    pack_to_max_length=False):
    # https://huggingface.co/datasets/Anthropic/hh-rlhf
    template_map_fn = template_map_fn_factory(template=prompt_template)
    dataset_org = load_dataset(path, data_dir=data_dir, trust_remote_code=True)
    dataset = process_hf_dataset(
        dataset=dataset_org,
        tokenizer=tokenizer,
        max_length=max_length,
        split=split,
        dataset_map_fn=dataset_map_fn,
        template_map_fn=template_map_fn,
        remove_unused_columns=remove_unused_columns,
        shuffle_before_pack=shuffle_before_pack,
        pack_to_max_length=pack_to_max_length)
    return dataset


def load_from_hf(hf_dir, tokenizer, data_dir=None):
    if 'Anthropic/hh-rlhf' in hf_dir:
        # train: Dataset({
        #     features: ['chosen', 'rejected'],
        #     num_rows: 160800
        # })
        # test: Dataset({
        #     features: ['chosen', 'rejected'],
        #     num_rows: 8552
        # })
        if data_dir is not None:
            data_dir = data_dir
        elif 'helpful-base' in hf_dir:
            data_dir = 'helpful-base'
        elif 'harmless-base' in hf_dir:
            data_dir = 'harmless-base'

        dataset = read_hf_dataset(
            tokenizer=tokenizer,
            path='Anthropic/hh-rlhf',
            data_dir=data_dir,
            max_length=8192,
            split='train',
            dataset_map_fn=hhrlhf_map_fn)
    if 'summarize_from_feedback' in hf_dir:
        # train_prefs: Dataset({
        #     features: ['prompt', 'chosen', 'rejected'],
        #     num_rows: 92858
        # })
        # train_sft: Dataset({
        #     features: ['prompt', 'chosen', 'rejected'],
        #     num_rows: 92858
        # })
        dataset = read_hf_dataset(
            tokenizer=tokenizer,
            path='HuggingFaceH4/summarize_from_feedback',
            data_dir=data_dir,
            max_length=8192,
            split='train_prefs',
            dataset_map_fn=H4_summarize_map_fn)
    return dataset
