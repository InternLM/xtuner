from datasets import load_dataset
from loguru import logger

from xtuner.dataset import process_hf_dataset
from xtuner.dataset.map_fns import template_map_fn_factory
# yapf: disable
from xtuner.rlhf.dataset.utils import (FW_fineweb_edu_map_fn,
                                       H4_hhh_alignment_map_fn,
                                       H4_summarize_map_fn,
                                       argilla_prompt_map_fn, default_map_fn,
                                       hhrlhf_map_fn, nvidia_HelpSteer_map_fn,
                                       nvidia_OpenMathInstruct_map_fn,
                                       nvidia_sft_datablend_v1_map_fn,
                                       stingning_ultrachat_map_fn)
# yapf: enable
from xtuner.utils import PROMPT_TEMPLATE


def read_hf_dataset(tokenizer,
                    path: str = None,
                    data_dir: str = None,
                    name: str = None,
                    data_files: dict = None,
                    dataset_map_fn=None,
                    max_length=8192,
                    split='train',
                    prompt_template=PROMPT_TEMPLATE.internlm_chat,
                    remove_unused_columns=False,
                    shuffle_before_pack=False,
                    pack_to_max_length=False):
    template_map_fn = template_map_fn_factory(template=prompt_template)
    dataset_org = load_dataset(
        path,
        name=name,
        data_dir=data_dir,
        data_files=data_files,
        trust_remote_code=True)
    logger.info(f'load_dataset {path}, {dataset_org}')
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
        if data_dir is not None:
            data_dir = data_dir
        elif 'helpful-base' in hf_dir:
            data_dir = 'helpful-base'
        elif 'harmless-base' in hf_dir:
            data_dir = 'harmless-base'
        logger.info(f'loading from `Anthropic/hh-rlhf`, data_dir={data_dir},'
                    ' split=`train`, map_fn=hhrlhf_map_fn...')
        dataset = read_hf_dataset(
            tokenizer=tokenizer,
            path='Anthropic/hh-rlhf',
            data_dir=data_dir,
            max_length=8192,
            split='train',
            dataset_map_fn=hhrlhf_map_fn)
    elif 'HuggingFaceH4' in hf_dir:
        if 'summarize_from_feedback' in hf_dir:
            H4_path = 'HuggingFaceH4/summarize_from_feedback'
            H4_map_fn = H4_summarize_map_fn
        elif 'hhh_alignment':
            H4_path = 'HuggingFaceH4/hhh_alignment'
            H4_map_fn = H4_hhh_alignment_map_fn
        else:
            logger.warning(f'Please specify your dataset_map_fn for {hf_dir}')
            H4_path = hf_dir
            H4_map_fn = default_map_fn
        logger.info(f'loading {H4_path}, data_dir={data_dir}, '
                    f'split=`train_prefs`, map_fn={H4_map_fn}...')
        dataset = read_hf_dataset(
            tokenizer=tokenizer,
            path=H4_path,
            data_dir=data_dir,
            max_length=8192,
            split='train_prefs',
            dataset_map_fn=H4_map_fn)
    elif 'ultrachat' in hf_dir:
        logger.info(
            f'loading from `stingning/ultrachat`, data_dir={data_dir}, '
            'split=`train`, map_fn=stingning_ultrachat_map_fn...')
        dataset = read_hf_dataset(
            tokenizer=tokenizer,
            path='stingning/ultrachat',
            data_dir=data_dir,
            max_length=8192,
            split='train',
            dataset_map_fn=stingning_ultrachat_map_fn)
    elif 'nvidia' in hf_dir:
        if 'HelpSteer' in hf_dir:
            nvidia_map_fn = nvidia_HelpSteer_map_fn
        elif 'OpenMathInstruct' in hf_dir:
            nvidia_map_fn = nvidia_OpenMathInstruct_map_fn
        elif 'sft_datablend_v1' in hf_dir:
            nvidia_map_fn = nvidia_sft_datablend_v1_map_fn
        else:
            logger.warning(f'Please specify your dataset_map_fn for {hf_dir}')
            nvidia_map_fn = default_map_fn
        logger.info(f'loading from {hf_dir}, data_dir={data_dir}, '
                    f'split=`train`, map_fn={nvidia_map_fn}...')
        dataset = read_hf_dataset(
            tokenizer=tokenizer,
            path=hf_dir,
            data_dir=data_dir,
            max_length=8192,
            split='train',
            dataset_map_fn=nvidia_map_fn)
    elif 'argilla' in hf_dir:
        if 'prompt-collective' in hf_dir:
            argilla_path = 'argilla/prompt-collective'
            argilla_map_fn = argilla_prompt_map_fn
        else:
            logger.warning(f'Please specify your dataset_map_fn for {hf_dir}')
            argilla_path = hf_dir
            argilla_map_fn = default_map_fn
        logger.info(f'loading from {argilla_path}, data_dir={data_dir}, '
                    f'split=`train`, map_fn={argilla_map_fn}...')
        dataset = read_hf_dataset(
            tokenizer=tokenizer,
            path=argilla_path,
            data_dir=data_dir,
            max_length=8192,
            split='train',
            dataset_map_fn=argilla_map_fn)
    elif 'HuggingFaceFW' in hf_dir:
        if 'fineweb-edu' in hf_dir:
            FW_path = 'HuggingFaceFW/fineweb-edu'
            FW_name = 'CC-MAIN-2024-10'
            FW_data_files = {
                'train': [
                    'data/CC-MAIN-2024-10/train-00000-of-00020.parquet',
                ]
            }
            FW_map_fn = FW_fineweb_edu_map_fn
        else:
            logger.warning(f'Please specify your dataset_map_fn for {hf_dir}')
            FW_path = hf_dir
            FW_map_fn = default_map_fn
        logger.info(f'loading from {FW_path}, name={FW_name}, '
                    f'data_files={FW_data_files}, data_dir={data_dir}, '
                    f'split=`train`, map_fn={FW_map_fn}...')
        dataset = read_hf_dataset(
            tokenizer=tokenizer,
            path=FW_path,
            name=FW_name,
            data_files=FW_data_files,
            data_dir=data_dir,
            max_length=8192,
            split='train',
            dataset_map_fn=FW_map_fn)
    else:
        try:
            logger.warning(f'Please specify your dataset_map_fn with {hf_dir}')
            dataset = read_hf_dataset(
                tokenizer=tokenizer,
                path=hf_dir,
                data_dir=data_dir,
                max_length=8192,
                split='train',
                dataset_map_fn=default_map_fn)
        except Exception as e:
            logger.error(f'{e}')
            logger.error(f'Cannot load {hf_dir}, '
                         'checkout your datapath or dataset_map_fn...')
    logger.info(f'Loaded {hf_dir}, {dataset}')
    return dataset
