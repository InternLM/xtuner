# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import re

import torch
from transformers import StoppingCriteriaList

from xtuner.utils import StopWordStoppingCriteria


def get_base_model(model):
    if hasattr(model, 'llm'):
        model = model.llm
    if 'PeftModel' in model.__class__.__name__:
        model = model.base_model.model
    return model


def get_stop_criteria(
    tokenizer,
    stop_words=[],
):
    stop_criteria = StoppingCriteriaList()
    for word in stop_words:
        stop_criteria.append(StopWordStoppingCriteria(tokenizer, word))
    return stop_criteria


def auto_dtype_of_deepspeed_config(ds_config):
    if ds_config.get('fp16') and not ds_config.get('bf16'):
        if ds_config.get('fp16').get('enabled') == 'auto':
            ds_config['fp16']['enabled'] = torch.cuda.is_available()
    elif not ds_config.get('fp16') and ds_config.get('bf16'):
        if ds_config.get('bf16').get('enabled') == 'auto':
            ds_config['bf16']['enabled'] = torch.cuda.is_bf16_supported()
    elif ds_config.get('fp16') and ds_config.get('bf16'):
        if ds_config.get('fp16').get('enabled') == 'auto':
            ds_config['fp16']['enabled'] = torch.cuda.is_available()
        if ds_config.get('bf16').get('enabled') == 'auto':
            ds_config['bf16']['enabled'] = torch.cuda.is_bf16_supported()
        if (ds_config['fp16']['enabled'] is True
                and ds_config['bf16']['enabled'] is True):
            ds_config['fp16']['enabled'] = False
            ds_config['bf16']['enabled'] = True
    return ds_config


def is_cn_string(s):
    if re.search('[\u4e00-\u9fff]', s):
        return True
    return False


def get_seed_from_checkpoint(pth_model):
    if osp.isfile(pth_model):
        checkpoint = torch.load(pth_model, map_location='cpu')
    elif osp.isdir(pth_model):
        try:
            from deepspeed.utils.zero_to_fp32 import get_model_state_files
        except ImportError:
            raise ImportError(
                'The provided PTH model appears to be a DeepSpeed checkpoint. '
                'However, DeepSpeed library is not detected in current '
                'environment. This suggests that DeepSpeed may not be '
                'installed or is incorrectly configured. Please verify your '
                'setup.')
        filename = get_model_state_files(pth_model)[0]
        checkpoint = torch.load(filename, map_location='cpu')
    else:
        raise FileNotFoundError(f'Cannot find {pth_model}')
    return checkpoint['meta']['seed']
