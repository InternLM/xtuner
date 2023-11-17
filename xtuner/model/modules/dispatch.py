# Copyright (c) OpenMMLab. All rights reserved.
import logging
import types

import torch
from mmengine import print_log
from mmengine.utils import digit_version

from .baichuan import (baichuan2_norm_head_forward, baichuan_7b_attn_forward,
                       baichuan_13b_attn_forward)
from .internlm import internlm_attn_forward, internlm2_attn_forward#, InternLMRotaryEmbedding
from .llama import llama_attn_forward
from .llama2 import llama_flash_attention, rms_norm_forward

NO_ATTN_WEIGHTS_MSG = (
    'Due to the implementation of the PyTorch version of flash attention, '
    'even when the `output_attentions` flag is set to True, it is not '
    'possible to return the `attn_weights`.')


def dispatch_llama2_rmsnorm_forward(model):
    print_log('dispatch llama2 rmsnorm forward', 'current')
    for module in model.modules():
        if type(module).__name__ == 'LlamaRMSNorm':
            module.forward = types.MethodType(rms_norm_forward, module)


def dispatch_llama2_attn_forward(model):
    print_log('dispatch llama2 attn forward', 'current')
    print_log(NO_ATTN_WEIGHTS_MSG, 'current', logging.WARNING)
    for module in model.modules():
        if type(module).__name__ == 'LlamaAttention':
            module.forward = types.MethodType(llama_flash_attention, module)

def dispatch_llama_attn_forward(model):
    if digit_version(torch.__version__) < digit_version('2.0.0'):
        # flash attention is only supported after pytorch2.0
        return
    print_log('dispatch llama attn forward', 'current')
    print_log(NO_ATTN_WEIGHTS_MSG, 'current', logging.WARNING)
    for module in model.modules():
        if type(module).__name__ == 'LlamaAttention':
            module.forward = types.MethodType(llama_attn_forward, module)


def dispatch_internlm_attn_forward(model):
    if digit_version(torch.__version__) < digit_version('2.0.0'):
        # flash attention is only supported after pytorch2.0
        return
    print_log('dispatch internlm attn forward', 'current')
    print_log(NO_ATTN_WEIGHTS_MSG, 'current', logging.WARNING)
    for module in model.modules():
        if type(module).__name__ == 'InternLMAttention':
            module.forward = types.MethodType(internlm_attn_forward, module)

def dispatch_internlm2_attn_forward(model):
    print_log('dispatch internlm2 attn forward', 'current')
    print_log(NO_ATTN_WEIGHTS_MSG, 'current', logging.WARNING)
    for module in model.modules():
        if type(module).__name__ == 'InternLMAttention':
            module.forward = types.MethodType(internlm2_attn_forward, module)


def dispatch_internlm2_rmsnorm_forward(model):
    print_log('dispatch internlm2 rmsnorm forward', 'current')
    for module in model.modules():
        if type(module).__name__ == 'InternLMRMSNorm':
            module.forward = types.MethodType(rms_norm_forward, module)


# def replace_internlm2_pos_embedding(model):
#     def replace(module):
#         for name, child in module.named_children():
#             if type(module).__name__ == 'InternLMRotaryEmbedding':
#                 dim_model = module.inv_freq.shape[0] * 2
#                 setattr(module, name, InternLMRotaryEmbedding(dim_model, module.max_seq_len_cached))
#             else:
#                 replace(child)

#     print_log('dispatch internlm2 pos embedding', 'current')
#     replace(model)


def dispath_baichuan2_norm_head_forward(model):
    print_log('dispatch baichuan2 NormHead forward', 'current')
    for module in model.modules():
        if type(module).__name__ == 'NormHead':
            module.forward = types.MethodType(baichuan2_norm_head_forward,
                                              module)


def dispath_baichuan_7b_attn_forward(model):
    if digit_version(torch.__version__) < digit_version('2.0.0'):
        # flash attention is only supported after pytorch2.0
        return
    print_log('dispatch baichuan2-7B attn forward', 'current')
    print_log(NO_ATTN_WEIGHTS_MSG, 'current', logging.WARNING)
    for module in model.modules():
        if type(module).__name__ == 'Attention':
            module.forward = types.MethodType(baichuan_7b_attn_forward, module)


def dispath_baichuan_13b_attn_forward(model):
    if digit_version(torch.__version__) < digit_version('2.0.0'):
        # flash attention is only supported after pytorch2.0
        return
    print_log('dispatch baichuan2-13B attn forward', 'current')
    print_log(NO_ATTN_WEIGHTS_MSG, 'current', logging.WARNING)
    for module in model.modules():
        if type(module).__name__ == 'BaichuanAttention':
            module.forward = types.MethodType(baichuan_13b_attn_forward,
                                              module)


def dispatch_modules(model):
    model_name = model.__class__.__name__.lower()
    if 'internlm' in model_name:
        # dispatch_internlm_attn_forward(model)
        dispatch_internlm2_attn_forward(model)
        dispatch_internlm2_rmsnorm_forward(model)
    if 'llama' in model_name:
        # dispatch_llama_attn_forward(model)
        dispatch_llama2_attn_forward(model)
        dispatch_llama2_rmsnorm_forward(model)
    if 'baichuan' in model_name:
        dispath_baichuan2_norm_head_forward(model)
        dispath_baichuan_7b_attn_forward(model)
        dispath_baichuan_13b_attn_forward(model)
