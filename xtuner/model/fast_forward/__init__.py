# Copyright (c) OpenMMLab. All rights reserved.
import types
import warnings

import torch
from mmengine import print_log
from mmengine.utils import digit_version

from .internlm_attn import internlm_attn_forward
from .llama_attn import llama_attn_forward


def dispatch_llama_attn_forward(model):
    if digit_version(torch.__version__) < digit_version('2.0.0'):
        # flash attention is only supported after pytorch2.0
        return
    print_log('dispatch llama attn forward', 'current')
    warnings.warn(
        'Due to the implementation of the PyTorch version of '
        'flash attention, even when the `output_attentions` flag is set to '
        'True, it is not possible to return the `attn_weights`.')
    for module in model.modules():
        if type(module).__name__ == 'LlamaAttention':
            module.forward = types.MethodType(llama_attn_forward, module)


def dispatch_internlm_attn_forward(model):
    if digit_version(torch.__version__) < digit_version('2.0.0'):
        # flash attention is only supported after pytorch2.0
        return
    print_log('dispatch internlm attn forward', 'current')
    warnings.warn(
        'Due to the implementation of the PyTorch version of '
        'flash attention, even when the `output_attentions` flag is set to '
        'True, it is not possible to return the `attn_weights`.')
    for module in model.modules():
        if type(module).__name__ == 'InternLMAttention':
            module.forward = types.MethodType(internlm_attn_forward, module)


def dispatch_fast_forward(model):
    dispatch_llama_attn_forward(model)
    dispatch_internlm_attn_forward(model)


__all__ = ['dispatch_fast_forward']
