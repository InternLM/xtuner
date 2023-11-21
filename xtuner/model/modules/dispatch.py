# Copyright (c) OpenMMLab. All rights reserved.
import logging
import types

import torch
from mmengine import print_log
from mmengine.utils import digit_version

from .baichuan import (baichuan2_norm_head_forward, baichuan_7b_attn_forward,
                       baichuan_13b_attn_forward)
from .internlm import internlm_attn_forward
from .llama import llama_attn_forward
from .yi import yi_attn_forward

SUPPORT_FLASH1 = digit_version(torch.__version__) >= digit_version('2.0.0')
SUPPORT_XFORMERS = False
SUPPORT_FLASH2 = False

try:
    import xformers.ops as xops  # pre-check # noqa: F401

    SUPPORT_XFORMERS = True
except ImportError:
    pass

try:
    from flash_attn import flash_attn_func  # pre-check # noqa: F401

    SUPPORT_FLASH2 = True
except ImportError:
    pass

SUPPORT_FLASH = SUPPORT_FLASH1 or SUPPORT_FLASH2 or SUPPORT_XFORMERS

SUPPORT_TRITON = False
try:
    import triton  # pre-check # noqa: F401
    import triton.language as tl  # pre-check # noqa: F401
    SUPPORT_TRITON = True
except ImportError:
    pass

NO_ATTN_WEIGHTS_MSG = (
    'Due to the implementation of the PyTorch version of flash attention, '
    'even when the `output_attentions` flag is set to True, it is not '
    'possible to return the `attn_weights`.')


def dispatch_llama_attn_forward(model):
    if not SUPPORT_FLASH:
        return
    print_log('dispatch llama attn forward', 'current')
    print_log(NO_ATTN_WEIGHTS_MSG, 'current', logging.WARNING)
    for module in model.modules():
        if type(module).__name__ == 'LlamaAttention':
            module.forward = types.MethodType(llama_attn_forward, module)


def dispatch_llama_rmsnorm_forward(model):
    if not SUPPORT_TRITON:
        return
    from .triton_kernels import rms_norm_forward
    print_log('dispatch llama rmsnorm forward', 'current')
    for module in model.modules():
        if type(module).__name__ == 'LlamaRMSNorm':
            module.forward = types.MethodType(rms_norm_forward, module)


def dispatch_internlm_attn_forward(model):
    if not SUPPORT_FLASH:
        return
    print_log('dispatch internlm attn forward', 'current')
    print_log(NO_ATTN_WEIGHTS_MSG, 'current', logging.WARNING)
    for module in model.modules():
        if type(module).__name__ == 'InternLMAttention':
            module.forward = types.MethodType(internlm_attn_forward, module)


def dispatch_internlm_rmsnorm_forward(model):
    if not SUPPORT_TRITON:
        return
    from .triton_kernels import rms_norm_forward
    print_log('dispatch internlm rmsnorm forward', 'current')
    for module in model.modules():
        if type(module).__name__ == 'InternLMRMSNorm':
            module.forward = types.MethodType(rms_norm_forward, module)


def dispath_baichuan2_norm_head_forward(model):
    print_log('dispatch baichuan2 NormHead forward', 'current')
    for module in model.modules():
        if type(module).__name__ == 'NormHead':
            module.forward = types.MethodType(baichuan2_norm_head_forward,
                                              module)


def dispath_baichuan_7b_attn_forward(model):
    if not SUPPORT_FLASH:
        return
    print_log('dispatch baichuan2-7B attn forward', 'current')
    print_log(NO_ATTN_WEIGHTS_MSG, 'current', logging.WARNING)
    for module in model.modules():
        if type(module).__name__ == 'Attention':
            module.forward = types.MethodType(baichuan_7b_attn_forward, module)


def dispath_baichuan_13b_attn_forward(model):
    if not SUPPORT_FLASH:
        return
    print_log('dispatch baichuan2-13B attn forward', 'current')
    print_log(NO_ATTN_WEIGHTS_MSG, 'current', logging.WARNING)
    for module in model.modules():
        if type(module).__name__ == 'BaichuanAttention':
            module.forward = types.MethodType(baichuan_13b_attn_forward,
                                              module)


def dispatch_yi_attn_forward(model):
    if not SUPPORT_FLASH:
        return
    print_log('dispatch yi attn forward', 'current')
    print_log(NO_ATTN_WEIGHTS_MSG, 'current', logging.WARNING)
    for module in model.modules():
        if type(module).__name__ == 'YiAttention':
            module.forward = types.MethodType(yi_attn_forward, module)


def dispatch_modules(model):
    model_name = model.__class__.__name__.lower()
    if 'internlm' in model_name:
        dispatch_internlm_attn_forward(model)
        dispatch_internlm_rmsnorm_forward(model)
    if 'llama' in model_name:
        dispatch_llama_attn_forward(model)
        dispatch_llama_rmsnorm_forward(model)
    if 'baichuan' in model_name:
        dispath_baichuan2_norm_head_forward(model)
        dispath_baichuan_7b_attn_forward(model)
        dispath_baichuan_13b_attn_forward(model)
    if 'yi' in model_name:
        dispatch_yi_attn_forward(model)
