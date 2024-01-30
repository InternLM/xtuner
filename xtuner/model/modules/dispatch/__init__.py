# Copyright (c) OpenMMLab. All rights reserved.
import logging
import types

import torch
from mmengine import print_log
from mmengine.utils import digit_version

from .baichuan import (baichuan2_norm_head_forward, baichuan_7b_attn_forward,
                       baichuan_13b_attn_forward)
from .yi import yi_attn_forward

SUPPORT_FLASH1 = digit_version(torch.__version__) >= digit_version('2.0.0')
SUPPORT_FLASH2 = False

try:
    from flash_attn import flash_attn_func  # pre-check # noqa: F401

    SUPPORT_FLASH2 = True
except ImportError:
    pass

SUPPORT_FLASH = SUPPORT_FLASH1 or SUPPORT_FLASH2

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


def dispatch_llama_attn_forward(model, use_varlen_attn):
    if use_varlen_attn:
        assert SUPPORT_FLASH2 and SUPPORT_TRITON, \
            'flash_attn and triton is required if you want to use varlen_attn.'
    elif not SUPPORT_FLASH1:
        return

    from .llama import llama_attn_forward, llama_varlen_attn_forward

    print_log(NO_ATTN_WEIGHTS_MSG, 'current', logging.WARNING)
    for module in model.modules():
        if type(module).__name__ in ('LlamaAttention', 'LlamaFlashAttention2',
                                     'LlamaSdpaAttention'):
            if use_varlen_attn:
                print_log('dispatch llama local attn forward', 'current')
                module.forward = types.MethodType(llama_varlen_attn_forward,
                                                  module)
            else:
                print_log('dispatch llama attn forward', 'current')
                module.forward = types.MethodType(llama_attn_forward, module)


def dispatch_llama_rmsnorm_forward(model):
    if not SUPPORT_TRITON:
        return

    from .triton_kernels import rms_norm_forward

    for module in model.modules():
        if type(module).__name__ == 'LlamaRMSNorm':
            print_log('dispatch llama rmsnorm forward', 'current')
            module.forward = types.MethodType(rms_norm_forward, module)


def dispatch_internlm_attn_forward(model, use_varlen_attn):
    if use_varlen_attn:
        assert SUPPORT_FLASH2 and SUPPORT_TRITON, \
            'flash_attn and triton is required if you want to use varlen_attn.'
    elif not SUPPORT_FLASH1:
        return

    from .internlm import internlm_attn_forward, internlm_varlen_attn_forward

    print_log(NO_ATTN_WEIGHTS_MSG, 'current', logging.WARNING)
    for module in model.modules():
        if type(module).__name__ == 'InternLMAttention':
            if use_varlen_attn:
                print_log('dispatch internlm local attn forward', 'current')
                module.forward = types.MethodType(internlm_varlen_attn_forward,
                                                  module)
            else:
                print_log('dispatch internlm attn forward', 'current')
                module.forward = types.MethodType(internlm_attn_forward,
                                                  module)


def dispatch_internlm2_attn_forward(model, use_varlen_attn):
    if use_varlen_attn:
        assert SUPPORT_FLASH2 and SUPPORT_TRITON, \
            'flash_attn and triton is required if you want to use varlen_attn.'
    elif not SUPPORT_FLASH1:
        return

    from .internlm2 import (internlm2_attn_forward,
                            internlm2_varlen_attn_forward)

    print_log(NO_ATTN_WEIGHTS_MSG, 'current', logging.WARNING)
    for module in model.modules():
        if type(module).__name__ == 'InternLM2Attention':
            if use_varlen_attn:
                print_log('dispatch internlm2 local attn forward', 'current')
                module.forward = types.MethodType(
                    internlm2_varlen_attn_forward, module)
            else:
                print_log('dispatch internlm2 attn forward', 'current')
                module.forward = types.MethodType(internlm2_attn_forward,
                                                  module)


def dispatch_internlm_rmsnorm_forward(model):
    if not SUPPORT_TRITON:
        return

    from .triton_kernels import rms_norm_forward

    for module in model.modules():
        if type(module).__name__ == 'InternLMRMSNorm':
            print_log('dispatch internlm rmsnorm forward', 'current')
            module.forward = types.MethodType(rms_norm_forward, module)


def dispatch_internlm2_rmsnorm_forward(model):
    if not SUPPORT_TRITON:
        return

    from .triton_kernels import rms_norm_forward

    for module in model.modules():
        if type(module).__name__ == 'InternLM2RMSNorm':
            print_log('dispatch internlm2 rmsnorm forward', 'current')
            module.forward = types.MethodType(rms_norm_forward, module)


def replace_internlm_rote(model):
    from .internlm import InternLMRotaryEmbedding

    def traverse(module):
        for name, child in module.named_children():
            if type(child).__name__ in (
                    'InternLMRotaryEmbedding',
                    'InternLMDynamicNTKScalingRotaryEmbedding'):
                print_log('replace internlm rope', 'current')
                dim_model = child.inv_freq.shape[0] * 2
                child_new = InternLMRotaryEmbedding(
                    dim_model, child.max_seq_len_cached).to(
                        device=child.inv_freq.device,
                        dtype=child.inv_freq.dtype)
                setattr(module, name, child_new)
            else:
                traverse(child)

    traverse(model)


def replace_internlm2_rote(model):
    from .internlm2 import InternLM2RotaryEmbedding

    rotary_base = model.config.rope_theta

    def traverse(module):
        for name, child in module.named_children():
            if type(child).__name__ in (
                    'InternLM2RotaryEmbedding',
                    'InternLM2DynamicNTKScalingRotaryEmbedding'):
                print_log('replace internlm2 rope', 'current')
                dim_model = child.inv_freq.shape[0] * 2
                child_new = InternLM2RotaryEmbedding(
                    dim_model, child.max_seq_len_cached, rotary_base).to(
                        device=child.inv_freq.device,
                        dtype=child.inv_freq.dtype)
                setattr(module, name, child_new)
            else:
                traverse(child)

    traverse(model)


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


def dispatch_yi_attn_forward(model):
    if digit_version(torch.__version__) < digit_version('2.0.0'):
        # flash attention is only supported after pytorch2.0
        return
    print_log('dispatch yi attn forward', 'current')
    print_log(NO_ATTN_WEIGHTS_MSG, 'current', logging.WARNING)
    for module in model.modules():
        if type(module).__name__ == 'YiAttention':
            module.forward = types.MethodType(yi_attn_forward, module)


def dispatch_mistral_attn_forward(model, use_varlen_attn):
    if (not SUPPORT_FLASH) or (not use_varlen_attn):
        return
    if use_varlen_attn:
        assert SUPPORT_FLASH2 and SUPPORT_TRITON, \
            'flash_attn and triton is required if you want to use varlen_attn.'

    from .mistral import mistral_varlen_attn_forward

    print_log(NO_ATTN_WEIGHTS_MSG, 'current', logging.WARNING)
    for module in model.modules():
        if type(module).__name__ in ('MistralAttention',
                                     'MistralFlashAttention2'):
            print_log('dispatch mistral local attn forward', 'current')
            module.forward = types.MethodType(mistral_varlen_attn_forward,
                                              module)


def dispatch_mistral_rmsnorm_forward(model):
    if not SUPPORT_TRITON:
        return

    from .triton_kernels import rms_norm_forward

    for module in model.modules():
        if type(module).__name__ == 'MistralRMSNorm':
            print_log('dispatch mistral rmsnorm forward', 'current')
            module.forward = types.MethodType(rms_norm_forward, module)


def replace_mistral_rote(model):
    from .mistral import MistralRotaryEmbedding

    rotary_base = model.config.rope_theta

    def traverse(module):
        for name, child in module.named_children():
            if type(child).__name__ == 'MistralRotaryEmbedding':
                print_log('replace mistral rope', 'current')
                dim_model = child.inv_freq.shape[0] * 2
                child_new = MistralRotaryEmbedding(
                    dim_model, child.max_seq_len_cached, rotary_base).to(
                        device=child.inv_freq.device,
                        dtype=child.inv_freq.dtype)
                setattr(module, name, child_new)
            else:
                traverse(child)

    traverse(model)


def dispatch_modules(model, use_varlen_attn=False):
    model_name = model.__class__.__name__.lower()
    if 'internlm2' in model_name:
        dispatch_internlm2_attn_forward(model, use_varlen_attn)
        dispatch_internlm2_rmsnorm_forward(model)
        replace_internlm2_rote(model)
    elif 'internlm' in model_name:
        dispatch_internlm_attn_forward(model, use_varlen_attn)
        dispatch_internlm_rmsnorm_forward(model)
        replace_internlm_rote(model)
    elif 'llama' in model_name:
        dispatch_llama_attn_forward(model, use_varlen_attn)
        dispatch_llama_rmsnorm_forward(model)
    elif 'baichuan' in model_name:
        dispath_baichuan2_norm_head_forward(model)
        dispath_baichuan_7b_attn_forward(model)
        dispath_baichuan_13b_attn_forward(model)
    elif 'yi' in model_name:
        dispatch_yi_attn_forward(model)
    elif 'mistral' in model_name:
        dispatch_mistral_attn_forward(model, use_varlen_attn)
        dispatch_mistral_rmsnorm_forward(model)
        replace_mistral_rote(model)


__all__ = ['dispatch_modules']
