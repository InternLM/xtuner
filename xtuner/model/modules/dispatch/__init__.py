# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
import types

import torch
import transformers
from mmengine import print_log
from mmengine.utils import digit_version
from transformers.integrations import is_deepspeed_zero3_enabled

from .baichuan import (baichuan2_norm_head_forward, baichuan_7b_attn_forward,
                       baichuan_13b_attn_forward)
from .yi import yi_attn_forward

IS_LOW_VERSION_TRANSFORMERS = digit_version(
    transformers.__version__) < digit_version('4.38')
# Transformers requires torch version >= 2.1.1 when using Torch SDPA.
# Refer to https://github.com/huggingface/transformers/blob/caa5c65db1f4db617cdac2ad667ba62edf94dd98/src/transformers/modeling_utils.py#L1611  # noqa: E501
SUPPORT_FLASH1 = digit_version(torch.__version__) >= digit_version('2.1.1')
SUPPORT_FLASH2 = False

try:
    from flash_attn import flash_attn_func  # pre-check # noqa: F401

    SUPPORT_FLASH2 = True
except ImportError:
    pass

SUPPORT_FLASH = SUPPORT_FLASH1 or SUPPORT_FLASH2

USE_TRITON_KERNEL = bool(os.getenv('USE_TRITON_KERNEL', default=0))
SUPPORT_TRITON = False
try:
    import triton  # pre-check # noqa: F401
    import triton.language as tl  # pre-check # noqa: F401
    SUPPORT_TRITON = True
except ImportError:
    if USE_TRITON_KERNEL:
        raise RuntimeError(
            'USE_TRITON_KERNEL is set to 1, but triton has not been installed.'
            ' Run `pip install triton==2.1.0` to install triton.')

NO_ATTN_WEIGHTS_MSG = (
    'Due to the implementation of the PyTorch version of flash attention, '
    'even when the `output_attentions` flag is set to True, it is not '
    'possible to return the `attn_weights`.')


def dispatch_llama_attn_forward(model, use_varlen_attn):
    if use_varlen_attn:
        assert SUPPORT_FLASH2 and SUPPORT_TRITON, \
            'flash_attn and triton is required if you want to use varlen_attn.'
    elif not SUPPORT_FLASH2:
        return

    from .llama import (llama_attn_forward, llama_attn_forward_legacy,
                        llama_varlen_attn_forward,
                        llama_varlen_attn_forward_legacy)

    print_log(NO_ATTN_WEIGHTS_MSG, 'current', logging.WARNING)
    for module in model.modules():
        # Do not need to dispatch if
        # type(module).__name__ in ('LlamaAttention', 'LlamaSdpaAttention').
        # If we set `attn_implementation` to `sdpa` or `eager` in xtuner
        # configs, we can not use varlen attn and sequence parallel.
        if type(module).__name__ == 'LlamaFlashAttention2':
            if use_varlen_attn:
                print_log('dispatch llama varlen attn forward', 'current')
                if IS_LOW_VERSION_TRANSFORMERS:
                    module.forward = types.MethodType(
                        llama_varlen_attn_forward_legacy, module)
                else:
                    module.forward = types.MethodType(
                        llama_varlen_attn_forward, module)
            else:
                print_log('dispatch llama attn forward', 'current')
                if IS_LOW_VERSION_TRANSFORMERS:
                    module.forward = types.MethodType(
                        llama_attn_forward_legacy, module)
                else:
                    module.forward = types.MethodType(llama_attn_forward,
                                                      module)


def dispatch_llama_rmsnorm_forward(model):
    if not SUPPORT_TRITON:
        return

    from .triton_kernels import rms_norm_forward

    for module in model.modules():
        if type(module).__name__ == 'LlamaRMSNorm':
            print_log('dispatch llama rmsnorm forward', 'current')
            module.forward = types.MethodType(rms_norm_forward, module)


def dispatch_phi3_attn_forward(model, use_varlen_attn):
    if use_varlen_attn:
        assert SUPPORT_FLASH2 and SUPPORT_TRITON, \
            'flash_attn and triton is required if you want to use varlen_attn.'
    elif not SUPPORT_FLASH2:
        return

    from .phi3 import phi3_attn_forward, phi3_varlen_attn_forward

    print_log(NO_ATTN_WEIGHTS_MSG, 'current', logging.WARNING)
    for module in model.modules():
        # Do not need to dispatch if
        # type(module).__name__ == 'Phi3SdpaAttention', as flash_attn is
        # required when using sequence parallel
        if type(module).__name__ in ('Phi3Attention', 'Phi3FlashAttention2'):
            if use_varlen_attn:
                print_log('dispatch phi3 varlen attn forward', 'current')
                if IS_LOW_VERSION_TRANSFORMERS:
                    raise RuntimeError(
                        'Phi-3 need transformers version >= 4.39, but got '
                        f'{transformers.__version__}')
                else:
                    module.forward = types.MethodType(phi3_varlen_attn_forward,
                                                      module)
            else:
                print_log('dispatch phi3 attn forward', 'current')
                if IS_LOW_VERSION_TRANSFORMERS:
                    raise RuntimeError(
                        'Phi-3 need transformers version >= 4.39, but got '
                        f'{transformers.__version__}')
                else:
                    module.forward = types.MethodType(phi3_attn_forward,
                                                      module)


def dispatch_phi3_rmsnorm_forward(model):
    if not SUPPORT_TRITON:
        return

    from .triton_kernels import rms_norm_forward

    for module in model.modules():
        if type(module).__name__ == 'Phi3RMSNorm':
            print_log('dispatch phi3 rmsnorm forward', 'current')
            module.forward = types.MethodType(rms_norm_forward, module)


def dispatch_internlm_attn_forward(model, use_varlen_attn):
    if use_varlen_attn:
        assert SUPPORT_FLASH2 and SUPPORT_TRITON, \
            'flash_attn and triton is required if you want to use varlen_attn.'
    elif not SUPPORT_FLASH:
        return

    from .internlm import internlm_attn_forward, internlm_varlen_attn_forward

    print_log(NO_ATTN_WEIGHTS_MSG, 'current', logging.WARNING)
    for module in model.modules():
        if type(module).__name__ == 'InternLMAttention':
            if use_varlen_attn:
                print_log('dispatch internlm varlen attn forward', 'current')
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
    elif not SUPPORT_FLASH:
        return

    from .internlm2 import (internlm2_attn_forward,
                            internlm2_varlen_attn_forward)

    print_log(NO_ATTN_WEIGHTS_MSG, 'current', logging.WARNING)
    for module in model.modules():
        # Do not need to dispatch if
        # type(module).__name__ == 'InternLM2Attention'.
        # If we set `attn_implementation` to `eager` in xtuner
        # configs, we can not use varlen attn and sequence parallel.
        if type(module).__name__ == 'InternLM2FlashAttention2':
            if use_varlen_attn:
                print_log('dispatch internlm2 varlen attn forward', 'current')
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
                    'InternLM2LinearScalingRotaryEmbedding',
                    'InternLM2DynamicNTKScalingRotaryEmbedding'):
                print_log('replace internlm2 rope', 'current')
                dim_model = child.inv_freq.shape[0] * 2
                child_new = InternLM2RotaryEmbedding(
                    dim_model, child.max_position_embeddings, rotary_base).to(
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
    if use_varlen_attn:
        assert SUPPORT_FLASH2 and SUPPORT_TRITON, \
            'flash_attn and triton is required if you want to use varlen_attn.'
    elif not SUPPORT_FLASH2:
        return

    from .mistral import mistral_attn_forward, mistral_varlen_attn_forward

    print_log(NO_ATTN_WEIGHTS_MSG, 'current', logging.WARNING)
    for module in model.modules():
        # Do not need to dispatch if
        # type(module).__name__ in ('MistralAttention', 'MistralSdpaAttention',
        #                           'MixtralAttention', 'MixtralSdpaAttention')
        # If we set `attn_implementation` to `sdpa` or `eager` in xtuner
        # configs, we can not use varlen attn and sequence parallel.
        if type(module).__name__ in ('MistralFlashAttention2',
                                     'MixtralFlashAttention2'):
            if use_varlen_attn:
                print_log('dispatch mistral varlen attn forward', 'current')
                module.forward = types.MethodType(mistral_varlen_attn_forward,
                                                  module)
            else:
                print_log('dispatch mistral attn forward', 'current')
                module.forward = types.MethodType(mistral_attn_forward, module)


def dispatch_mistral_rmsnorm_forward(model):
    if not SUPPORT_TRITON:
        return

    from .triton_kernels import rms_norm_forward

    for module in model.modules():
        if type(module).__name__ in ('MistralRMSNorm', 'MixtralRMSNorm'):
            print_log('dispatch mistral rmsnorm forward', 'current')
            module.forward = types.MethodType(rms_norm_forward, module)


def set_mixtral_moe_blocks_z3_leaf_modules(model):
    from deepspeed.utils import set_z3_leaf_modules
    from transformers.models.mixtral.modeling_mixtral import \
        MixtralSparseMoeBlock
    set_z3_leaf_modules(model, [MixtralSparseMoeBlock])


def replace_mistral_rote(model):
    from .mistral import MistralRotaryEmbedding

    rotary_base = model.config.rope_theta

    def traverse(module):
        for name, child in module.named_children():
            if type(child).__name__ in ('MistralRotaryEmbedding',
                                        'MixtralRotaryEmbedding'):
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


def dispatch_cohere_attn_forward(model, use_varlen_attn):
    if use_varlen_attn:
        raise NotImplementedError
    elif not SUPPORT_FLASH2:
        return

    from .cohere import cohere_attn_forward

    print_log(NO_ATTN_WEIGHTS_MSG, 'current', logging.WARNING)
    for module in model.modules():
        # Do not need to dispatch if
        # type(module).__name__ in ('CohereAttention', 'CohereSdpaAttention').
        # If we set `attn_implementation` to `sdpa` or `eager` in xtuner
        # configs, we can not use varlen attn and sequence parallel.
        if type(module).__name__ == 'CohereFlashAttention2':
            print_log('dispatch cohere attn forward', 'current')
            module.forward = types.MethodType(cohere_attn_forward, module)


def dispatch_cohere_layernorm_forward(model):
    from .triton_kernels import layer_norm_forward

    for module in model.modules():
        if type(module).__name__ == 'CohereLayerNorm':
            print_log('dispatch cohere layernorm forward', 'current')
            module.forward = types.MethodType(layer_norm_forward, module)


def dispatch_qwen2_attn_forward(model, use_varlen_attn):
    if use_varlen_attn:
        assert SUPPORT_FLASH2 and SUPPORT_TRITON, \
            'flash_attn and triton is required if you want to use varlen_attn.'
    elif not SUPPORT_FLASH2:
        return

    from .qwen2 import qwen2_attn_forward, qwen2_varlen_attn_forward

    print_log(NO_ATTN_WEIGHTS_MSG, 'current', logging.WARNING)
    for module in model.modules():
        # Do not need to dispatch if
        # type(module).__name__ in ('Qwen2Attention', 'Qwen2SdpaAttention',
        #                         'Qwen2MoeAttention', 'Qwen2MoeSdpaAttention')
        # If we set `attn_implementation` to `sdpa` or `eager` in xtuner
        # configs, we can not use varlen attn and sequence parallel.
        if type(module).__name__ in ('Qwen2FlashAttention2',
                                     'Qwen2MoeFlashAttention2'):
            if use_varlen_attn:
                print_log('dispatch qwen2 varlen attn forward', 'current')
                module.forward = types.MethodType(qwen2_varlen_attn_forward,
                                                  module)
            else:
                print_log('dispatch qwen2 attn forward', 'current')
                module.forward = types.MethodType(qwen2_attn_forward, module)


def dispatch_qwen2_rmsnorm_forward(model):
    if not SUPPORT_TRITON:
        return

    from .triton_kernels import rms_norm_forward

    for module in model.modules():
        if type(module).__name__ == 'Qwen2RMSNorm':
            print_log('dispatch qwen2 rmsnorm forward', 'current')
            module.forward = types.MethodType(rms_norm_forward, module)


def set_qwen_moe_blocks_z3_leaf_modules(model):
    from deepspeed.utils import set_z3_leaf_modules
    try:
        from transformers.models.qwen2_moe.modeling_qwen2_moe import \
            Qwen2MoeSparseMoeBlock
    except ImportError:
        raise ImportError('QWen moe requires transformers version at least'
                          f'4.40.0, but got {transformers.__version__}')
    set_z3_leaf_modules(model, [Qwen2MoeSparseMoeBlock])


def dispatch_modules(model, use_varlen_attn=False):
    model_name = model.__class__.__name__.lower()
    if 'internlm2' in model_name:
        dispatch_internlm2_attn_forward(model, use_varlen_attn)
        if USE_TRITON_KERNEL:
            dispatch_internlm2_rmsnorm_forward(model)
        replace_internlm2_rote(model)
    elif 'internlm' in model_name:
        dispatch_internlm_attn_forward(model, use_varlen_attn)
        if USE_TRITON_KERNEL:
            dispatch_internlm_rmsnorm_forward(model)
        replace_internlm_rote(model)
    elif 'llama' in model_name:
        dispatch_llama_attn_forward(model, use_varlen_attn)
        if USE_TRITON_KERNEL:
            dispatch_llama_rmsnorm_forward(model)
    elif 'phi3' in model_name:
        dispatch_phi3_attn_forward(model, use_varlen_attn)
        if USE_TRITON_KERNEL:
            dispatch_phi3_rmsnorm_forward(model)
    elif 'baichuan' in model_name:
        dispath_baichuan2_norm_head_forward(model)
        dispath_baichuan_7b_attn_forward(model)
        dispath_baichuan_13b_attn_forward(model)
    elif 'yi' in model_name:
        dispatch_yi_attn_forward(model)
    elif ('mistral' in model_name) or ('mixtral' in model_name):
        dispatch_mistral_attn_forward(model, use_varlen_attn)
        if USE_TRITON_KERNEL:
            dispatch_mistral_rmsnorm_forward(model)
        replace_mistral_rote(model)
        if 'moe' in model_name and is_deepspeed_zero3_enabled():
            set_mixtral_moe_blocks_z3_leaf_modules(model)
    elif 'cohere' in model_name:
        dispatch_cohere_attn_forward(model, use_varlen_attn)
        dispatch_cohere_layernorm_forward(model)
    elif 'qwen2' in model_name:
        # qwen2 and qwen2moe
        dispatch_qwen2_attn_forward(model, use_varlen_attn)
        if USE_TRITON_KERNEL:
            dispatch_qwen2_rmsnorm_forward(model)
        if 'moe' in model_name and is_deepspeed_zero3_enabled():
            set_qwen_moe_blocks_z3_leaf_modules(model)


__all__ = ['dispatch_modules']
